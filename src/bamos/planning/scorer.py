"""Candidate scoring: apparent area a, detection likelihood r, P+ and I(Z;X) per pose (spec 6, 7, 8.1, 9).

Naming: the local variable `alpha` is the specification's area `a_{t,q}(v,s)`; the Greek name is kept only
because `a` would read badly next to the array arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from bamos.belief.state import BeliefState
from bamos.observation.likelihood import DetectionLikelihood
from bamos.planning.candidates import CandidateBatch
from bamos.planning.utility import LOG2, ig_loc
from bamos.visibility.engine import VisibilityEngine


@dataclass
class ScoreBatch:
    pose_ids: np.ndarray
    ig_raw: np.ndarray  # nats
    ig_normalized: np.ndarray  # / log 2
    p_plus: np.ndarray
    entropy: float
    motion_cost: np.ndarray  # normalized path cost min(c / motion_scale_m, 1); logged only, never scored
    extra: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)


@dataclass
class PredictiveSnapshot:
    pose_id: int
    r: np.ndarray  # [n_C * n_S], the pre-observation likelihood the update must use (spec 8.3): the cell average
    p_plus: float
    belief_version: int
    ig_raw: float
    ig_normalized: float
    r_sub: np.ndarray | None = None  # [Q, n_C * n_S] r at each of the cell's quadrature points (spec 7.3), for kappa


@dataclass
class ScorerConfig:
    ig_negative_tolerance: float = 1e-9
    motion_scale_m: float = 5.0  # only used for the logged normalized path cost
    h_area0: float = 0.0001  # a0: detection threshold on the area a (spec 7.1)
    h_slope: float = 3.0  # k: slope of the soft threshold in log-area (spec 7.1)


class Scorer:
    def __init__(self, engine: VisibilityEngine, det_lik: DetectionLikelihood, cfg: ScorerConfig | None = None):
        self.engine = engine
        self.det = det_lik
        self.cfg = cfg or ScorerConfig()
        # [n_S] hypothesis size scales s (spec 3), set by the runner
        self.sizes: np.ndarray | None = None
        # the voxel belief b_t (spec 4), set by the runner; None: no occlusion gate (the ablation)
        self.voxel_belief = None

    def area_per_m2(self, world_from_camera: np.ndarray, camera, offset: int = 0, idx: torch.Tensor | None = None) -> torch.Tensor:
        """[P, n_C] factor fx fy / (W H max(z, z_min)^2); times s^2 it is the apparent area a_hat (spec 6).

        idx, here and below: the candidates to evaluate (n_C becomes len(idx)); None: all."""
        eng = self.engine
        cam = eng.camera if camera is None else camera
        T = eng.to_tensor(world_from_camera)
        if T.dim() == 2:
            T = T[None]
        R = T[:, :3, :3]
        t = T[:, :3, 3]
        cc = eng.centres_at(offset, idx)  # [n_C,3]
        z = torch.einsum("pj,nj->pn", R[:, :, 2], cc) - (R[:, :, 2] * t).sum(dim=1)[:, None]  # [P,n_C] = R^T(c-t) z-row
        z = torch.clamp(z, min=cam.near)
        return cam.fx * cam.fy / (cam.n_pixels * z * z)

    def projected_area(self, world_from_camera: np.ndarray, camera, offset: int = 0, idx: torch.Tensor | None = None) -> torch.Tensor:
        """[P, n_C, n_S] a_hat_q(v, s) = fx fy s^2 / (W H z^2). The body has no orientation, so the silhouette
        does not depend on the viewing direction (spec 6)."""
        apu = self.area_per_m2(world_from_camera, camera, offset, idx)
        s2 = torch.as_tensor(np.asarray(self.sizes, dtype=np.float64) ** 2, dtype=apu.dtype, device=apu.device)
        return apu[:, :, None] * s2[None, None, :]

    def _hill(self, a: torch.Tensor, a0: float, k: float) -> torch.Tensor:
        """Soft threshold in log-area: sigma(k ln(a / a0)) = a^k / (a^k + a0^k) (spec 7.1).

        Exactly 0 at a = 0 and 1 in saturation, so the detection likelihood's floor and ceiling are exactly
        eta_f and eta_f + (1 - eta_f) eta_t (spec 7.2). k is dimensionless, so the curve depends on a / a0 only.
        Evaluated as the logistic of the log-ratio: a^k and a0^k both underflow to 0 for a fitted steep k (an
        oracle's hard threshold), and their ratio would then be 0 / 0.
        """
        la = torch.log(torch.clamp(a, min=1e-300))
        return torch.where(a > 0, torch.sigmoid(k * (la - float(np.log(a0)))), torch.zeros_like(a))

    def area(self, state: BeliefState, positions: np.ndarray, world_from_camera: np.ndarray,
             camera=None, offset: int = 0, idx: torch.Tensor | None = None) -> torch.Tensor:
        """[P, n_C * n_S] a_{t,q}(v, s) = phi_q(v, s) a_hat_q(v, s) (spec 6): the apparent area times the share of the
        hypothesis' silhouette inside the image, for the body centred at x_v + offsets[offset]; occlusion is not modelled."""
        alpha = self.engine.candidate_coverage(world_from_camera, self.sizes, camera, offset, idx) \
            * self.projected_area(world_from_camera, camera, offset, idx)
        return alpha.reshape(alpha.shape[0], -1)

    def detection_likelihood_at(self, state: BeliefState, positions: np.ndarray, world_from_camera: np.ndarray,
                                camera=None, offset: int = 0, idx: torch.Tensor | None = None) -> torch.Tensor:
        """[P, n_C * n_S] r = eta_f + (1 - eta_f) eta_t T_bar h(a) (spec 7.2) for the body centred at
        x_v + offsets[offset]: the voxel belief's expected front-face transmittance T_bar gates the detectable part."""
        alpha = self.area(state, positions, world_from_camera, camera, offset, idx)
        f, gain = self.det.false_positive_rate, self.det.gain
        h = self._hill(alpha, self.cfg.h_area0, self.cfg.h_slope)
        if self.voxel_belief is not None:
            T = self.engine.transmittance(positions, self.sizes, self.voxel_belief, offset, idx)
            h = h * T.reshape(T.shape[0], -1)
        return f + (1.0 - f) * gain * h

    def detection_likelihood_sub(self, state: BeliefState, positions: np.ndarray, world_from_camera: np.ndarray,
                                 camera=None, idx: torch.Tensor | None = None) -> torch.Tensor:
        """[Q, P, n_C * n_S] r at each of the cell's quadrature points (spec 7.3)."""
        return torch.stack([self.detection_likelihood_at(state, positions, world_from_camera, camera, q, idx)
                            for q in range(len(self.engine.offsets))])

    def detection_likelihood(self, state: BeliefState, positions: np.ndarray, world_from_camera: np.ndarray,
                             camera=None, idx: torch.Tensor | None = None) -> torch.Tensor:
        """[P, n_C * n_S] r_{t,q}(v, s): the probability of a DET given the target in cell v with size s, the mean of r over
        the cell's quadrature points (spec 7.3), accumulated so the [Q, P, N] stack is never held for all poses."""
        r = None
        for q in range(len(self.engine.offsets)):
            rq = self.detection_likelihood_at(state, positions, world_from_camera, camera, q, idx)
            r = rq if r is None else r + rq
        return r / len(self.engine.offsets)

    def present_value(self, state: BeliefState, cand: CandidateBatch, v: int, min_size_weight: float = 0.05):
        """The present view after a declaration (spec 10): for each candidate pose, (V [P], whole [P]).

        V(q) = sum_s pi(s | v) (1/8) sum_j phi_{q,j}(v, s) a_hat_{q,j}(v, s) T_bar_{q,j}(v, s), the expected apparent area of
        the MAP cell's body that is inside the image and not hidden, averaged over the cell's points (spec 7.3) and the
        cell's size posterior. whole(q): the body's silhouette lies inside the image at every one of the cell's points for
        every size with pi(s | v) >= min_size_weight, i.e. the view shows it untruncated wherever in the cell it is."""
        return self.present_value_poses(state.location.probs(), cand.positions, cand.world_from_camera, v, min_size_weight)

    def present_value_poses(self, pi_joint: np.ndarray, positions: np.ndarray, world_from_camera: np.ndarray, v: int,
                            min_size_weight: float = 0.05, only_v: bool = False):
        """present_value for poses given directly. only_v: evaluate the terms on cell v alone (the same values, for a
        candidate set too large to evaluate whole for every pose)."""
        n_s = len(self.sizes)
        ps = np.asarray(pi_joint, dtype=np.float64).reshape(-1, n_s)[v]
        w = ps / max(ps.sum(), 1e-300)
        wt = self.engine.to_tensor(w)
        keep = torch.as_tensor(w >= min_size_weight, device=wt.device)
        T = world_from_camera
        idx, col = (torch.tensor([v]), 0) if only_v else (None, v)
        V = None
        whole = None
        for q in range(len(self.engine.offsets)):
            phi = self.engine.candidate_coverage(T, self.sizes, None, q, idx)[:, col, :]  # [P, n_S]
            a_hat = self.projected_area(T, None, q, idx)[:, col, :]
            tb = self.engine.transmittance(positions, self.sizes, self.voxel_belief, q, idx)[:, col, :] if self.voxel_belief is not None \
                else torch.ones_like(phi)
            vq = (phi * a_hat * tb * wt[None, :]).sum(dim=1)
            ok = torch.all((phi >= 1.0 - 1e-9) | ~keep[None, :], dim=1)
            V = vq if V is None else V + vq
            whole = ok if whole is None else whole & ok
        return (V / len(self.engine.offsets)).cpu().numpy(), whole.cpu().numpy()

    def score(self, state: BeliefState, cand: CandidateBatch, need_ig: bool = True, chunk: int = 512) -> tuple[ScoreBatch, None]:
        """Returns (ScoreBatch, None): I(Z;X) and P+ of every candidate pose, computed over chunks of poses so the
        [P, n_C*n_S] r of a large room is never held at once; snapshot() recomputes r for the chosen pose.
        need_ig=False (view rules that read no target belief) skips them: the ScoreBatch carries zeros."""
        pi = self.engine.to_tensor(state.location.probs())
        if need_ig:
            igs, pps = [], []
            for lo in range(0, cand.n, chunk):
                r = self.detection_likelihood(state, cand.positions[lo:lo + chunk], cand.world_from_camera[lo:lo + chunk], None)
                ig_c, pp_c = ig_loc(pi, r)
                igs.append(ig_c)
                pps.append(pp_c)
                del r
            ig = torch.cat(igs) if igs else torch.zeros(0, dtype=pi.dtype, device=pi.device)
            p_plus = torch.cat(pps) if pps else torch.zeros(0, dtype=pi.dtype, device=pi.device)
        else:
            ig = torch.zeros(cand.n, dtype=pi.dtype, device=pi.device)
            p_plus = torch.zeros(cand.n, dtype=pi.dtype, device=pi.device)
        ig_np = ig.cpu().numpy()
        neg = ig_np < -self.cfg.ig_negative_tolerance
        diag = {"n_negative_ig": int(neg.sum()), "min_ig": float(ig_np.min()) if len(ig_np) else 0.0, "max_ig_norm": float(ig_np.max() / LOG2) if len(ig_np) else 0.0}
        ig_np = np.where(np.abs(ig_np) <= self.cfg.ig_negative_tolerance, np.maximum(ig_np, 0.0), ig_np)
        motion = np.zeros(cand.n) if cand.path_cost is None else np.minimum(cand.path_cost / self.cfg.motion_scale_m, 1.0)
        return ScoreBatch(cand.pose_ids, ig_np, ig_np / LOG2, p_plus.cpu().numpy(), state.location.entropy(),
                          motion, {}, diag), None

    def snapshot(self, sb: ScoreBatch, r, k: int, state: BeliefState, cand: CandidateBatch) -> PredictiveSnapshot:
        """The pre-observation snapshot of the chosen pose k: r and its per-point values r_sub, recomputed for that pose
        (score() keeps no [P, N] r). P+ is recomputed with it, so it is right also when score() skipped the IG."""
        r_sub_t = self.detection_likelihood_sub(state, cand.positions[k : k + 1], cand.world_from_camera[k : k + 1], None)[:, 0]
        r_k = r_sub_t.mean(dim=0)
        pi = self.engine.to_tensor(state.location.probs())
        p_plus = float((r_k * pi).sum())
        return PredictiveSnapshot(int(sb.pose_ids[k]), r_k.cpu().numpy(), p_plus, state.version,
                                  float(sb.ig_raw[k]), float(sb.ig_normalized[k]), r_sub_t.cpu().numpy())

    def predict_pose(self, state: BeliefState, world_from_camera: np.ndarray, pose_id: int = -1, camera=None) -> PredictiveSnapshot:
        """Pre-observation snapshot for a prescribed pose (the robot start view)."""
        pi = self.engine.to_tensor(state.location.probs())
        T = np.asarray(world_from_camera)[None] if np.asarray(world_from_camera).ndim == 2 else np.asarray(world_from_camera)
        r_sub = self.detection_likelihood_sub(state, T[:, :3, 3], T, camera)
        r = r_sub.mean(dim=0)
        ig, p_plus = ig_loc(pi, r)
        return PredictiveSnapshot(pose_id, r[0].cpu().numpy(), float(p_plus[0]), state.version, float(ig[0]),
                                  float(ig[0]) / LOG2, r_sub[:, 0].cpu().numpy())
