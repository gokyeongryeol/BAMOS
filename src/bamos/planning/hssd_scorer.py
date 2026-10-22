"""The Scorer of planning/scorer.py for a house-sized candidate set: r is evaluated on the cells within reach of the camera.

The depth sensor returns nothing beyond camera.far, so phi = 0 for every hypothesis whose centre lies beyond far along the
optical axis, and r = eta_f there exactly. No centre farther than `reach` from the camera position (the farthest centre
whose square silhouette can still overlap the image within the depth range, plus the largest cell offset) has phi > 0, so
each camera position evaluates r on the cells within reach only, and P+ and I(Z;X) add the constant out-of-range part in
closed form. Nothing is approximated: the values are the Scorer's. (The same bookkeeping as active-grounding-objectnav's
objectnav/likelihood.py.)
"""
from __future__ import annotations

import numpy as np
import torch

from bamos.belief.state import BeliefState
from bamos.mapping.camera import CameraModel
from bamos.observation.likelihood import DetectionLikelihood
from bamos.planning.candidates import CandidateBatch
from bamos.planning.scorer import PredictiveSnapshot, ScoreBatch, Scorer, ScorerConfig
from bamos.planning.utility import LOG2, binary_entropy_t, ig_loc
from bamos.visibility.engine import VisibilityEngine


def reach_of(camera: CameraModel, sizes: np.ndarray, offsets: np.ndarray) -> float:
    """A hypothesis has phi > 0 only if its centre has optical depth z in [near, far] and its square silhouette (half side
    s/2 at that depth) overlaps the image on each axis: |x| <= z tx + s/2, |y| <= z ty + s/2 (tx, ty the tangents of the
    half fovs). So no centre farther than hypot(far, hypot(far tx + s_max/2, far ty + s_max/2)), plus the largest cell
    offset, is ever in view."""
    tx, ty = (camera.width / 2) / camera.fx, (camera.height / 2) / camera.fy
    far, h = float(camera.far), 0.5 * float(np.max(sizes))
    return float(np.hypot(far, np.hypot(far * tx + h, far * ty + h))) + float(np.linalg.norm(offsets, axis=1).max())


class HssdScorer(Scorer):
    def __init__(self, engine: VisibilityEngine, det_lik: DetectionLikelihood, cfg: ScorerConfig, reach_m: float):
        super().__init__(engine, det_lik, cfg)
        self.reach = float(reach_m)
        self.eta_f = float(det_lik.false_positive_rate)

    def in_range(self, position: np.ndarray) -> torch.Tensor:
        p = torch.as_tensor(np.asarray(position, dtype=np.float64), device=self.engine.device)
        c = self.engine.candidate_centers.to(torch.float64)
        return torch.nonzero(torch.linalg.norm(c - p[None], dim=1) <= self.reach).squeeze(1)

    def _r_sub_full(self, state: BeliefState, T: np.ndarray) -> torch.Tensor:
        """[Q, n_C * n_S] r at each of the cell's points for one pose, eta_f outside the range."""
        T = np.asarray(T, dtype=np.float64).reshape(1, 4, 4)
        idx = self.in_range(T[0, :3, 3])
        n_c, n_s, Q = self.engine.candidate_centers.shape[0], len(self.sizes), len(self.engine.offsets)
        r_sub = torch.full((Q, n_c, n_s), self.eta_f, dtype=torch.float64, device=self.engine.device)
        if idx.numel():
            r = self.detection_likelihood_sub(state, T[:, :3, 3], T, None, idx)  # [Q, 1, n * n_S]
            r_sub[:, idx, :] = r[:, 0].reshape(Q, -1, n_s).to(torch.float64)
        return r_sub.reshape(Q, -1)

    def score(self, state: BeliefState, cand: CandidateBatch, need_ig: bool = True, chunk: int = 512) -> tuple[ScoreBatch, None]:
        """Scorer.score with r evaluated per camera position on the cells within reach (module docstring)."""
        n_s = len(self.sizes)
        P = cand.n
        ig = np.zeros(P)
        p_plus = np.full(P, self.eta_f)
        if need_ig and P:
            pi = self.engine.to_tensor(state.location.probs()).to(torch.float64).reshape(-1, n_s)
            ef = torch.tensor(self.eta_f, dtype=torch.float64, device=pi.device)
            h_ef = binary_entropy_t(ef)
            pos, Ts = cand.positions, cand.world_from_camera
            uniq, inv = np.unique(np.round(pos, 6), axis=0, return_inverse=True)
            inv = np.asarray(inv).reshape(-1)
            for u in range(len(uniq)):
                sel = np.nonzero(inv == u)[0]
                idx = self.in_range(pos[sel[0]])
                if idx.numel() == 0:
                    continue
                pin = pi[idx].reshape(-1)
                m_in = pin.sum()
                for lo in range(0, len(sel), chunk):
                    s = sel[lo:lo + chunk]
                    r = self.detection_likelihood(state, pos[s], Ts[s], None, idx).to(torch.float64)  # [Ps, n * n_S]
                    pp = (r * pin[None]).sum(dim=1) + self.eta_f * (1.0 - m_in)
                    g = binary_entropy_t(pp) - (pin[None] * binary_entropy_t(r)).sum(dim=1) - (1.0 - m_in) * h_ef
                    ig[s] = g.cpu().numpy()
                    p_plus[s] = pp.cpu().numpy()
        neg = ig < -self.cfg.ig_negative_tolerance
        diag = {"n_negative_ig": int(neg.sum()), "min_ig": float(ig.min()) if P else 0.0,
                "max_ig_norm": float(ig.max() / LOG2) if P else 0.0}
        ig = np.where(np.abs(ig) <= self.cfg.ig_negative_tolerance, np.maximum(ig, 0.0), ig)
        motion = np.zeros(P) if cand.path_cost is None else np.minimum(cand.path_cost / self.cfg.motion_scale_m, 1.0)
        return ScoreBatch(cand.pose_ids, ig, ig / LOG2, p_plus, state.location.entropy(), motion, {}, diag), None

    def snapshot(self, sb: ScoreBatch, r, k: int, state: BeliefState, cand: CandidateBatch) -> PredictiveSnapshot:
        r_sub = self._r_sub_full(state, cand.world_from_camera[k])
        r_k = r_sub.mean(dim=0)
        pi = self.engine.to_tensor(state.location.probs()).to(torch.float64)
        return PredictiveSnapshot(int(sb.pose_ids[k]), r_k.cpu().numpy(), float((r_k * pi).sum()), state.version,
                                  float(sb.ig_raw[k]), float(sb.ig_normalized[k]), r_sub.cpu().numpy())

    def predict_pose(self, state: BeliefState, world_from_camera: np.ndarray, pose_id: int = -1, camera=None) -> PredictiveSnapshot:
        if camera is not None and camera != self.engine.camera:
            raise ValueError("HssdScorer: the robot camera only (its reach is the robot camera's)")
        r_sub = self._r_sub_full(state, world_from_camera)
        r = r_sub.mean(dim=0)
        pi = self.engine.to_tensor(state.location.probs()).to(torch.float64)
        ig, p_plus = ig_loc(pi, r[None])
        return PredictiveSnapshot(pose_id, r.cpu().numpy(), float(p_plus[0]), state.version, float(ig[0]),
                                  float(ig[0]) / LOG2, r_sub.cpu().numpy())

    def present_value(self, state: BeliefState, cand: CandidateBatch, v: int, min_size_weight: float = 0.05):
        return self.present_value_poses(state.location.probs(), cand.positions, cand.world_from_camera, v, min_size_weight,
                                        only_v=True)
