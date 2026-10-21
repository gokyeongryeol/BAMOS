"""Closed-loop episode runner (spec 11)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from bamos.belief.location_belief import LocationBelief
from bamos.belief.state import BeliefState
from bamos.environment.base import Environment
from bamos.evaluation.evaluator import Evaluator
from bamos.evaluation.records import StepRecord
from bamos.mapping.camera import CameraModel, angle_diff, yaw_pitch_from_transform
from bamos.mapping.grid import VoxelGrid
from bamos.observation.likelihood import DetectionLikelihood
from bamos.observation.types import DetectionStatus, PlannerObservation
from bamos.planning.candidates import CandidateBatch
from bamos.planning.policies import Policy, PolicyConfig, method_flags
from bamos.planning.scorer import PredictiveSnapshot, Scorer, ScorerConfig
from bamos.planning.stopping import StopConfig, StopTracker, evaluate_stop, success_probability
from bamos.visibility.engine import EngineConfig, VisibilityEngine, cell_offsets


def make_grid(cfg: dict, scene) -> VoxelGrid:
    vs = cfg["mapping"]["voxel_size_m"]
    off = cfg["mapping"]["grid_offset_m"]
    lo = np.asarray(scene.workspace_lo) - off
    hi = np.asarray(scene.workspace_hi) + vs
    return VoxelGrid.from_bounds(lo, hi, vs)


def make_candidates(cfg: dict, grid: VoxelGrid, scene) -> np.ndarray:
    """Fixed candidate center set C: workspace interior (with wall clearance), plausible height band, strided lattice."""
    tc = cfg["target"]
    s = int(tc["center_stride_voxels"])
    ijk = grid.unravel(np.arange(grid.n_voxels))
    c = grid.centers(ijk)
    lo = np.asarray(scene.workspace_lo) + tc["candidate_wall_clearance_m"]
    hi = np.asarray(scene.workspace_hi) - tc["candidate_wall_clearance_m"]
    z0, z1 = tc["candidate_z_range_m"]
    m = (ijk[:, 0] % s == 0) & (ijk[:, 1] % s == 0) & (ijk[:, 2] % s == 0)
    m &= (c[:, 0] >= lo[0]) & (c[:, 0] <= hi[0]) & (c[:, 1] >= lo[1]) & (c[:, 1] <= hi[1]) & (c[:, 2] >= z0) & (c[:, 2] <= z1)
    return np.nonzero(m)[0]


def camera_from_cfg(c: dict) -> CameraModel:
    return CameraModel.from_hfov(int(c["width"]), int(c["height"]), np.deg2rad(float(c["hfov_deg"])), float(c["near_m"]), float(c["far_m"]))


@dataclass
class EpisodeContext:
    experiment_id: str
    episode_id: str
    layout_id: str
    layout_seed: int
    episode_seed: int
    method_id: str
    planner_seed: int
    sensor_seed: int
    config_hash: str
    code_commit: str = "n/a"


@dataclass
class EpisodeResult:
    ctx: EpisodeContext
    termination_reason: str
    declared_stop: bool
    declared_stop_step: int | None
    actual_success: bool
    wrong_instance_stop: bool
    fixed_initial_view_count: int
    robot_initial_view_count: int
    active_robot_view_count: int
    total_view_count: int
    total_path_length_m: float
    final_gt: dict
    first_detection_source: str | None
    first_true_detection_observation_step: int | None
    first_confidence_step: int | None
    first_correct_localization_step: int | None
    initial_visibility_class: str
    target_type: str
    placement_mode: str
    runtime_s: float
    planning_latency_p50_s: float
    planning_latency_p95_s: float
    n_model_inconsistency: int
    n_negative_ig_steps: int
    extra: dict = field(default_factory=dict)


class EpisodeRunner:
    def __init__(self, env: Environment, cfg: dict, ctx: EpisodeContext, step_writer=None, array_store: dict | None = None):
        self.env = env
        self.cfg = cfg
        self.ctx = ctx
        self.writer = step_writer
        self.arrays = array_store
        self.method = ctx.method_id
        method_flags(ctx.method_id)  # BAMOS is the only method this build accepts
        self.rng_planner = np.random.default_rng(ctx.planner_seed)
        self.rng_sensor = np.random.default_rng(ctx.sensor_seed)
        scene = env.scene
        self.grid = self.make_grid()
        self.candidates = self.make_candidates()
        # size hypotheses S (spec 3): size scales s of a convex body with no orientation, s^2 its mean silhouette area
        self.sizes = self.size_grid()  # [n_S]
        loc = LocationBelief.uniform(self.candidates, len(self.sizes))
        self.state = BeliefState(self.grid, loc)
        dl = cfg["detection_likelihood"]
        self.det_lik = DetectionLikelihood(dl["detection_gain"], dl["false_positive_rate"])
        ex = cfg["experiment"]
        from bamos.environment.analytic_renderer import set_cast_backend

        set_cast_backend(ex["device"])
        mp = cfg["mapping"]
        self.engine = VisibilityEngine(self.grid, env.robot_camera, self.candidates, self.engine_config())
        # spec 7.3: each cell's likelihood is the average over points inside the cell (a centre uniform in the cell)
        step = float(mp["voxel_size_m"]) * int(cfg["target"]["center_stride_voxels"])
        self.engine.offsets = cell_offsets(step)
        self.cell_step = step
        pl = cfg["planner"]
        ab = cfg.get("ablation", {})
        self.scorer = self.make_scorer(ScorerConfig(
            pl["ig_negative_tolerance"], pl["motion_scale_m"],
            float(cfg["detection_likelihood"]["h_area0"]), float(cfg["detection_likelihood"]["h_slope"])))
        self.scorer.sizes = self.sizes
        # the voxel belief b_t (spec 4): Bernoulli occupancy on its own grid over the same box; the ablation
        # leaves r without the occlusion gate
        self.voxel_belief = None
        if bool(cfg.get("ablation", {}).get("use_voxel_belief", True)):
            self.voxel_belief = self.make_voxel_belief()
        self.scorer.voxel_belief = self.voxel_belief
        self.use_kappa = bool(ab.get("use_kappa", True))
        # ablation (outline D2): NO_DET leaves pi untouched, as MISSING does (spec 8.3)
        self.use_nodet = bool(ab.get("use_nodet", True))
        # kappa (spec 8.2): m_q = x_v + beta s n and Sigma = (lambda s)^2 I, both dimensionless in s
        self.kappa_lambda = float(pl["kappa_lambda"])
        self.kappa_beta = float(pl["kappa_beta"])
        # u of spec 8.2: a false positive's centroid is uniform over the workspace
        self.workspace_volume = float(np.prod(np.asarray(scene.workspace_hi) - np.asarray(scene.workspace_lo)))
        self.utility_mode = str(pl.get("utility_mode", "ig"))
        self.policy = Policy(PolicyConfig(self.method, pl["tie_tolerance"], self.utility_mode))
        # "coverage" needs to know which candidate cells have already been brought into view
        self.covered = np.zeros(len(self.candidates), dtype=bool) if self.utility_mode == "coverage" else None
        # "scene_coverage": the probability that each candidate cell has been seen through the occupancy belief so far
        self.seen = np.zeros(len(self.candidates)) if self.utility_mode == "scene_coverage" else None
        st = cfg["stop"]
        self.stop_cfg = StopConfig(float(st.get("success_radius_m", 0.2)), float(st.get("tau_pi", 0.5)),
                                   str(st.get("mode", "joint")), int(st.get("confirmations", 1)))
        self.act_on_stop = self.stop_cfg.mode != "none"
        self.stop_tracker = StopTracker(self.stop_cfg)
        # spec 10 present view: after S_hat = 1 the robot moves to the view that shows the MAP cell best and declares if
        # that view confirms it (DET, pi(v_hat) >= tau_pi, the MAP still there); otherwise the declaration is withdrawn
        # and the search resumes. Besides the DET and the MAP in place, it must confirm p_succ, the probability that
        # declaring the MAP cell succeeds, >= tau_pi (formulation §10)
        self.present_views = int(st.get("present_views", 1))
        if self.present_views < 1:
            raise ValueError(f"stop.present_views={self.present_views} must be >= 1")
        # stop.rule: "belief" (spec 10) or "first_det" (2026-10-06, the baselines without a target belief): the first
        # robot DET declares the candidate cell nearest the mask's back-projected centroid
        self.stop_rule = str(st.get("rule", "belief"))
        if self.stop_rule not in ("belief", "first_det"):
            raise ValueError(f"stop.rule={self.stop_rule!r} must be 'belief' or 'first_det'")
        self.declared_center = None
        self.present_left = 0
        self.pending_center = None
        self.final_declared = False
        self._stop_info: dict = {}
        self.evaluator = self.make_evaluator()
        self.n_inconsistent = 0
        self.n_negative_ig = 0
        self.latencies: list[float] = []
        self.first_confidence_step = None
        self.first_correct_step = None
        self.total_path = 0.0

    # ---------------------------------------------------------------------------------------------- set-up hooks
    # The Isaac benchmark's choices; an environment with its own world (HSSD, planning/hssd_runner.py) overrides them.
    def make_grid(self) -> VoxelGrid:
        return make_grid(self.cfg, self.env.scene)

    def make_candidates(self) -> np.ndarray:
        return make_candidates(self.cfg, self.grid, self.env.scene)

    def size_grid(self) -> np.ndarray:
        return np.asarray(self.cfg["planner"]["size_grid_m"], dtype=np.float64)

    def engine_config(self) -> EngineConfig:
        ex, mp = self.cfg["experiment"], self.cfg["mapping"]
        return EngineConfig(device=ex["device"], dtype=ex["dtype"], ray_margin_m=float(mp["occupancy_ray_margin_m"]))

    def make_scorer(self, scfg: ScorerConfig) -> Scorer:
        return Scorer(self.engine, self.det_lik, scfg)

    def make_voxel_belief(self):
        from bamos.belief.voxel_belief import VoxelBelief

        ex, mp = self.cfg["experiment"], self.cfg["mapping"]
        hi = self.grid.lower + np.asarray(self.grid.shape) * self.grid.voxel_size
        p0, q_occ, q_free = (float(x) for x in mp["occupancy_model"])
        vb = VoxelBelief(VoxelGrid.from_bounds(self.grid.lower, hi, float(mp["occupancy_voxel_m"])),
                         p0, q_occ, q_free, ex["device"], int(mp["occupancy_pixel_stride"]))
        vb.near_hits = bool(mp.get("near_hits", False))
        if mp.get("occupancy_source", "belief") == "ground_truth":
            # a perfect scene model (the world-model comparison): the true geometry, opaque, the target left out
            from bamos.belief.voxel_belief import GroundTruthOccupancy
            vb = GroundTruthOccupancy.from_scene(vb.grid, self.env.scene, ex["device"])
        return vb

    def make_evaluator(self) -> Evaluator:
        return Evaluator(self.env, self.cfg["evaluation"], self.state.candidate_centers())

    def feasible_poses(self, current_pose_id: int) -> np.ndarray:
        """[P] bool: the poses the next view may take (before the revisit rule)."""
        return self.env.pool.feasible.copy()

    def after_observation(self, obs: PlannerObservation) -> None:
        """Called once per robot frame, after integrate()."""

    # ---------------------------------------------------------------------------------------------- helpers
    def integrate(self, obs: PlannerObservation, snap: PredictiveSnapshot) -> dict:
        """Update pi with the pre-observation snapshot r and, on DET, the mask centroid (spec 8). Returns diagnostics.

        A DET with a centroid c has the likelihood eta_f u(c) + (r - eta_f) N(c; m, (lambda s)^2 I): r splits into a
        false positive, whose centroid says nothing about the target (uniform u over the workspace), and a true
        detection, whose centroid is the Gaussian of spec 8.2. Integrated over c it is r, as a DET without a
        usable centroid (fewer than n_min valid pixels, or the D3 ablation) contributes."""
        t0 = time.time()
        diag = {"model_inconsistency": False}
        st = self.state
        # location update: DET/NO_DET only, snapshot r from before this observation
        updates_pi = obs.detection == DetectionStatus.DET or (obs.detection == DetectionStatus.NO_DET and self.use_nodet)
        if updates_pi:
            det = obs.detection == DetectionStatus.DET
            if snap.belief_version != st.version:
                raise RuntimeError("snapshot belief version mismatch: likelihood must be pre-observation")
            loglik = None
            if self.use_kappa and det and obs.target_mask is not None and obs.target_mask.any() and obs.depth is not None:
                loglik, cinfo = self.mask_centroid_likelihood(obs)
            if loglik is None:
                info = st.location.update(snap.r, det)  # the event alone: r (DET) or 1 - r (NO_DET)
            else:
                # the true-detection branch averaged over the cell's points o_q (spec 7.3): mean_q (r_q - eta_f) N_q(c)
                r_sub = np.asarray(snap.r_sub if snap.r_sub is not None else np.asarray(snap.r)[None], dtype=np.float64)
                eta_f = float(self.det_lik.false_positive_rate)
                with np.errstate(divide="ignore"):
                    log_fp = np.log(eta_f) - np.log(self.workspace_volume)
                    log_true = np.logaddexp.reduce(np.log(np.clip(r_sub - eta_f, 0.0, None)) + loglik, axis=0) - np.log(len(r_sub))
                logf = np.logaddexp(log_fp, log_true)
                info = st.location.update_with_log_likelihood(logf)
                post = st.location.probs()
                # posterior probability that this detection was a false positive (diagnostic)
                p_fp = float(np.sum(post * np.exp(log_fp - logf))) if info.get("model_inconsistency") is False else float("nan")
                diag["maskloc"] = {**cinfo, "p_false_positive": p_fp}
            if info["model_inconsistency"]:
                diag["model_inconsistency"] = True
                self.n_inconsistent += 1
        # the voxel belief takes the frame's depth after pi has used the pre-observation r (spec 4)
        if self.voxel_belief is not None and obs.depth is not None and obs.depth_valid:
            vinfo = self.voxel_belief.update(obs.depth, obs.target_mask if obs.detection == DetectionStatus.DET else None,
                                             obs.camera, obs.world_from_camera, obs.meta.get("depth_ignore"))
            diag["voxel_belief"] = vinfo
        st.step += 1
        st.version += 1
        if not np.all(np.isfinite(st.location.log_pi[np.isfinite(st.location.log_pi)])) or np.isnan(st.location.log_pi).any():
            diag["numerical_failure"] = True
        diag["update_latency_s"] = time.time() - t0
        return diag

    def mask_centroid(self, obs: PlannerObservation):
        """The world centroid of the reported mask's valid back-projected depth pixels, or None (fewer than 3). With
        several instances (open_set_multi_mask) the selected one's, the first."""
        m = obs.instance_masks[0] if obs.instance_masks else obs.target_mask
        if m is None or obs.depth is None or not np.asarray(m).any():
            return None
        vs, us = np.nonzero(m)
        d = obs.depth[vs, us]
        ok = np.isfinite(d) & (d > 0) & (d <= obs.camera.far)
        if ok.sum() < 3:
            return None
        pc = obs.camera.backproject(np.stack([us[ok], vs[ok]], 1).astype(np.float64), d[ok])
        return (pc @ obs.world_from_camera[:3, :3].T + obs.world_from_camera[:3, 3]).mean(axis=0)

    def mask_centroid_likelihood(self, obs: PlannerObservation):
        """[Q, n_C * n_S] log kappa_{t+1}(v, s) of spec 8.2 for the body centred at each of the cell's points
        x_v + o_q (spec 7.3), flattened over the joint (v, s), or None if the mask has fewer than n_min = 3 valid depth
        pixels (then kappa = 1, i.e. no update).

        c is the world centroid of the back-projected valid mask pixels, the centroid of the VISIBLE surface:
        it sits beta s towards the camera from the body centre, and what is left spreads over the body itself,
        hence the isotropic (lambda s)^2. Returned as its log; integrate() mixes it with the false-positive branch.

        Several instances (open_set_multi_mask): at most one is the target, each with probability 1/M, the others false
        positives uniform over the workspace; the common factor u^(M-1) cancels, so kappa's Gaussian becomes the mean
        over the instances' centroids (those with n_min valid pixels). M = 1 is the one-mask case.
        """
        if obs.instance_masks and len(obs.instance_masks) > 1:
            parts = [self._centroid_loglik(obs, m) for m in obs.instance_masks]
            ok = [p for p in parts if p[0] is not None]
            if not ok:
                return None, {"n_instances": len(parts), "n_valid": 0}
            stack = np.stack([p[0] for p in ok])  # [M, Q, n_C * n_S]
            return np.logaddexp.reduce(stack, axis=0) - np.log(len(ok)), \
                {"centroids": [p[1]["centroid"] for p in ok], "n_instances": len(parts), "n_used": len(ok)}
        return self._centroid_loglik(obs, obs.target_mask)

    def _centroid_loglik(self, obs: PlannerObservation, mask: np.ndarray):
        """[Q, n_C * n_S] log N(c; x_v + o_q + beta s n, (lambda s)^2 I) for the centroid c of one mask, or None."""
        vs, us = np.nonzero(mask)
        d = obs.depth[vs, us]
        ok = np.isfinite(d) & (d > 0) & (d <= obs.camera.far)
        if ok.sum() < 3:
            return None, {"n_mask_px": int(len(us)), "n_valid": int(ok.sum())}
        pc = obs.camera.backproject(np.stack([us[ok], vs[ok]], 1).astype(np.float64), d[ok])
        pw = pc @ obs.world_from_camera[:3, :3].T + obs.world_from_camera[:3, 3]
        c = pw.mean(axis=0)
        sz = self.sizes  # [n_S]
        out = []
        for o in self.engine.offsets:
            cc = self.state.candidate_centers() + o[None, :]
            nvec = obs.world_from_camera[:3, 3][None, :] - cc
            nvec = nvec / np.maximum(np.linalg.norm(nvec, axis=1, keepdims=True), 1e-9)  # [n_C,3], direction only
            m = cc[:, None, :] + self.kappa_beta * sz[None, :, None] * nvec[:, None, :]
            dvec = m - c[None, None, :]  # [n_C,n_S,3]
            var = np.broadcast_to(((self.kappa_lambda * sz) ** 2)[None, :], (cc.shape[0], len(sz)))  # [n_C,n_S]
            maha = np.sum(dvec * dvec, axis=2) / var
            out.append((-1.5 * np.log(2.0 * np.pi) - 1.5 * np.log(var) - 0.5 * maha).reshape(-1))  # log N(c; m, var I)
        return np.stack(out), {"centroid": c.tolist(), "n_valid": int(ok.sum())}

    def expected_s2(self) -> float:
        """E[S^2] under the size marginal of pi."""
        return float(self.state.location.probs_size() @ self.sizes**2)

    def _record(self, phase: str, obs: PlannerObservation, snap: PredictiveSnapshot, sel_info: dict, cand_count: int, path_len: float, prev_T: np.ndarray | None, stop, diag: dict, term: str, counts: tuple[int, int, int], active_step: int, plan_latency: float, gt_success: bool, wrong_inst: bool):
        st = self.state
        loc = self.evaluator.localization(st, self.declared_center)
        vq = self.evaluator.view_quality(obs)
        yaw, pitch = yaw_pitch_from_transform(obs.world_from_camera)
        dyaw = dpitch = 0.0
        if prev_T is not None and obs.camera_type == "robot":
            py, pp = yaw_pitch_from_transform(prev_T)
            dyaw, dpitch = abs(angle_diff(yaw, py)), abs(pitch - pp)
        if loc["gt_map_center_correct"] and self.first_correct_step is None:
            self.first_correct_step = st.step
        if st.location.max_prob() >= self.stop_cfg.tau_pi and self.first_confidence_step is None:
            self.first_confidence_step = st.step
        rec = StepRecord(
            self.ctx.experiment_id, self.ctx.episode_id, self.ctx.layout_id, self.method, self.ctx.config_hash, self.ctx.code_commit,
            self.ctx.layout_seed, self.ctx.episode_seed, self.ctx.sensor_seed, self.ctx.planner_seed,
            st.step, active_step, phase, obs.camera_id, obs.camera_type, obs.capture_id, -1 if obs.pose_id is None else int(obs.pose_id),
            obs.world_from_camera.tolist(), counts[0], counts[1], counts[2], float(path_len), float(dyaw), float(dpitch), int(cand_count),
            snap.ig_raw, snap.ig_normalized, float(sel_info.get("motion_cost", 0.0)),
            float(sel_info.get("utility", float("nan"))), int(sel_info.get("tie_count", 0)), snap.p_plus, obs.detection.value, bool(obs.depth_valid),
            st.location.entropy(), st.location.max_prob(), st.map_center().tolist(), obs.observed_mask_area, obs.normalized_mask_area,
            float(stop.pi_map) if stop is not None else float("nan"), bool(stop.declared) if stop is not None else False, term, diag,
            loc["gt_center_error_m"], loc["gt_bin_posterior_mass"], loc["gt_bin_nll"], vq["gt_visible_fraction"], vq["gt_area_px"], vq["gt_normalized_area"], vq["gt_truncated"], vq["gt_detected"],
            bool(gt_success), bool(wrong_inst), self.evaluator.first_detection_source or "", loc["gt_map_center_correct"], float(plan_latency), float(diag.get("update_latency_s", 0.0)),
            {
                "stop_components": None if stop is None else {"loc_ok": stop.loc_ok, "det_ok": stop.det_ok, "pi_map": stop.pi_map,
                                                              **self._stop_info},
                "noise_applied": obs.meta.get("noise_applied", {}),
                "gt_bin_offset_m": self.evaluator.gt_bin_offset_m,
                # evaluator only: the target's unoccluded silhouette in this frame (> 0 means it is in view,
                # so a small gt_visible_fraction then means occluded rather than out of the frustum)
                "gt_iso_area_px": int(self.env.gt_quality_of_observation(obs).get("iso_area_px", 0)),
                "selected_scores": sel_info.get("selected_scores", {}),
            },
        )
        if self.writer is not None:
            self.writer.write(rec)
        if self.arrays is not None:
            self.arrays.setdefault("pi", []).append(st.location.probs().astype(np.float32))
        return rec

    def _stop_step(self, obs: PlannerObservation, stop, was_present: bool):
        """The declaration of spec 10 with its present view. Returns (declare now, the stop decision as logged).

        S_hat = 1 (StopTracker) opens a present phase at the MAP centre; the next present_views views are chosen by
        present_value(); one that sees a DET with pi(v_hat) >= tau_pi and the MAP within tau of where the phase opened
        declares; if none does, the phase is withdrawn and a later S_hat = 1 opens a new one. After a declaration
        (stop.mode none: the run goes on) no phase opens again."""
        from dataclasses import replace
        st = self.state
        if self.stop_rule == "first_det":
            declare = False
            info = {"s_hat": False, "present_view": False, "confirmed": False, "withdrawn": False, "opens_present": False}
            if not self.final_declared and obs.camera_type == "robot" and obs.detection == DetectionStatus.DET:
                c = self.mask_centroid(obs)
                if c is not None:
                    cc = st.candidate_centers()
                    self.declared_center = cc[int(np.argmin(np.linalg.norm(cc - c[None, :], axis=1)))]
                    declare = self.final_declared = True
                    info["first_det_centroid"] = c.tolist()
            self._stop_info = info
            return declare, replace(stop, declared=declare)
        s_hat = bool(self.stop_tracker.update(st, stop))
        p_succ = success_probability(st, self.stop_cfg.success_radius_m, self.cell_step)
        info = {"s_hat": s_hat, "present_view": was_present, "confirmed": False, "withdrawn": False, "opens_present": False,
                "p_succ": p_succ}
        declare = False
        if not self.final_declared:
            if was_present:
                same = self.pending_center is not None and \
                    np.linalg.norm(np.asarray(st.map_center()) - self.pending_center) <= self.stop_cfg.success_radius_m + 1e-9
                if stop.det_ok and p_succ >= self.stop_cfg.tau_pi and same:
                    declare = True
                    info["confirmed"] = True
                    self.present_left = 0
                else:
                    self.present_left -= 1
                    if self.present_left <= 0:
                        info["withdrawn"] = True
                        self.pending_center = None
            if not declare and s_hat and self.present_left <= 0:
                self.present_left = self.present_views
                self.pending_center = np.asarray(st.map_center(), dtype=float)
                info["opens_present"] = True
        if declare:
            self.final_declared = True
        self._stop_info = info
        return declare, replace(stop, declared=declare)

    # ---------------------------------------------------------------------------------------------- main loop
    def run(self) -> EpisodeResult:
        t_start = time.time()
        env, st, cfg = self.env, self.state, self.cfg
        ex = cfg["experiment"]
        n_fixed = 0  # no fixed cameras (the benchmark's K = 0); kept in the counts the records carry
        n_robot_init = 0
        n_active = 0
        term = "running"
        declared = False
        declared_step = None
        success = False
        wrong_inst = False
        last_obs = None
        current_pose_id = env.robot_start_pose_id
        current_T = env.pool.world_from_camera[current_pose_id]

        def counts():
            return (n_fixed, n_robot_init, n_active)

        # ---- robot start view
        if ex["include_robot_start_view"]:
            snap = self.scorer.predict_pose(st, current_T, current_pose_id)
            obs = env.observe_robot(current_pose_id, self.rng_sensor, 0)
            diag = self.integrate(obs, snap)
            self.after_observation(obs)
            n_robot_init += 1
            st.visited_pose_ids.add(int(current_pose_id))
            self.evaluator.note_detection(obs, st.step)
            decl, stop = self._stop_step(obs, evaluate_stop(self.stop_cfg, st, obs), False)
            if decl and self.act_on_stop:
                declared, declared_step = True, st.step
                success, wrong_inst = self.evaluator.actual_success(True, st, obs, self.declared_center)
                term = "FOUND"
            if diag.get("model_inconsistency"):
                term = "model_inconsistency"
            self._record("initialization", obs, snap, {}, 0, 0.0, None, stop, diag, term, counts(), 0, 0.0, success, wrong_inst)
            last_obs = obs
        # ---- active loop
        budget = int(ex["additional_robot_view_budget"])
        for k in range(budget):
            if term != "running":
                break
            feas = self.feasible_poses(current_pose_id)
            if ex["forbid_exact_revisit"]:
                for v in st.visited_pose_ids:
                    feas[v] = False
            if not feas.any():
                term = "no_candidate"
                break
            cand = CandidateBatch(env.pool, np.nonzero(feas)[0])
            cand.path_cost = env.path_costs(current_pose_id, cand.pose_ids)
            reach = np.isfinite(cand.path_cost)
            if not reach.any():
                term = "no_candidate"
                break
            cand = CandidateBatch(env.pool, cand.idx[reach], cand.path_cost[reach])
            t0 = time.time()
            present = self.present_left > 0
            if present:
                # the present view (spec 10): the MAP cell shown whole and as large and unhidden as possible
                V, whole = self.scorer.present_value(st, cand, int(st.location.map_index()))
                pool = np.nonzero(whole)[0] if whole.any() else np.arange(cand.n)
                best = float(V[pool].max())
                ties = pool[V[pool] >= best * (1.0 - 1e-9)]
                sel = int(ties[np.argmin(cand.path_cost[ties])])
                snap = self.scorer.predict_pose(st, cand.world_from_camera[sel], int(cand.pose_ids[sel]))
                plan_latency = time.time() - t0
                self.latencies.append(plan_latency)
                sel_info = {"motion_cost": float(min(cand.path_cost[sel] / float(cfg["planner"]["motion_scale_m"]), 1.0)),
                            "utility": float(V[sel]), "tie_count": int(len(ties)),
                            "selected_scores": {"present_value": float(V[sel]), "whole": bool(whole[sel]), "n_whole": int(whole.sum())}}
            else:
                # the view rules that read no target belief need no I(Z;X) over the candidates
                sb, r = self.scorer.score(st, cand, need_ig=self.utility_mode == "ig")
            if not present and self.seen is not None:
                eng = self.scorer.engine

                def vis_of(rows):
                    # [len(rows), n_C] the probability that the cell centre's smallest body is in view and unhidden;
                    # all sizes, as r uses them (the transmittance cache is keyed by position only); the smallest body's
                    fov = (eng.candidate_fov(cand.world_from_camera[rows]) > 0.5).to(torch.float64)
                    T = eng.transmittance(cand.positions[rows], self.sizes, self.voxel_belief, 0)[:, :, 0].to(torch.float64)
                    return fov * T

                unseen = torch.as_tensor(1.0 - self.seen, dtype=torch.float64, device=eng.device)
                # in pose chunks: [P, n_C] at once does not fit a house-sized candidate set
                chunk = max(1, int(2**27 // max(len(self.seen), 1)))
                sc = np.empty(cand.n)
                for c0 in range(0, cand.n, chunk):
                    rows = np.arange(c0, min(c0 + chunk, cand.n))
                    sc[rows] = (vis_of(rows) * unseen[None, :]).sum(dim=1).cpu().numpy()
                sb.extra["scene_coverage"] = sc / len(self.seen)
            if not present and self.covered is not None:
                eng = self.scorer.engine
                fresh = torch.as_tensor(~self.covered, device=eng.device)
                # in pose chunks, as scene_coverage: [P, n_C] at once does not fit a house-sized candidate set
                chunk = max(1, int(2**27 // max(len(self.covered), 1)))
                cov = np.empty(cand.n)
                for c0 in range(0, cand.n, chunk):
                    rows = np.arange(c0, min(c0 + chunk, cand.n))
                    f_ = eng.candidate_fov(cand.world_from_camera[rows])
                    cov[rows] = ((f_ > 0.5) & fresh[None, :]).sum(dim=1).cpu().numpy()
                sb.extra["coverage"] = cov / len(self.covered)
            if not present:
                if sb.diagnostics.get("n_negative_ig", 0) > 0:
                    self.n_negative_ig += 1
                sel, sel_info = self.policy.select(sb, cand, self.rng_planner)
                plan_latency = time.time() - t0
                self.latencies.append(plan_latency)
                sel_info["motion_cost"] = float(sb.motion_cost[sel])
                sel_info["selected_scores"] = {"ig_norm": float(sb.ig_normalized[sel]), "p_plus": float(sb.p_plus[sel]), **{k: float(v[sel]) for k, v in sb.extra.items()}}
                snap = self.scorer.snapshot(sb, r, sel, st, cand)
                if self.arrays is not None:
                    self.arrays.setdefault("scores", []).append({"pose_ids": cand.pose_ids.astype(np.int32), "ig_norm": sb.ig_normalized.astype(np.float32), "p_plus": sb.p_plus.astype(np.float32), "utility": self.policy.utility(sb).astype(np.float32)})
            pose_id = int(cand.pose_ids[sel])
            path_len = float(cand.path_cost[sel])
            self.total_path += path_len
            prev_T = current_T
            if self.covered is not None and not present:
                self.covered |= (self.scorer.engine.candidate_fov(cand.world_from_camera[[sel]])[0] > 0.5).cpu().numpy()
            if self.seen is not None and not present:
                self.seen = 1.0 - (1.0 - self.seen) * (1.0 - vis_of(np.array([sel]))[0].cpu().numpy())
            obs = env.observe_robot(pose_id, self.rng_sensor, k + 1)
            if obs.depth is None or not obs.depth_valid:
                term = "sensor_failure"
            diag = self.integrate(obs, snap)
            self.after_observation(obs)
            n_active += 1
            current_pose_id = pose_id
            current_T = env.pool.world_from_camera[pose_id]
            st.visited_pose_ids.add(pose_id)
            self.evaluator.note_detection(obs, st.step)
            decl, stop = self._stop_step(obs, evaluate_stop(self.stop_cfg, st, obs), present)
            if decl and self.act_on_stop and term == "running":
                declared, declared_step = True, st.step
                success, wrong_inst = self.evaluator.actual_success(True, st, obs, self.declared_center)
                term = "FOUND"
            if diag.get("model_inconsistency") and term == "running":
                term = "model_inconsistency"
            if diag.get("numerical_failure") and term == "running":
                term = "numerical_failure"
            if term == "running" and k == budget - 1:
                term = "budget_exhausted"
            self._record("active", obs, snap, sel_info, cand.n, path_len, prev_T, stop, diag, term, counts(), k + 1, plan_latency, success, wrong_inst)
            last_obs = obs
        if term == "running":
            term = "budget_exhausted" if budget == 0 else term
        final_gt = {**self.evaluator.localization(st), **(self.evaluator.view_quality(last_obs) if last_obs is not None else {})}
        gt = env.ground_truth()
        lat = np.array(self.latencies) if self.latencies else np.array([0.0])
        return EpisodeResult(
            self.ctx, term, declared, declared_step, success, wrong_inst, n_fixed, n_robot_init, n_active, n_fixed + n_robot_init + n_active, self.total_path, final_gt,
            self.evaluator.first_detection_source, self.evaluator.first_true_detection_step, self.first_confidence_step, self.first_correct_step,
            gt.initial_visibility_class, gt.target_type, gt.placement_mode, time.time() - t_start, float(np.percentile(lat, 50)), float(np.percentile(lat, 95)),
            self.n_inconsistent, self.n_negative_ig,
            {"n_candidates_C": int(len(self.candidates)), "n_voxels": int(self.grid.n_voxels), "pool_feasible": int(env.pool.feasible.sum()), "gt_bin_offset_m": self.evaluator.gt_bin_offset_m},
        )
