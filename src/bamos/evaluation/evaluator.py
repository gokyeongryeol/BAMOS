"""Evaluator: GT metrics separated from the planner (spec 10). Never feeds values back to the policy."""
from __future__ import annotations

import numpy as np

from bamos.belief.state import BeliefState
from bamos.environment.base import Environment
from bamos.observation.types import DetectionStatus, PlannerObservation


class Evaluator:
    def __init__(self, env: Environment, eval_cfg: dict, candidate_centers: np.ndarray):
        self.env = env
        self.gt = env.ground_truth()
        self.cfg = eval_cfg
        self.candidate_centers = candidate_centers
        d = np.linalg.norm(candidate_centers - self.gt.target_center[None, :], axis=1)
        self.gt_bin = int(np.argmin(d))
        self.gt_bin_offset_m = float(d[self.gt_bin])
        self.first_detection_source: str | None = None
        self.first_true_detection_step: int | None = None

    def localization(self, state: BeliefState, center=None) -> dict:
        """center: the declared location when it is not the belief's MAP (stop.rule = first_det)."""
        pi = state.location.probs_loc()
        map_c = state.map_center() if center is None else np.asarray(center, dtype=np.float64)
        err = float(np.linalg.norm(map_c - self.gt.target_center))
        mass = float(pi[self.gt_bin])
        return {
            "gt_center_error_m": err,
            "gt_bin_posterior_mass": mass,
            "gt_bin_nll": float(-np.log(mass)) if mass > 0 else float("inf"),
            "gt_map_center_correct": err <= self.cfg["max_center_error_m"],
        }

    def view_quality(self, obs: PlannerObservation) -> dict:
        q = self.env.gt_quality_of_observation(obs)
        return {
            "gt_visible_fraction": float(q["visible_fraction"]),
            "gt_area_px": int(q["area_px"]),
            "gt_normalized_area": float(q["normalized_area"]),
            "gt_truncated": bool(q["truncated"]),
            "gt_detected": bool(q["gt_detected"]),
        }

    def note_detection(self, obs: PlannerObservation, step: int):
        q = self.env.gt_quality_of_observation(obs)
        if obs.detection == DetectionStatus.DET and self.first_detection_source is None:
            self.first_detection_source = obs.camera_id
        if q["gt_detected"] and obs.detection == DetectionStatus.DET and "wrong_instance_id" not in obs.meta and self.first_true_detection_step is None:
            self.first_true_detection_step = step

    def actual_success(self, declared: bool, state: BeliefState, obs: PlannerObservation, center=None) -> tuple[bool, bool]:
        """(actual success, wrong-instance stop) for a FOUND declaration on the current robot view: the MAP is within
        max_center_error_m of the target. How well the declared view shows the target is not part of the test (the
        model only localises); the view thresholds define which episodes are solvable (environment/episodes.py)."""
        if not declared:
            return False, False
        loc = self.localization(state, center)
        wrong_instance = "wrong_instance_id" in obs.meta
        ok = loc["gt_center_error_m"] <= self.cfg["max_center_error_m"] and not wrong_instance
        return bool(ok), bool(wrong_instance)
