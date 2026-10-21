"""Evaluator for HSSD ObjectNav (several goal instances, each a box): the Evaluator's metrics against the nearest goal.

- localization: the declared (or MAP) cell centre's distance to the nearest goal box (0 inside one); the posterior mass of
  the cells within max_center_error_m of a goal box; correct when the distance is within max_center_error_m.
- actual_success (the declaration): the declared centre within max_center_error_m of a goal box. Logged beside it:
  `viewpoint_success`, the declaring camera within viewpoint_success_m horizontally of a goal's box, on its floor (its
  navmesh point no higher than the box top and no more than 1.5 m below the box bottom): the benchmark's view points lie
  within 1 m of the goal's box.
Never feeds anything back to the planner.
"""
from __future__ import annotations

import numpy as np

from bamos.belief.state import BeliefState
from bamos.evaluation.evaluator import Evaluator
from bamos.observation.types import PlannerObservation


def box_distance(points: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """[N, G] Euclidean distance of each point to each axis-aligned box (0 inside); lo, hi [G, 3]."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 1, 3)
    d = np.maximum(np.maximum(lo[None] - p, 0.0), p - hi[None])
    return np.linalg.norm(d, axis=2)


class ObjectNavEvaluator(Evaluator):
    def __init__(self, env, eval_cfg: dict, candidate_centers: np.ndarray, viewpoint_success_m: float, sensor_height_m: float):
        self.env = env
        self.gt = env.ground_truth()
        self.cfg = eval_cfg
        self.candidate_centers = candidate_centers
        self.lo = np.stack([g["lo"] for g in self.gt.goals])
        self.hi = np.stack([g["hi"] for g in self.gt.goals])
        self.r = float(eval_cfg["max_center_error_m"])
        d = box_distance(candidate_centers, self.lo, self.hi).min(axis=1)
        self.near_goal = d <= self.r + 1e-9
        self.gt_bin = int(np.argmin(d))
        self.gt_bin_offset_m = float(d[self.gt_bin])
        self.viewpoint_m = float(viewpoint_success_m)
        self.sensor_height = float(sensor_height_m)
        self.first_detection_source: str | None = None
        self.first_true_detection_step: int | None = None
        self.viewpoint_success = False
        self.declared_goal: int | None = None

    def localization(self, state: BeliefState, center=None) -> dict:
        pi = state.location.probs_loc()
        c = state.map_center() if center is None else np.asarray(center, dtype=np.float64)
        err = float(box_distance(c, self.lo, self.hi).min())
        mass = float(pi[self.near_goal].sum())
        return {"gt_center_error_m": err, "gt_bin_posterior_mass": mass,
                "gt_bin_nll": float(-np.log(mass)) if mass > 0 else float("inf"),
                "gt_map_center_correct": err <= self.r}

    def camera_near_goal(self, world_from_camera: np.ndarray) -> bool:
        p = np.asarray(world_from_camera)[:3, 3]
        dxy = np.linalg.norm(np.maximum(np.maximum(self.lo[:, :2] - p[None, :2], 0.0), p[None, :2] - self.hi[:, :2]), axis=1)
        base = p[2] - self.sensor_height
        same_floor = (base <= self.hi[:, 2]) & (base >= self.lo[:, 2] - 1.5)
        return bool(np.any((dxy <= self.viewpoint_m) & same_floor))

    def actual_success(self, declared: bool, state: BeliefState, obs: PlannerObservation, center=None) -> tuple[bool, bool]:
        if not declared:
            return False, False
        loc = self.localization(state, center)
        c = state.map_center() if center is None else np.asarray(center, dtype=np.float64)
        self.declared_goal = int(np.argmin(box_distance(c, self.lo, self.hi)[0]))
        self.viewpoint_success = self.camera_near_goal(obs.world_from_camera)
        return bool(loc["gt_center_error_m"] <= self.r), False
