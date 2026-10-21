"""Environment interface shared by the analytic environment and the Isaac client."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bamos.environment.scene import SceneDescription
from bamos.mapping.camera import CameraModel
from bamos.observation.types import PlannerObservation
from bamos.planning.candidates import CandidatePool


@dataclass
class EvaluatorGroundTruth:
    """Evaluator-only access. Never handed to the planner."""

    target_center: np.ndarray
    target_instance_id: int
    target_extent: np.ndarray
    target_type: str
    placement_mode: str
    anchor_name: str
    reachable_quality_pose_ids: np.ndarray  # pose ids in the final pool meeting the task quality thresholds
    best_quality_pose_id: int
    best_quality: dict
    fixed_camera_visibility: dict  # {}: no fixed cameras (kept in the manifest's evaluator_only)
    initial_visibility_class: str  # "hidden": the oracle detects the target from no initial view (difficulty)


class Environment:
    """Sensor + navigation oracle. Subclasses implement rendering."""

    scene: SceneDescription
    robot_camera: CameraModel
    pool: CandidatePool
    robot_start_pose_id: int

    def observe_robot(self, pose_id: int, rng: np.random.Generator, step_index: int) -> PlannerObservation:
        raise NotImplementedError

    def path_costs(self, from_pose_id: int, to_pose_ids: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def ground_truth(self) -> EvaluatorGroundTruth:
        raise NotImplementedError

    def gt_quality_of_observation(self, obs: PlannerObservation) -> dict:
        raise NotImplementedError

    def close(self):
        pass
