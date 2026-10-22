"""Step / episode records and JSONL writers."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o


@dataclass
class StepRecord:
    experiment_id: str
    episode_id: str
    layout_id: str
    method_id: str
    config_hash: str
    code_commit: str
    layout_seed: int
    episode_seed: int
    sensor_seed: int
    planner_seed: int
    observation_step: int
    active_robot_step: int
    initialization_or_active: str
    camera_id: str
    camera_type: str
    capture_id: str
    selected_pose_id: int
    camera_pose: list
    fixed_initial_view_count: int
    robot_initial_view_count: int
    active_robot_view_count: int
    path_length: float
    yaw_change_rad: float
    pitch_change_rad: float
    candidate_count: int
    ig_raw: float
    ig_normalized: float
    motion_cost: float
    selected_utility: float
    utility_tie_count: int
    pre_observation_detection_probability: float
    detection_status: str
    observation_validity: bool
    posterior_entropy: float
    posterior_max: float
    map_center: list
    observed_mask_area: int
    observed_normalized_area: float
    rho_current_map: float
    declared_stop: bool
    termination_reason: str
    numerical_diagnostics: dict
    # evaluator-only (gt_ prefix)
    gt_center_error_m: float
    gt_bin_posterior_mass: float
    gt_bin_nll: float
    gt_visible_fraction: float
    gt_area_px: int
    gt_normalized_area: float
    gt_truncated: bool
    gt_detected: bool
    gt_actual_success: bool
    gt_wrong_instance_stop: bool
    gt_first_detection_source: str
    gt_map_center_correct: bool
    planning_latency_s: float
    update_latency_s: float  # the belief update (spec 8.3)
    extra: dict = field(default_factory=dict)


class JsonlWriter:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a")

    def write(self, rec: Any):
        d = asdict(rec) if hasattr(rec, "__dataclass_fields__") else rec
        self._f.write(json.dumps(_jsonable(d)) + "\n")
        self._f.flush()

    def close(self):
        self._f.close()
