"""Observation data interfaces (spec 2)."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from bamos.mapping.camera import CameraModel


class DetectionStatus(str, Enum):
    DET = "DET"
    NO_DET = "NO_DET"
    MISSING = "MISSING"  # detector/sensor failure: not negative evidence


@dataclass
class PlannerObservation:
    """Everything the planner/belief updater is allowed to see for one capture."""

    camera_id: str
    camera_type: str  # "robot"
    capture_id: str
    camera: CameraModel
    world_from_camera: np.ndarray  # 4x4, optical frame
    depth: np.ndarray | None  # [H, W] float, depth_kind given by camera.depth_kind; non-finite/<=0 invalid
    target_mask: np.ndarray | None  # [H, W] bool, None if detector did not run / failed
    detection: DetectionStatus
    depth_valid: bool = True
    rgb: np.ndarray | None = None
    pose_id: int | None = None  # candidate pool id for robot views
    meta: dict[str, Any] = field(default_factory=dict)
    # sensor_detector.open_set_multi_mask: every kept instance of a DET, the selected one first (None: one mask)
    instance_masks: list[np.ndarray] | None = None

    @property
    def observed_mask_area(self) -> int:
        if self.target_mask is None:
            return 0
        return int(self.target_mask.sum())

    @property
    def normalized_mask_area(self) -> float:
        return self.observed_mask_area / float(self.camera.n_pixels)
