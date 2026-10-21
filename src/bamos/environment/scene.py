"""Parametric scene description shared by the analytic renderer, the Isaac adapter and the evaluator.

Primitives: axis-aligned-in-local-frame boxes with a yaw rotation about z, and z-axis cylinders. Every SceneObject
owns an instance id; the searchable target is exactly one object with role 'target'.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np


@dataclass
class Box:
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    yaw: float = 0.0
    kind: str = "box"

    def corners(self) -> np.ndarray:
        c = np.asarray(self.center)
        h = 0.5 * np.asarray(self.size)
        signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float64)
        local = signs * h
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        R = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        return local @ R.T + c

    def aabb(self) -> tuple[np.ndarray, np.ndarray]:
        cs = self.corners()
        return cs.min(axis=0), cs.max(axis=0)


@dataclass
class Cylinder:
    center: tuple[float, float, float]  # center of the cylinder (mid-height)
    radius: float
    height: float
    kind: str = "cylinder"

    def aabb(self) -> tuple[np.ndarray, np.ndarray]:
        c = np.asarray(self.center)
        r, h = self.radius, 0.5 * self.height
        return c - np.array([r, r, h]), c + np.array([r, r, h])


@dataclass
class Sphere:
    center: tuple[float, float, float]
    radius: float
    kind: str = "sphere"

    def aabb(self) -> tuple[np.ndarray, np.ndarray]:
        c = np.asarray(self.center)
        return c - self.radius, c + self.radius


Primitive = Box | Cylinder | Sphere


@dataclass
class SceneObject:
    instance_id: int
    name: str
    role: str  # floor | wall | anchor | clutter | target
    prims: list[Primitive] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def aabb(self) -> tuple[np.ndarray, np.ndarray]:
        los, his = zip(*[p.aabb() for p in self.prims])
        return np.min(np.stack(los), axis=0), np.max(np.stack(his), axis=0)

    def center(self) -> np.ndarray:
        lo, hi = self.aabb()
        return 0.5 * (lo + hi)


@dataclass
class FixedCameraSpec:
    camera_id: str
    world_from_camera: np.ndarray  # 4x4 optical


@dataclass
class SceneDescription:
    """Planner-visible layout information plus evaluator-only fields (kept separate by naming)."""

    layout_id: str
    archetype: str
    workspace_lo: tuple[float, float, float]
    workspace_hi: tuple[float, float, float]
    objects: list[SceneObject]
    fixed_cameras: list[FixedCameraSpec]
    robot_z: float
    robot_start_xy_yaw_pitch: tuple[float, float, float, float]
    # evaluator-only
    gt_target_instance_id: int
    gt_target_center: tuple[float, float, float]
    gt_target_extent: tuple[float, float, float]
    gt_target_type: str
    gt_placement_mode: str
    gt_anchor_name: str
    meta: dict[str, Any] = field(default_factory=dict)

    def target(self) -> SceneObject:
        return next(o for o in self.objects if o.instance_id == self.gt_target_instance_id)

    def obstacles_for_navigation(self) -> list[SceneObject]:
        """Everything except floor (the target is an object on an anchor, but it is also a physical obstacle).

        The navigation oracle uses only footprints; it never exposes which object is the target.
        """
        return [o for o in self.objects if o.role != "floor"]

    def to_manifest(self) -> dict:
        d = {
            "layout_id": self.layout_id,
            "archetype": self.archetype,
            "workspace_lo": list(self.workspace_lo),
            "workspace_hi": list(self.workspace_hi),
            "robot_z": self.robot_z,
            "robot_start_xy_yaw_pitch": list(self.robot_start_xy_yaw_pitch),
            "fixed_cameras": [{"camera_id": c.camera_id, "world_from_camera": np.asarray(c.world_from_camera).tolist()} for c in self.fixed_cameras],
            "objects": [
                {"instance_id": o.instance_id, "name": o.name, "role": o.role, "prims": [asdict(p) for p in o.prims], "meta": o.meta}
                for o in self.objects
            ],
            "meta": self.meta,
            "evaluator_only": {
                "gt_target_instance_id": self.gt_target_instance_id,
                "gt_target_center": list(self.gt_target_center),
                "gt_target_extent": list(self.gt_target_extent),
                "gt_target_type": self.gt_target_type,
                "gt_placement_mode": self.gt_placement_mode,
                "gt_anchor_name": self.gt_anchor_name,
            },
        }
        return d
