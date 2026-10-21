"""Discrete movable-camera action space (the pose set Q_t of spec 2).

a = (x, y, psi, theta) in X x Y x Psi x Theta;  q(a) = (x, y, z_robot, roll=0, yaw=psi, pitch=theta).
Pose ids are tied to the 4-axis grid index. All methods share the same pool and feasibility filtering.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bamos.mapping.camera import world_from_camera


@dataclass
class CandidatePool:
    pose_ids: np.ndarray  # [P] int (== row index)
    grid_index: np.ndarray  # [P,4] ix, iy, iyaw, ipitch
    positions: np.ndarray  # [P,3]
    yaw: np.ndarray  # [P]
    pitch: np.ndarray  # [P]
    world_from_camera: np.ndarray  # [P,4,4]
    feasible: np.ndarray  # [P] bool (clearance + reachability)
    x_values: np.ndarray
    y_values: np.ndarray
    yaw_values: np.ndarray
    pitch_values: np.ndarray
    z_robot: float

    @property
    def n(self) -> int:
        return len(self.pose_ids)

    @staticmethod
    def build(x_values, y_values, yaw_values, pitch_values, z_robot: float, feasible_xy: np.ndarray | None = None) -> "CandidatePool":
        x_values = np.asarray(x_values, dtype=np.float64)
        y_values = np.asarray(y_values, dtype=np.float64)
        yaw_values = np.asarray(yaw_values, dtype=np.float64)
        pitch_values = np.asarray(pitch_values, dtype=np.float64)
        # yaw 0 and 2pi must not both appear
        wrapped = np.mod(yaw_values, 2 * np.pi)
        if len(np.unique(np.round(wrapped, 9))) != len(yaw_values):
            raise ValueError("duplicate yaw values modulo 2pi")
        ix, iy, iw, ip = np.meshgrid(np.arange(len(x_values)), np.arange(len(y_values)), np.arange(len(yaw_values)), np.arange(len(pitch_values)), indexing="ij")
        gi = np.stack([ix.ravel(), iy.ravel(), iw.ravel(), ip.ravel()], axis=1)
        P = len(gi)
        pos = np.stack([x_values[gi[:, 0]], y_values[gi[:, 1]], np.full(P, z_robot)], axis=1)
        yaw = yaw_values[gi[:, 2]]
        pitch = pitch_values[gi[:, 3]]
        T = np.stack([world_from_camera(pos[i], yaw[i], pitch[i]) for i in range(P)])
        feas = np.ones(P, dtype=bool) if feasible_xy is None else feasible_xy[gi[:, 0], gi[:, 1]]
        return CandidatePool(np.arange(P), gi, pos, yaw, pitch, T, feas, x_values, y_values, yaw_values, pitch_values, float(z_robot))


@dataclass
class CandidateBatch:
    """A view on a subset of the pool (feasible, unvisited poses at planning time) plus execution path cost."""

    pool: CandidatePool
    idx: np.ndarray  # indices into pool
    path_cost: np.ndarray | None = None  # [len(idx)] meters from current pose

    @property
    def pose_ids(self) -> np.ndarray:
        return self.pool.pose_ids[self.idx]

    @property
    def positions(self) -> np.ndarray:
        return self.pool.positions[self.idx]

    @property
    def world_from_camera(self) -> np.ndarray:
        return self.pool.world_from_camera[self.idx]

    @property
    def n(self) -> int:
        return len(self.idx)
