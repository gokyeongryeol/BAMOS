"""Habitat <-> BAMOS frames.

Habitat world: y up, the agent looks down -z at yaw 0. BAMOS world (mapping/camera.py): z up, metres. The map
C: (x, y, z)_hab -> (x, -z, y) is a proper rotation (det +1), so headings and handedness carry over.
Habitat cameras look down -z with +y up; the BAMOS optical frame is x right, y down, z forward, hence
world_from_optical = C R_hab diag(1, -1, -1).
"""
from __future__ import annotations

import numpy as np

C = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
HABCAM_FROM_OPTICAL = np.diag([1.0, -1.0, -1.0])


def hab_to_world(p) -> np.ndarray:
    return np.asarray(p, dtype=np.float64) @ C.T


def quat_wxyz_to_matrix(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def world_from_optical(sensor_position_hab, sensor_rotation_wxyz) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = C @ quat_wxyz_to_matrix(sensor_rotation_wxyz) @ HABCAM_FROM_OPTICAL
    T[:3, 3] = hab_to_world(sensor_position_hab)
    return T
