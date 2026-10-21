"""Camera intrinsics and pose conventions.

World frame: z up, meters. Camera optical frame: x right, y down, z forward.
A pose is stored as the 4x4 world-from-camera(optical) homogeneous transform.
Image pixel (u, v): u = column (x right), v = row (y down); pixel centers at
integer coordinates, so u = fx * x/z + cx.

Yaw psi: 0 along world +x, positive counter-clockwise about +z.
Pitch theta: 0 horizontal, positive looking up.
Forward direction f(psi, theta) = [cos t cos p, cos t sin p, sin t].
Roll is always 0 for the movable camera: image x axis is horizontal.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraModel:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    near: float
    far: float
    depth_kind: str = "optical_axis"  # or "ray_range"

    @staticmethod
    def from_hfov(width: int, height: int, hfov_rad: float, near: float, far: float) -> "CameraModel":
        fx = 0.5 * width / np.tan(0.5 * hfov_rad)
        return CameraModel(width, height, float(fx), float(fx), 0.5 * (width - 1), 0.5 * (height - 1), near, far)

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    @property
    def n_pixels(self) -> int:
        return self.width * self.height

    def pixel_rays(self, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Unit-free camera-frame directions (z=1) for a pixel lattice.

        Returns (dirs [N,3] with z=1, pixel (u,v) integer coordinates [N,2]).
        """
        vs, us = np.meshgrid(np.arange(0, self.height, stride), np.arange(0, self.width, stride), indexing="ij")
        us = us.ravel().astype(np.float64)
        vs = vs.ravel().astype(np.float64)
        dirs = np.stack([(us - self.cx) / self.fx, (vs - self.cy) / self.fy, np.ones_like(us)], axis=1)
        return dirs, np.stack([us, vs], axis=1).astype(np.int64)

    def project(self, p_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Camera-frame points [N,3] -> pixel coords [N,2] and depth z [N]."""
        p_cam = np.asarray(p_cam, dtype=np.float64)
        z = p_cam[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.fx * p_cam[..., 0] / z + self.cx
            v = self.fy * p_cam[..., 1] / z + self.cy
        return np.stack([u, v], axis=-1), z

    def backproject(self, uv: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """Pixel coords + depth -> camera-frame points, honoring depth_kind."""
        uv = np.asarray(uv, dtype=np.float64)
        d = np.asarray(depth, dtype=np.float64)
        dirs = np.stack([(uv[..., 0] - self.cx) / self.fx, (uv[..., 1] - self.cy) / self.fy, np.ones_like(d)], axis=-1)
        if self.depth_kind == "optical_axis":
            return dirs * d[..., None]
        if self.depth_kind == "ray_range":
            dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
            return dirs * d[..., None]
        raise ValueError(f"unknown depth_kind {self.depth_kind}")


def forward_from_yaw_pitch(yaw: float, pitch: float) -> np.ndarray:
    return np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])


def world_from_camera(position, yaw: float, pitch: float, roll: float = 0.0) -> np.ndarray:
    """4x4 world-from-optical transform for a camera at position with yaw/pitch (roll 0).

    Optical x = right (horizontal, perpendicular to forward and world z),
    optical y = down, optical z = forward.
    """
    f = forward_from_yaw_pitch(yaw, pitch)
    up = np.array([0.0, 0.0, 1.0])
    if abs(f[2]) > 1.0 - 1e-9:  # looking straight up/down: use yaw to fix the horizontal direction
        right = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    else:
        right = np.cross(f, up)
        right /= np.linalg.norm(right)
    down = np.cross(f, right)
    R = np.stack([right, down, f], axis=1)
    if roll != 0.0:
        c, s = np.cos(roll), np.sin(roll)
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        R = R @ Rz
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(position, dtype=np.float64)
    return T


def yaw_pitch_from_transform(T: np.ndarray) -> tuple[float, float]:
    f = T[:3, 2]
    return float(np.arctan2(f[1], f[0])), float(np.arcsin(np.clip(f[2], -1.0, 1.0)))


def look_at(position, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """World-from-optical transform looking from position toward target with roll 0 (horizontal x axis)."""
    position = np.asarray(position, dtype=np.float64)
    f = np.asarray(target, dtype=np.float64) - position
    f /= np.linalg.norm(f)
    yaw = float(np.arctan2(f[1], f[0]))
    pitch = float(np.arcsin(np.clip(f[2], -1.0, 1.0)))
    return world_from_camera(position, yaw, pitch)


def inverse_transform(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def angle_diff(a: float, b: float) -> float:
    d = (a - b + np.pi) % (2 * np.pi) - np.pi
    return float(d)
