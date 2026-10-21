"""The field of view of spec 6: how much of each target hypothesis a camera pose sees.

a_{t,q}(v, s) = phi_q(v, s) a_hat_q(v, s): phi is the share of the hypothesis' silhouette inside the image, the
silhouette taken as the square of the same area centred on the projection of x_v (side sqrt(fx fy) s / z pixels), and
0 outside the [near, far] depth range. The point indicator 1{v in FOV(q)} is its limit for a vanishing size; it is
kept for the coverage baseline, which asks which candidates a pose brings into view. Occlusion is not modelled
(spec 6), so nothing here reads the scene; occlusion is the voxel belief's (spec 4), and transmittance() below turns
it into the expected visibility of each hypothesis' front face.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from bamos.mapping.camera import CameraModel
from bamos.mapping.grid import VoxelGrid
from bamos.visibility.dda_triton import HAVE_TRITON, dda_log_transmittance

FACE = np.array([(a, b) for a in (-1.0, 0.0, 1.0) for b in (-1.0, 0.0, 1.0)]) / 3.0  # 3 x 3 cell centres, units of s


def cell_offsets(step: float) -> np.ndarray:
    """[Q, 3] the points a candidate cell's likelihood is averaged over (spec 7.3): the centres of the cell's 8 octants,
    +-step/4 on each axis, the 8-point midpoint rule for a centre uniform in the cell."""
    q = step / 4.0
    return np.array([(a, b, c) for a in (-q, q) for b in (-q, q) for c in (-q, q)], dtype=np.float64)


@dataclass
class EngineConfig:
    device: str = "cpu"
    dtype: str = "float64"
    pose_chunk: int = 256
    ray_margin_m: float = 0.10  # a ray to a front-face point ignores its last margin (the surface the body rests on)
    dda_max_iter: int | None = None  # voxel steps per transmittance ray; None: enough to cross the whole grid
    # the transmittance cache is emptied when it would pass this size: a scene model that never changes (true geometry)
    # never clears it otherwise, and a house-sized candidate set fills the GPU (values are memoised, not changed)
    cache_max_bytes: int = 8 * 2**30


class VisibilityEngine:
    def __init__(self, grid: VoxelGrid, camera: CameraModel, candidate_voxels: np.ndarray, cfg: EngineConfig | None = None):
        self.cfg = cfg or EngineConfig()
        self.grid = grid
        self.camera = camera
        self.device = torch.device(self.cfg.device)
        self.dtype = getattr(torch, self.cfg.dtype)
        self.centers = torch.as_tensor(grid.centers(), dtype=self.dtype, device=self.device)  # [n_vox,3]
        self.candidate_voxels = torch.as_tensor(np.asarray(candidate_voxels, dtype=np.int64), device=self.device)
        self.candidate_centers = self.centers[self.candidate_voxels]  # [n_C,3]
        # [Q, 3] the points within each cell the likelihood is averaged over (spec 7.3); set by the runner
        self.offsets = np.zeros((1, 3))
        self._T_cache: dict = {}
        self._T_version = None

    # ------------------------------------------------------------------ occlusion (spec 4, 6)
    def transmittance(self, positions: np.ndarray, sizes: np.ndarray, vb, offset: int = 0,
                      idx: torch.Tensor | None = None) -> torch.Tensor:
        """[P, n_C, n_S] T_bar_{t,q}(v, s): the mean over the 3 x 3 front-face points y_k of the expected
        transmittance of the voxel belief vb along camera -> y_k, prod (1 - b q_occ - (1 - b) q_free) (spec 6), for the
        body centred at x_v + offsets[offset]. Depends on the camera position only; cached per position until vb changes.
        idx: the candidates to evaluate (n_C becomes len(idx)); None: all."""
        pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        s = np.asarray(sizes, dtype=np.float64)
        if self.device.type != "cuda" or not HAVE_TRITON:
            raise RuntimeError("the voxel belief's transmittance needs a CUDA device with Triton (experiment.device: cuda:N)")
        if self._T_version != (id(vb), vb.version):
            self._T_cache.clear()
            self._T_bytes = 0
            self._T_version = (id(vb), vb.version)
        log_free = vb.log_free().to(self.device)
        idx = self._index(idx)
        sub = None if idx is None else idx.cpu().numpy().tobytes()
        keys = [(tuple(np.round(p, 6)), int(offset)) if sub is None else (tuple(np.round(p, 6)), int(offset), sub)
                for p in pos]
        if getattr(self, "_T_bytes", 0) > self.cfg.cache_max_bytes:
            self._T_cache.clear()
            self._T_bytes = 0
        for key in dict.fromkeys(keys):
            if key not in self._T_cache:
                t = self._transmittance_at(np.asarray(key[0]), s, log_free, vb.grid, key[1], idx)
                self._T_cache[key] = t
                self._T_bytes = getattr(self, "_T_bytes", 0) + t.numel() * t.element_size()
        return torch.stack([self._T_cache[k] for k in keys])

    def _index(self, idx) -> torch.Tensor | None:
        return None if idx is None else torch.as_tensor(idx, dtype=torch.int64, device=self.device).reshape(-1)

    def centres_at(self, offset: int = 0, idx: torch.Tensor | None = None) -> torch.Tensor:
        """[n_C, 3] x_v + offsets[offset], for the candidates idx (None: all)."""
        c = self.candidate_centers if idx is None else self.candidate_centers[self._index(idx)]
        return c + torch.as_tensor(self.offsets[offset], dtype=self.dtype, device=self.device)[None, :]

    def _transmittance_at(self, p: np.ndarray, s: np.ndarray, log_free: torch.Tensor, g: VoxelGrid, offset: int = 0,
                          idx: torch.Tensor | None = None) -> torch.Tensor:
        pt = torch.as_tensor(p, dtype=torch.float64, device=self.device)
        x = self.centres_at(offset, idx).to(torch.float64)  # [n_C,3]
        n = pt[None, :] - x
        n = n / torch.clamp(torch.linalg.norm(n, dim=1, keepdim=True), min=1e-9)
        up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64, device=self.device).expand_as(n)
        e1 = torch.linalg.cross(n, up)
        flat = torch.linalg.norm(e1, dim=1, keepdim=True) < 1e-6  # looking straight up or down
        e1 = torch.where(flat, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64, device=self.device).expand_as(n), e1)
        e1 = e1 / torch.linalg.norm(e1, dim=1, keepdim=True)
        e2 = torch.linalg.cross(n, e1)
        st = torch.as_tensor(s, dtype=torch.float64, device=self.device)[None, :, None, None]  # [1,n_S,1,1]
        f = torch.as_tensor(FACE, dtype=torch.float64, device=self.device)  # [9,2]
        pts = (x[:, None, None, :] + st * (0.5 * n[:, None, None, :] + f[None, None, :, 0:1] * e1[:, None, None, :]
                                            + f[None, None, :, 1:2] * e2[:, None, None, :]))  # [n_C,n_S,9,3]
        nx, ny, nz = g.shape
        cam_ijk = np.floor((p - g.lower) / g.voxel_size).astype(np.int64)
        cam_lin = int((cam_ijk[0] * ny + cam_ijk[1]) * nz + cam_ijk[2]) if np.all((cam_ijk >= 0) & (cam_ijk < g.shape)) else -1
        max_iter = int(sum(g.shape)) + 3 if self.cfg.dda_max_iter is None else int(self.cfg.dda_max_iter)
        logT = dda_log_transmittance(pts.reshape(-1, 3), log_free, p, g.lower, g.voxel_size, g.shape, cam_lin,
                                     max_iter, self.cfg.ray_margin_m)
        return torch.exp(logT).reshape(pts.shape[:-1]).mean(dim=-1).to(self.dtype)  # [n_C,n_S]

    def to_tensor(self, a: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.asarray(a), dtype=self.dtype, device=self.device)

    def candidate_coverage(self, world_from_camera: np.ndarray, sizes: np.ndarray,
                           camera: CameraModel | None = None, offset: int = 0,
                           idx: torch.Tensor | None = None) -> torch.Tensor:
        """[P, n_C, n_S] phi_q(v, s), the share of the hypothesis' square silhouette inside the image (spec 6), for the
        body centred at x_v + offsets[offset]; idx: the candidates to evaluate (None: all)."""
        cam = self.camera if camera is None else camera
        T = np.asarray(world_from_camera)
        T = T[None] if T.ndim == 2 else T
        s = self.to_tensor(np.asarray(sizes, dtype=np.float64))
        x = self.centres_at(offset, idx)
        out = torch.empty((T.shape[0], x.shape[0], len(s)), dtype=self.dtype, device=self.device)
        w, h = float(cam.width), float(cam.height)
        for lo in range(0, T.shape[0], self.cfg.pose_chunk):
            Tc = self.to_tensor(T[lo : lo + self.cfg.pose_chunk])
            R, t = Tc[:, :3, :3], Tc[:, :3, 3]
            pc = torch.einsum("pji,nj->pni", R, x) - torch.einsum("pji,pj->pi", R, t)[:, None, :]
            z = pc[..., 2]
            zz = torch.clamp(z, min=cam.near)
            u = (cam.fx * pc[..., 0] / zz + cam.cx)[..., None]
            v = (cam.fy * pc[..., 1] / zz + cam.cy)[..., None]
            L = (float(np.sqrt(cam.fx * cam.fy)) * s)[None, None, :] / zz[..., None]  # side of the square, pixels

            def share(c, extent):  # the part of [c - L/2, c + L/2] inside [-1/2, extent - 1/2], as a fraction of L
                return torch.clamp((torch.minimum(c + L / 2, torch.full_like(c, extent - 0.5))
                                    - torch.maximum(c - L / 2, torch.full_like(c, -0.5))) / L, 0.0, 1.0)

            depth_ok = ((z >= cam.near) & (z <= cam.far)).to(self.dtype)[..., None]
            out[lo : lo + self.cfg.pose_chunk] = share(u, w) * share(v, h) * depth_ok
        return out

    def candidate_fov(self, world_from_camera: np.ndarray, camera: CameraModel | None = None) -> torch.Tensor:
        """[P, n_C] 1{v in FOV(q)} as the engine's dtype, for world_from_camera [P,4,4] (or one [4,4])."""
        cam = self.camera if camera is None else camera
        T = np.asarray(world_from_camera)
        T = T[None] if T.ndim == 2 else T
        out = torch.empty((T.shape[0], self.candidate_centers.shape[0]), dtype=self.dtype, device=self.device)
        for lo in range(0, T.shape[0], self.cfg.pose_chunk):
            Tc = self.to_tensor(T[lo : lo + self.cfg.pose_chunk])
            R, t = Tc[:, :3, :3], Tc[:, :3, 3]
            # camera coordinates R^T (x - t)
            pc = torch.einsum("pji,nj->pni", R, self.candidate_centers) - torch.einsum("pji,pj->pi", R, t)[:, None, :]
            z = pc[..., 2]
            zz = torch.where(z.abs() > 1e-12, z, torch.full_like(z, 1e-12))
            u = cam.fx * pc[..., 0] / zz + cam.cx
            v = cam.fy * pc[..., 1] / zz + cam.cy
            m = (z >= cam.near) & (z <= cam.far) & (u >= -0.5) & (u <= cam.width - 0.5) & (v >= -0.5) & (v <= cam.height - 0.5)
            out[lo : lo + self.cfg.pose_chunk] = m.to(self.dtype)
        return out
