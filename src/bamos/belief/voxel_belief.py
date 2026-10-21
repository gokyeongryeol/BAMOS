"""The voxel belief b_t of formulation §4: an independent Bernoulli occupancy per voxel of a fine grid.

Each voxel g is occupied (o_g = 1) or free, independently, with prior P(o_g = 1) = p0. A frame gives each voxel at most
one observation: a hit if some depth ray of the frame ends in it, otherwise a pass if some ray crosses it before its end. The sensor model: a ray entering an occupied voxel ends there with probability
q_occ, a ray entering a free one with probability q_free (discretisation, grazing). The observations are then binomial
given o_g, and the posterior after n_hit hits and n_pass passes is Bernoulli with

    logit b_t(g) = logit p0 + n_hit log(q_occ / q_free) + n_pass log((1 - q_occ) / (1 - q_free)).

A ray survives the voxel with probability 1 - [b q_occ + (1 - b) q_free]; the voxels are independent, so the expected
transmittance along a path is the product over the voxels it crosses.

Depth endpoints inside the reported target mask (a DET frame's) are not hits: the target is not an occluder of its own
hypotheses. The voxels such a ray crossed before its end are still passes. Rays with no valid depth, or depth beyond the camera's far limit, are passes up to far. Rays in `ignore` (a sensor
that clips short returns to its near limit, so the surface is somewhere closer but not where: Habitat's 0.5 m) give
neither, unless `near_hits` is set (mapping.near_hits): then each gives a hit at the near limit and no pass, so a surface
the camera stands close to occludes what lies beyond it (the voxel marked is at or just behind the true surface). As in
active-grounding-growing's voxel belief.
"""
from __future__ import annotations

import numpy as np
import torch

from bamos.mapping.camera import CameraModel
from bamos.mapping.grid import VoxelGrid


class VoxelBelief:
    def __init__(self, grid: VoxelGrid, p0: float, q_occ: float, q_free: float, device: str = "cpu", pixel_stride: int = 2):
        self.grid = grid
        self.p0, self.q_occ, self.q_free = float(p0), float(q_occ), float(q_free)
        self.device = torch.device(device)
        self.stride = int(pixel_stride)
        self.n_hit = torch.zeros(grid.n_voxels, dtype=torch.int32, device=self.device)
        self.n_pass = torch.zeros(grid.n_voxels, dtype=torch.int32, device=self.device)
        self.version = 0
        self._log_free = None
        self.near_hits = False  # ignored (too-near) rays: a hit at the near limit (module docstring)

    def occupancy(self) -> torch.Tensor:
        """[n_vox] b_t(g) = P(o_g = 1 | the hits and passes so far)."""
        lo = (np.log(self.p0 / (1 - self.p0)) + self.n_hit.to(torch.float64) * np.log(self.q_occ / self.q_free)
              + self.n_pass.to(torch.float64) * np.log((1 - self.q_occ) / (1 - self.q_free)))
        return torch.sigmoid(lo)

    def log_free(self) -> torch.Tensor:
        """[n_vox] log P(a ray entering g goes on) = log(1 - b q_occ - (1 - b) q_free), what the ray traversal sums."""
        if self._log_free is None:
            b = self.occupancy()
            self._log_free = torch.log(1.0 - b * self.q_occ - (1.0 - b) * self.q_free)
        return self._log_free

    def _lin(self, pts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        g = self.grid
        lower = torch.as_tensor(g.lower, dtype=torch.float64, device=self.device)
        shape = torch.as_tensor(g.shape, dtype=torch.int64, device=self.device)
        ijk = torch.floor((pts - lower) / g.voxel_size).to(torch.int64)
        inb = torch.all((ijk >= 0) & (ijk < shape), dim=-1)
        nx, ny, nz = g.shape
        lin = (ijk[..., 0] * ny + ijk[..., 1]) * nz + ijk[..., 2]
        return torch.where(inb, lin, torch.full_like(lin, -1)), inb

    def update(self, depth: np.ndarray | None, target_mask: np.ndarray | None, camera: CameraModel,
               world_from_camera: np.ndarray, ignore: np.ndarray | None = None) -> dict:
        """One observation per voxel from one frame's depth (see the module docstring); target_mask is the reported
        mask of a DET frame, None otherwise. Returns counts."""
        if depth is None:
            return {"hits": 0, "passes": 0}
        st = self.stride
        return self.update_subsampled(np.asarray(depth)[::st, ::st],
                                      None if target_mask is None else np.asarray(target_mask)[::st, ::st],
                                      camera, world_from_camera,
                                      None if ignore is None else np.asarray(ignore)[::st, ::st])

    def update_subsampled(self, depth_s: np.ndarray, mask_s: np.ndarray | None, camera: CameraModel,
                          world_from_camera: np.ndarray, ignore_s: np.ndarray | None = None) -> dict:
        """update() on depth and mask already taken at every stride-th pixel (as stage 1 stores them)."""
        st = self.stride
        dirs, uv = camera.pixel_rays(st)
        d = np.asarray(depth_s, dtype=np.float64)[uv[:, 1] // st, uv[:, 0] // st]
        valid = np.isfinite(d) & (d >= camera.near) & (d <= camera.far)
        end = np.where(valid, d, camera.far)  # no valid return: free up to far
        use = np.ones(len(d), dtype=bool)
        if ignore_s is not None:
            use = ~np.asarray(ignore_s, dtype=bool)[uv[:, 1] // st, uv[:, 0] // st]
        in_mask = np.zeros(len(d), dtype=bool)
        if mask_s is not None:
            in_mask = np.asarray(mask_s, dtype=bool)[uv[:, 1] // st, uv[:, 0] // st]
        near_hit = (~use & ~in_mask) if self.near_hits else np.zeros(len(d), dtype=bool)
        end = np.where(near_hit, camera.near, end)
        T = np.asarray(world_from_camera, dtype=np.float64)
        dev = self.device
        o = torch.as_tensor(T[:3, 3], dtype=torch.float64, device=dev)
        pc = camera.backproject(uv.astype(np.float64), end)  # camera-frame end points (depth_kind aware)
        pw = torch.as_tensor(pc @ T[:3, :3].T + T[:3, 3], dtype=torch.float64, device=dev)  # [R,3]
        # hits: the endpoints of the valid rays outside the target mask (and, with near_hits, of the too-near ones)
        hit_rays = torch.as_tensor((valid & ~in_mask & use) | near_hit, device=dev)
        lin_end, inb_end = self._lin(pw)
        hit_vox = torch.unique(lin_end[hit_rays & inb_end])
        # passes: voxels sampled strictly before each ray's end voxel (half-voxel steps), valid or not
        use_t = torch.as_tensor(use, device=dev)
        L = torch.linalg.norm(pw - o[None], dim=1)
        u = (pw - o[None]) / torch.clamp(L, min=1e-9)[:, None]
        step = 0.5 * self.grid.voxel_size
        n = int(np.ceil(float(L.max()) / step)) if len(L) else 0
        passed = []
        for lo in range(0, n, 64):  # in slabs of 64 samples per ray, to bound memory
            ts = torch.arange(lo + 1, min(n, lo + 64) + 1, dtype=torch.float64, device=dev) * step  # [S]
            keep = ts[None, :] < (L[:, None] - step)
            pts = o[None, None, :] + u[:, None, :] * ts[None, :, None]
            lin, inb = self._lin(pts)
            m = keep & inb & (lin != lin_end[:, None]) & use_t[:, None]
            if m.any():
                passed.append(torch.unique(lin[m]))
        pass_vox = torch.unique(torch.cat(passed)) if passed else torch.zeros(0, dtype=torch.int64, device=dev)
        # a voxel hit in this frame is not also a pass in it
        if len(hit_vox) and len(pass_vox):
            pass_vox = pass_vox[~torch.isin(pass_vox, hit_vox)]
        self.n_hit[hit_vox] += 1
        self.n_pass[pass_vox] += 1
        self.version += 1
        self._log_free = None
        return {"hits": int(len(hit_vox)), "passes": int(len(pass_vox))}


class GroundTruthOccupancy:
    """A perfect scene model in place of b_t (2026-10-08, the world-model comparison): the occupancy of the true scene
    geometry on the same voxel grid, every object but the target, opaque and fixed. A voxel is occupied if any of its
    k^3 sub-samples lies inside a primitive, so boards thinner than a voxel are kept. Frames do not change it."""

    LOG_BLOCK = float(np.log(1e-6))  # log P(a ray goes on) through an occupied voxel

    def __init__(self, grid: VoxelGrid, occupied: np.ndarray, device: str = "cpu"):
        self.grid = grid
        self.device = torch.device(device)
        self.occupied = torch.as_tensor(occupied, dtype=torch.bool, device=self.device)
        self.version = 0
        self._log_free = torch.where(self.occupied, torch.tensor(self.LOG_BLOCK, dtype=torch.float64, device=self.device),
                                     torch.tensor(0.0, dtype=torch.float64, device=self.device))

    @staticmethod
    def from_scene(grid: VoxelGrid, scene, device: str = "cpu", sub: int = 3) -> "GroundTruthOccupancy":
        from bamos.environment.scene import Box, Cylinder, Sphere

        lower, vs, shape = np.asarray(grid.lower, dtype=np.float64), float(grid.voxel_size), np.asarray(grid.shape)
        occ = np.zeros(tuple(shape), dtype=bool)
        f = (np.arange(sub) + 0.5) / sub  # sub-sample offsets inside a voxel, in voxels
        for o in scene.objects:
            if o.instance_id == scene.gt_target_instance_id:
                continue  # the target is what is searched for, not scene knowledge
            for p in o.prims:
                lo, hi = p.aabb()
                i0 = np.clip(np.floor((lo - lower) / vs).astype(int), 0, shape)
                i1 = np.clip(np.ceil((hi - lower) / vs).astype(int), 0, shape)
                if np.any(i1 <= i0):
                    continue
                ax = [lower[d] + (np.arange(i0[d], i1[d])[:, None] + f[None, :]).ravel() * vs for d in range(3)]
                X, Y, Z = np.meshgrid(*ax, indexing="ij")  # [nx*sub, ny*sub, nz*sub]
                c = np.asarray(p.center, dtype=np.float64)
                if isinstance(p, Box):
                    cy, sy = np.cos(p.yaw), np.sin(p.yaw)
                    dx, dy = X - c[0], Y - c[1]
                    h = 0.5 * np.asarray(p.size)
                    inside = ((np.abs(cy * dx + sy * dy) <= h[0]) & (np.abs(-sy * dx + cy * dy) <= h[1])
                              & (np.abs(Z - c[2]) <= h[2]))
                elif isinstance(p, Cylinder):
                    inside = ((X - c[0]) ** 2 + (Y - c[1]) ** 2 <= p.radius ** 2) & (np.abs(Z - c[2]) <= 0.5 * p.height)
                elif isinstance(p, Sphere):
                    inside = (X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2 <= p.radius ** 2
                else:
                    raise TypeError(f"unknown primitive {type(p).__name__}")
                n = i1 - i0
                hit = inside.reshape(n[0], sub, n[1], sub, n[2], sub).any(axis=(1, 3, 5))
                occ[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]] |= hit
        return GroundTruthOccupancy(grid, occ.ravel(), device)  # C order: lin = (i * ny + j) * nz + k, as _lin

    def occupancy(self) -> torch.Tensor:
        return self.occupied.to(torch.float64)

    def log_free(self) -> torch.Tensor:
        return self._log_free

    def update(self, *args, **kwargs) -> dict:
        return {"hits": 0, "passes": 0}


class KnownShapeVoxelBelief(VoxelBelief):
    """b_t with the building's shape known beforehand (HSSD: the stage and its windows, railings, stairs and doors):
    those voxels are opaque from the start and stay so; every other voxel is the Bernoulli belief above. Depth hits on
    a known voxel are still counted (they change nothing that is read)."""

    LOG_BLOCK = GroundTruthOccupancy.LOG_BLOCK

    def __init__(self, grid: VoxelGrid, known: np.ndarray, p0: float, q_occ: float, q_free: float, device: str = "cpu",
                 pixel_stride: int = 2):
        super().__init__(grid, p0, q_occ, q_free, device, pixel_stride)
        self.known = torch.as_tensor(np.asarray(known, dtype=bool).reshape(-1), device=self.device)

    def occupancy(self) -> torch.Tensor:
        return torch.where(self.known, torch.ones((), dtype=torch.float64, device=self.device), super().occupancy())

    def log_free(self) -> torch.Tensor:
        if self._log_free is None:
            b = super().occupancy()
            lf = torch.log(1.0 - b * self.q_occ - (1.0 - b) * self.q_free)
            self._log_free = torch.where(self.known, torch.tensor(self.LOG_BLOCK, dtype=torch.float64, device=self.device), lf)
        return self._log_free
