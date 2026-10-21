"""Axis-aligned voxel grid in world coordinates (meters, z up)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VoxelGrid:
    """Regular voxel grid.

    origin: world coordinate of the minimum corner of voxel (0,0,0).
    voxel_size: edge length in meters.
    shape: (nx, ny, nz).
    Voxel (i,j,k) covers [origin + (i,j,k)*s, origin + (i+1,j+1,k+1)*s).
    Linear index is C-order: idx = (i*ny + j)*nz + k.
    """

    origin: tuple[float, float, float]
    voxel_size: float
    shape: tuple[int, int, int]

    @staticmethod
    def from_bounds(lo, hi, voxel_size: float) -> "VoxelGrid":
        lo = np.asarray(lo, dtype=np.float64)
        hi = np.asarray(hi, dtype=np.float64)
        shape = tuple(int(x) for x in np.ceil((hi - lo) / voxel_size - 1e-9))
        return VoxelGrid(tuple(float(x) for x in lo), float(voxel_size), shape)

    @property
    def n_voxels(self) -> int:
        return int(np.prod(self.shape))

    @property
    def lower(self) -> np.ndarray:
        return np.asarray(self.origin, dtype=np.float64)

    def unravel(self, idx: np.ndarray) -> np.ndarray:
        return np.stack(np.unravel_index(np.asarray(idx, dtype=np.int64), self.shape), axis=-1)

    def centers(self, ijk: np.ndarray | None = None) -> np.ndarray:
        """World centers of given voxel indices (or of all voxels in linear order)."""
        if ijk is None:
            ijk = self.unravel(np.arange(self.n_voxels))
        return self.lower + (np.asarray(ijk, dtype=np.float64) + 0.5) * self.voxel_size

