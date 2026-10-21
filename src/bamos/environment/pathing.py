"""Common navigation oracle: 2D clearance map, reachability and path cost. Returns feasibility and path length only."""
from __future__ import annotations

import heapq

import numpy as np

from bamos.environment.scene import SceneObject


class NavigationOracle:
    def __init__(self, workspace_lo, workspace_hi, obstacles: list[SceneObject], robot_clearance: float, robot_z: float, cell: float = 0.1):
        self.lo = np.asarray(workspace_lo[:2], dtype=np.float64)
        self.hi = np.asarray(workspace_hi[:2], dtype=np.float64)
        self.cell = cell
        self.shape = tuple(int(np.ceil((self.hi[i] - self.lo[i]) / cell - 1e-9)) for i in range(2))
        occ = np.zeros(self.shape, dtype=bool)
        xs = self.lo[0] + (np.arange(self.shape[0]) + 0.5) * cell
        ys = self.lo[1] + (np.arange(self.shape[1]) + 0.5) * cell
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        for obj in obstacles:
            for p in obj.prims:
                plo, phi = p.aabb()
                if phi[2] <= 0.02 or plo[2] >= robot_z + 0.3:  # floor-level slabs and things above the robot body do not block
                    continue
                m = (X >= plo[0] - robot_clearance) & (X <= phi[0] + robot_clearance) & (Y >= plo[1] - robot_clearance) & (Y <= phi[1] + robot_clearance)
                occ |= m
        # wall clearance
        occ |= (X < self.lo[0] + robot_clearance) | (X > self.hi[0] - robot_clearance) | (Y < self.lo[1] + robot_clearance) | (Y > self.hi[1] - robot_clearance)
        self.occ = occ
        self._dist_cache: dict[tuple[int, int], np.ndarray] = {}

    def cell_of(self, xy) -> tuple[int, int]:
        ij = np.floor((np.asarray(xy[:2]) - self.lo) / self.cell).astype(int)
        ij = np.clip(ij, 0, np.asarray(self.shape) - 1)
        return int(ij[0]), int(ij[1])

    def free(self, xy) -> bool:
        i, j = self.cell_of(xy)
        return not self.occ[i, j]

    def distances_from(self, xy) -> np.ndarray:
        """Dijkstra geodesic distance (meters) from xy to every free cell; inf where unreachable."""
        key = self.cell_of(xy)
        if key in self._dist_cache:
            return self._dist_cache[key]
        dist = np.full(self.shape, np.inf)
        if self.occ[key]:
            self._dist_cache[key] = dist
            return dist
        dist[key] = 0.0
        pq = [(0.0, key)]
        nbrs = [(di, dj, self.cell * np.hypot(di, dj)) for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)]
        nx, ny = self.shape
        while pq:
            d, (i, j) = heapq.heappop(pq)
            if d > dist[i, j]:
                continue
            for di, dj, w in nbrs:
                a, b = i + di, j + dj
                if 0 <= a < nx and 0 <= b < ny and not self.occ[a, b]:
                    nd = d + w
                    if nd < dist[a, b]:
                        dist[a, b] = nd
                        heapq.heappush(pq, (nd, (a, b)))
        self._dist_cache[key] = dist
        return dist

    def path_costs(self, from_xy, to_xy: np.ndarray) -> np.ndarray:
        dist = self.distances_from(from_xy)
        cells = np.floor((to_xy[:, :2] - self.lo) / self.cell).astype(int)
        cells = np.clip(cells, 0, np.asarray(self.shape) - 1)
        return dist[cells[:, 0], cells[:, 1]]

    def reachable_from(self, from_xy, to_xy: np.ndarray) -> np.ndarray:
        return np.isfinite(self.path_costs(from_xy, to_xy))
