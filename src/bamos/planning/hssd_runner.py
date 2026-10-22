"""The episode runner (planning/runner.py) on HSSD ObjectNav with teleport poses (environment/hssd_env.py).

BAMOS itself is unchanged: the belief over (cell, size), its DET / NO_DET updates with r and kappa, the voxel belief's
transmittance gate, I(Z;X), the stop rule with its present view. What this module sets is the world it runs in:

- Candidate cells: the space above the house's floors (hssd.candidate_floors = "all": the navmesh lattice of every
  island; "start_island": option ii, the start island's pose positions only): 0.25 m cells within candidate_dilate_m
  horizontally of a floor position, candidate_band_m above that position's floor surface, seen from it past no
  known-shape voxel (a horizontal segment at the cell's height), and not inside the known shape.
- Size hypotheses: the episode category's (hssd.size_grid_by_category).
- The voxel belief starts from the known shape (stage, windows, railings, stairs, doors) as opaque voxels.
- r on the cells within reach of each camera position (planning/hssd_scorer.py).
- The next pose: a pose position at navmesh geodesic <= max_geodesic_m from the current one whose floor was seen in an
  earlier frame (an up-facing depth point near its floor surface; FloorSeenMap), at any yaw and pitch; a pose already
  taken is not a candidate (experiment.forbid_exact_revisit).
- Evaluation against every goal instance (evaluation/hssd_evaluator.py).
"""
from __future__ import annotations

import numpy as np

from bamos.belief.voxel_belief import KnownShapeVoxelBelief
from bamos.evaluation.hssd_evaluator import HssdEvaluator
from bamos.mapping.grid import VoxelGrid
from bamos.observation.types import PlannerObservation
from bamos.planning.hssd_scorer import HssdScorer, reach_of
from bamos.planning.runner import EpisodeResult, EpisodeRunner
from bamos.planning.scorer import ScorerConfig
from bamos.visibility.engine import EngineConfig


class FloorSeenMap:
    """Where a floor surface has been seen: the up-facing depth points of the frames so far (below the camera), binned in
    xy cells of `cell` and heights of `zbin`. seen() asks for a point within tol of a floor height in the cell of a
    position or one of its 8 neighbours."""

    def __init__(self, cell: float, zbin: float, stride: int = 4):
        self.cell, self.zbin, self.stride = float(cell), float(zbin), int(stride)
        self.keys = np.zeros(0, dtype=np.int64)

    @staticmethod
    def _key(ix, iy, iz):
        return (np.asarray(ix, np.int64) + (1 << 20)) * (1 << 42) + (np.asarray(iy, np.int64) + (1 << 20)) * (1 << 21) \
            + (np.asarray(iz, np.int64) + (1 << 20))

    def add(self, obs: PlannerObservation):
        cam, st = obs.camera, self.stride
        d = np.asarray(obs.depth, dtype=np.float64)[::st, ::st]
        vs, us = np.mgrid[0:cam.height:st, 0:cam.width:st]
        ok = np.isfinite(d) & (d >= cam.near) & (d <= cam.far)
        if not ok.any():
            return
        pc = cam.backproject(np.stack([us[ok], vs[ok]], 1).astype(np.float64), d[ok])
        T = obs.world_from_camera
        pw = pc @ T[:3, :3].T + T[:3, 3]
        pw = pw[pw[:, 2] < T[2, 3] - 0.05]  # below the camera: an up-facing surface
        k = self._key(np.floor(pw[:, 0] / self.cell), np.floor(pw[:, 1] / self.cell), np.floor(pw[:, 2] / self.zbin))
        self.keys = np.union1d(self.keys, k)

    def seen(self, xy: np.ndarray, floor_z: np.ndarray, tol: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        out = np.zeros(len(xy), dtype=bool)
        if not len(xy) or not len(self.keys):
            return out
        fin = np.isfinite(floor_z)
        if not fin.any():
            return out
        fz = np.where(fin, floor_z, 0.0)  # no floor known (the start): never seen here
        cx, cy = np.floor(xy[:, 0] / self.cell).astype(np.int64), np.floor(xy[:, 1] / self.cell).astype(np.int64)
        z0 = np.floor((fz - tol) / self.zbin).astype(np.int64)
        z1 = np.floor((fz + tol) / self.zbin).astype(np.int64)
        for dz in range(int((z1 - z0).max()) + 1):
            iz = z0 + dz
            live = iz <= z1
            for a in (-1, 0, 1):
                for b in (-1, 0, 1):
                    out |= live & np.isin(self._key(cx + a, cy + b, iz), self.keys)
        return out & fin


class HssdEpisodeRunner(EpisodeRunner):
    def __init__(self, env, cfg: dict, ctx, step_writer=None, array_store: dict | None = None):
        self.hc = cfg["hssd"]
        self._occ_cache = None
        super().__init__(env, cfg, ctx, step_writer, array_store)
        self.floor_seen = FloorSeenMap(self.hc["floor_seen_cell_m"], self.hc["floor_seen_bin_m"])
        self.n_floor_blocked: list[int] = []  # per active step: positions within the geodesic range whose floor was unseen

    # ---------------------------------------------------------------------------------------------- set-up
    def make_grid(self) -> VoxelGrid:
        sc = self.env.scene
        return VoxelGrid.from_bounds(np.asarray(sc.workspace_lo), np.asarray(sc.workspace_hi), float(self.cfg["mapping"]["voxel_size_m"]))

    def occupancy_grid(self) -> VoxelGrid:
        hi = self.grid.lower + np.asarray(self.grid.shape) * self.grid.voxel_size
        return VoxelGrid.from_bounds(self.grid.lower, hi, float(self.cfg["mapping"]["occupancy_voxel_m"]))

    def known(self) -> tuple[VoxelGrid, np.ndarray]:
        if self._occ_cache is None:
            g = self.occupancy_grid()
            self._occ_cache = (g, self.env.known_occupancy(g))
        return self._occ_cache

    def make_candidates(self) -> np.ndarray:
        """Option (ii): see the module docstring."""
        env, g, hc = self.env, self.grid, self.hc
        og, occ = self.known()
        occ3 = occ.reshape(og.shape)
        vs = g.voxel_size
        D = float(hc["candidate_dilate_m"])
        b0, b1 = (float(x) for x in hc["candidate_band_m"])
        pos = env.support_world  # the start island's pose positions, or every island's (hssd.candidate_floors)
        fz = env.support_floor_z
        keep = np.isfinite(fz)
        pos, fz = pos[keep], fz[keep]
        r = int(np.ceil(D / vs))
        di, dj = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1), indexing="ij")
        disk = (np.hypot(di, dj) * vs <= D)
        di, dj = di[disk], dj[disk]
        nx, ny, nz = g.shape

        def occupied(p):  # [M, 3] world points -> inside a known-shape voxel
            ijk = np.floor((p - og.lower) / og.voxel_size).astype(np.int64)
            ok = np.all((ijk >= 0) & (ijk < np.asarray(og.shape)), axis=1)
            out = np.zeros(len(p), dtype=bool)
            out[ok] = occ3[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]]
            return out

        cells = []
        n_s = int(np.ceil(D / 0.05)) + 1
        t = np.linspace(0.0, 1.0, n_s)
        for p, f in zip(pos, fz):
            i0, j0 = np.floor((p[:2] - g.lower[:2]) / vs).astype(np.int64)
            k0 = int(np.ceil((f + b0 - g.lower[2]) / vs - 0.5))
            k1 = int(np.floor((f + b1 - g.lower[2]) / vs - 0.5))
            ks = np.arange(max(k0, 0), min(k1, nz - 1) + 1)
            I, J = i0 + di, j0 + dj
            ok = (I >= 0) & (I < nx) & (J >= 0) & (J < ny)
            I, J = I[ok], J[ok]
            if not len(ks) or not len(I):
                continue
            II, KK = np.repeat(I, len(ks)), np.tile(ks, len(I))
            JJ = np.repeat(J, len(ks))
            c = g.lower + (np.stack([II, JJ, KK], 1) + 0.5) * vs  # cell centres
            a = np.concatenate([np.repeat(p[None, :2], len(c), 0), c[:, 2:3]], axis=1)  # the position, at the cell's height
            seg = a[:, None, :] + (c - a)[:, None, :] * t[None, :, None]  # [M, n_s, 3]
            clear = ~occupied(seg.reshape(-1, 3)).reshape(len(c), n_s).any(axis=1)
            cells.append(((II * ny + JJ) * nz + KK)[clear])
        lin = np.unique(np.concatenate(cells)) if cells else np.zeros(0, dtype=np.int64)
        return lin[~occupied(g.centers(g.unravel(lin)))] if len(lin) else lin

    def size_grid(self) -> np.ndarray:
        return np.asarray(self.hc["size_grid_by_category"][self.env.category], dtype=np.float64)

    def engine_config(self) -> EngineConfig:
        ec = super().engine_config()
        from bamos.visibility.engine import cell_offsets

        step = float(self.cfg["mapping"]["voxel_size_m"]) * int(self.cfg["target"]["center_stride_voxels"])
        self._reach = reach_of(self.env.robot_camera, self.sizes, cell_offsets(step))
        # a transmittance ray need only cross the reach, not the whole box
        ec.dda_max_iter = int(np.ceil(3.0 * (self._reach + 1.0) / float(self.cfg["mapping"]["occupancy_voxel_m"]))) + 3
        return ec

    def make_scorer(self, scfg: ScorerConfig) -> HssdScorer:
        return HssdScorer(self.engine, self.det_lik, scfg, self._reach)

    def make_voxel_belief(self):
        mp, ex = self.cfg["mapping"], self.cfg["experiment"]
        if mp.get("occupancy_source", "belief") == "ground_truth":
            # a perfect scene model (the world-model comparison, as Isaac's GroundTruthOccupancy): the true geometry on
            # the occupancy grid, opaque and fixed, every object but the episode's goal instances
            from bamos.belief.voxel_belief import GroundTruthOccupancy
            og = self.occupancy_grid()
            return GroundTruthOccupancy(og, self.env.true_occupancy(og), ex["device"])
        og, occ = self.known()
        p0, q_occ, q_free = (float(x) for x in mp["occupancy_model"])
        vb = KnownShapeVoxelBelief(og, occ, p0, q_occ, q_free, ex["device"], int(mp["occupancy_pixel_stride"]))
        vb.near_hits = bool(mp.get("near_hits", False))
        return vb

    def make_evaluator(self) -> HssdEvaluator:
        return HssdEvaluator(self.env, self.cfg["evaluation"], self.state.candidate_centers(),
                                  float(self.hc["viewpoint_success_m"]), float(self.hc["sensor_height_m"]))

    # ---------------------------------------------------------------------------------------------- the next pose
    def feasible_poses(self, current_pose_id: int) -> np.ndarray:
        env, hc = self.env, self.hc
        gi = env.pool.grid_index
        cur = int(gi[int(current_pose_id), 0])
        b = env.base_world
        R = float(hc["max_geodesic_m"])
        near = np.nonzero(np.hypot(b[:, 0] - b[cur, 0], b[:, 1] - b[cur, 1]) <= R)[0]
        geo = env.geodesic(cur, near)
        in_range = near[geo <= R]
        tol = np.where(env.sloped[in_range], float(hc["floor_tol_sloped_m"]), float(hc["floor_tol_m"]))
        seen = self.floor_seen.seen(b[in_range, :2], env.floor_z[in_range], tol)
        seen |= (in_range == 0) | (in_range == cur)  # the start, and the current position (other headings)
        self.n_floor_blocked.append(int((~seen).sum()))
        ok_pos = np.zeros(len(b), dtype=bool)
        ok_pos[in_range[seen]] = True
        return ok_pos[gi[:, 0]]

    def after_observation(self, obs: PlannerObservation) -> None:
        self.floor_seen.add(obs)

    def run(self) -> EpisodeResult:
        res = super().run()
        ev, gt = self.evaluator, self.env.ground_truth()
        spl = float(gt.geodesic_distance / max(self.total_path, gt.geodesic_distance)) if res.actual_success else 0.0
        res.extra.update({
            "viewpoint_success": bool(ev.viewpoint_success), "declared_goal": ev.declared_goal,
            "geodesic_distance_m": float(gt.geodesic_distance), "spl": spl,
            "n_goal_instances": len(gt.goals), "category": self.env.category, "scene": self.env.scene_name,
            "n_pose_positions": int(len(self.env.base_world)), "reach_m": self._reach,
            "floor_blocked_mean": float(np.mean(self.n_floor_blocked)) if self.n_floor_blocked else 0.0,
        })
        return res
