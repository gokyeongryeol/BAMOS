"""HSSD ObjectNav with teleport poses: the client of servers/hssd_server.py and the Environment the runner sees.

The scene, the episode, the navmesh and the camera are the HSSD ObjectNav benchmark's (habitat-lab objectnav_hssd-hab.yaml);
the agent is not walked but placed at the chosen pose, as the Isaac benchmark's robot camera is.

- Pose pool: navmesh points on a pose_lattice_m xz lattice on the episode start's navmesh island (every navigable height
  at a lattice point: other floors joined by stairs too), plus the start itself, times the yaws and pitches of
  cameras.robot. The camera is sensor_height_m above the navmesh point, as the benchmark agent's.
- Depth: the benchmark sensor's range [min_depth, max_depth]. A return nearer than min_depth is clipped by that sensor (the
  surface is somewhere closer, not where): such pixels are NaN and listed in meta["depth_ignore"]; no return within
  max_depth is +inf.
- Oracle detector: the goal instances (the episode's goals, by semantic id) with at least oracle_min_px pixels of valid
  depth; the one with the most such pixels is the frame's mask, all of them its instance masks (several goal instances in
  one frame: the multi-mask update of formulation §8.2).
- Known shape: the stage and the objects of known_shape_categories (windows, railings, stairs, doors), as surface points
  from the server's ray sweeps (cached per scene), voxelised onto any grid by known_occupancy().
- Path costs: navmesh geodesic distances between the poses' navmesh points.
Evaluator-only: the goals' boxes (ground_truth()), the per-frame quality dict. The planner never reads them.
"""
from __future__ import annotations

import hashlib
import json
import os
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from bamos.environment.base import Environment, EvaluatorGroundTruth
from bamos.environment.hssd_frames import hab_to_world
from bamos.environment.isaac_env import IsaacClient
from bamos.environment.scene import SceneDescription
from bamos.environment.sensor import corrupt_detection
from bamos.mapping.camera import CameraModel, world_from_camera
from bamos.mapping.grid import VoxelGrid
from bamos.observation.types import DetectionStatus, PlannerObservation
from bamos.planning.candidates import CandidatePool


class HssdClient(IsaacClient):
    """Same length-prefixed JSON + npz transport as the Isaac client."""

    def episodes(self) -> list[dict]:
        return self.call({"cmd": "episodes"})[0]["episodes"]

    def load(self, scene: str, episode_id: str) -> dict:
        return self.call({"cmd": "load", "scene": scene, "episode_id": str(episode_id)})[0]

    def lattice(self, step: float, start, all_islands: bool = False) -> tuple[dict, dict]:
        return self.call({"cmd": "lattice", "step": float(step), "start": [float(x) for x in start], "all_islands": bool(all_islands)})

    def geodesic(self, start, ends: np.ndarray) -> np.ndarray:
        return self.call({"cmd": "geodesic", "start": [float(x) for x in start]},
                         {"ends": np.asarray(ends, dtype=np.float64).reshape(-1, 3)})[1]["dist"]

    def render(self, poses: np.ndarray) -> dict:
        return self.call({"cmd": "render"}, {"poses": np.asarray(poses, dtype=np.float64).reshape(-1, 5)})[1]

    def surfaces(self, lo, hi, spacing: float, categories: list[str], with_ids: bool = False):
        h, a = self.call({"cmd": "surfaces", "lo": [float(x) for x in lo], "hi": [float(x) for x in hi],
                          "spacing": float(spacing), "categories": list(categories)})
        return (h, a["points"], a["semantic_ids"]) if with_ids else (h, a["points"])


# the text query per ObjectNav category: the task specification handed to the robot
OPEN_SET_QUERY_HSSD = {"bed": "bed", "chair": "chair", "couch": "couch", "potted_plant": "potted plant", "toilet": "toilet",
                       "tv": "tv"}


def hab_box_to_world(lo, hi) -> tuple[np.ndarray, np.ndarray]:
    """A Habitat axis-aligned box as a world one: (x, y, z)_hab -> (x, -z, y)."""
    lo, hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    return np.array([lo[0], -hi[2], lo[1]]), np.array([hi[0], -lo[2], hi[1]])


def start_yaw(rotation_xyzw) -> float:
    """The world yaw psi of a Habitat agent rotation about +y (quaternion x, y, z, w): habitat yaw theta looks along
    (-sin theta, 0, -cos theta), i.e. world (-sin theta, cos theta), so psi = theta + pi/2."""
    q = np.asarray(rotation_xyzw, dtype=np.float64)
    return float(2.0 * np.arctan2(q[1], q[3]) + np.pi / 2)


@dataclass
class HssdGroundTruth(EvaluatorGroundTruth):
    """Evaluator only. goals: [{semantic_id, lo, hi (world box), center}]; the single-target fields describe the first
    goal and exist for the shared records."""

    goals: list = field(default_factory=list)
    geodesic_distance: float = float("nan")  # the episode's shortest path to a goal view point (SPL)


class HssdEnvironment(Environment):
    def __init__(self, client: HssdClient, cfg: dict, scene: str, episode_id: str, robot_camera: CameraModel):
        self.client, self.cfg = client, cfg
        hc = cfg["hssd"]
        self.hc = hc
        h = client.load(scene, episode_id)
        if not h["goals"]:
            raise RuntimeError(f"{scene} {episode_id}: none of the goal objects is in the scene ({h.get('missing_goals')})")
        if os.path.basename(str(h.get("scene_data", ""))) != hc["scene_data"]:
            raise RuntimeError(f"the server's scene data is {h.get('scene_data')}, the config wants {hc['scene_data']} "
                               "(start it with --data-root on a root whose data/scene_datasets/hssd-hab is that copy)")
        self.header = h
        self.scene_name, self.episode_id, self.category = scene, str(episode_id), h["category"]
        self.robot_camera = robot_camera
        # ---- pose positions: the start first, then the lattice on its island
        lh, la = client.lattice(float(hc["pose_lattice_m"]), h["start_position"])
        start = np.asarray(h["start_position"], dtype=np.float64)
        self.base_hab = np.vstack([start[None], np.asarray(la["positions"], dtype=np.float64).reshape(-1, 3)])
        # the floor surface under each position (world z == Habitat y); the start's is not needed (it is never a candidate)
        self.floor_z = np.concatenate([[np.nan], np.asarray(la["floor_y"], dtype=np.float64)])
        self.sloped = np.concatenate([[False], np.asarray(la["sloped"], dtype=bool)])
        self.island = int(lh["island"])
        self.base_world = hab_to_world(self.base_hab)
        sh = float(hc["sensor_height_m"])
        cam = hab_to_world(self.base_hab + np.array([0.0, sh, 0.0]))
        rc = cfg["cameras"]["robot"]
        yaws = np.deg2rad(np.asarray(rc["yaw_degrees"], dtype=np.float64))
        pitches = np.deg2rad(np.asarray(rc["pitch_degrees"], dtype=np.float64))
        n_pos = len(cam)
        gi = np.stack(np.meshgrid(np.arange(n_pos), [0], np.arange(len(yaws)), np.arange(len(pitches)), indexing="ij"),
                      axis=-1).reshape(-1, 4)
        P = len(gi)
        yaw, pitch = yaws[gi[:, 2]], pitches[gi[:, 3]]
        T = np.stack([world_from_camera(cam[gi[i, 0]], yaw[i], pitch[i]) for i in range(P)])
        self.pool = CandidatePool(np.arange(P), gi, cam[gi[:, 0]], yaw, pitch, T, np.ones(P, dtype=bool),
                                  np.zeros(0), np.zeros(0), yaws, pitches, float("nan"))
        y0 = start_yaw(h["start_rotation"])
        iyaw = int(np.argmin(np.abs((yaws - y0 + np.pi) % (2 * np.pi) - np.pi)))
        ipitch = int(np.argmin(np.abs(pitches)))
        self.robot_start_pose_id = int(np.nonzero((gi[:, 0] == 0) & (gi[:, 2] == iyaw) & (gi[:, 3] == ipitch))[0][0])
        # ---- goals (evaluator only)
        goals = []
        for g in h["goals"]:
            lo, hi = hab_box_to_world(g["aabb_lo"], g["aabb_hi"])
            goals.append({"semantic_id": int(g["semantic_id"]), "handle": g["handle"], "lo": lo, "hi": hi,
                          "center": 0.5 * (lo + hi)})
        self.goal_ids = np.asarray([g["semantic_id"] for g in goals], dtype=np.int64)
        # ---- the detection channel: the oracle, optionally corrupted (fn_rate, fp_rate), or an open-set model
        from bamos.experiment.benchmark import detector_from_cfg  # benchmark imports the environments

        self.detector = detector_from_cfg(cfg)
        self.open_set_client = None  # set by the caller for sensor_detector.mode == "open_set"
        self.open_set_query = OPEN_SET_QUERY_HSSD.get(self.category, self.category.replace("_", " "))
        self.open_set_record_instances = False
        self.last_ids: np.ndarray | None = None
        # objects a false mask may be pinned on: rigid objects that are not goals and not part of the known shape
        structural = [w.lower() for w in hc["known_shape_categories"]]
        self._fp_instance_ids = np.array([int(i) for i, c in h.get("objects", [])
                                          if int(i) not in set(self.goal_ids.tolist())
                                          and not any(w in str(c).lower() for w in structural)], dtype=np.int64)
        # ---- the floors the candidate cells stand on (hssd.candidate_floors): the start island's pose positions, or
        # ("all") the navmesh lattice of every island, every floor of the house, though the robot never leaves its island
        if hc.get("candidate_floors", "start_island") == "all":
            _, sa = client.lattice(float(hc["pose_lattice_m"]), h["start_position"], all_islands=True)
            self.support_world = hab_to_world(np.asarray(sa["positions"], dtype=np.float64).reshape(-1, 3))
            self.support_floor_z = np.asarray(sa["floor_y"], dtype=np.float64)
        else:
            self.support_world = self.base_world[1:] if n_pos > 1 else self.base_world
            self.support_floor_z = self.floor_z[1:] if n_pos > 1 else self.base_world[:, 2]
        # ---- the workspace box: those floors, widened by the candidate dilation and a margin
        lat, fz = self.support_world, self.support_floor_z
        pad = float(hc["candidate_dilate_m"]) + float(hc["workspace_margin_m"])
        lo = np.array([lat[:, 0].min() - pad, lat[:, 1].min() - pad, np.nanmin(fz) - 0.3])
        hi = np.array([lat[:, 0].max() + pad, lat[:, 1].max() + pad, np.nanmax(fz) + float(hc["workspace_top_m"])])
        g0 = goals[0]
        self.scene = SceneDescription(
            f"hssd_{scene}", "hssd", tuple(lo), tuple(hi), [], [], sh, (float(cam[0, 0]), float(cam[0, 1]), y0, 0.0),
            g0["semantic_id"], tuple(g0["center"]), tuple(g0["hi"] - g0["lo"]), self.category, "n/a", g0["handle"],
            {"scene": scene, "episode_id": str(episode_id), "navigable_area": h["navigable_area"], "island": self.island,
             "n_positions": n_pos})
        self._gt = HssdGroundTruth(g0["center"], g0["semantic_id"], g0["hi"] - g0["lo"], self.category, "n/a", g0["handle"],
                                   np.zeros(0, dtype=np.int64), -1, {}, {}, "n/a", goals, float(h["geodesic_distance"]))
        self._geo: dict[int, dict[int, float]] = {}
        self._quality: dict[str, dict] = {}
        self._capture_counter = 0
        self._known_points = None

    # ------------------------------------------------------------------------------------------------ geometry
    def geodesic(self, from_pos: int, to_pos: np.ndarray) -> np.ndarray:
        """Navmesh geodesic distances [len(to_pos)] between pose positions (indices into base_hab); cached."""
        cache = self._geo.setdefault(int(from_pos), {})
        to_pos = np.asarray(to_pos, dtype=np.int64)
        need = [int(j) for j in np.unique(to_pos) if int(j) not in cache]
        if need:
            d = self.client.geodesic(self.base_hab[int(from_pos)], self.base_hab[need])
            cache.update({j: float(x) for j, x in zip(need, d)})
        return np.array([cache[int(j)] for j in to_pos], dtype=np.float64)

    def path_costs(self, from_pose_id: int, to_pose_ids: np.ndarray) -> np.ndarray:
        gi = self.pool.grid_index
        return self.geodesic(gi[int(from_pose_id), 0], gi[np.asarray(to_pose_ids, dtype=np.int64), 0])

    def known_points(self) -> np.ndarray:
        """[M, 3] world surface points of the known shape (cached per scene on disk)."""
        if self._known_points is None:
            hc = self.hc
            key = hashlib.sha1(json.dumps([sorted(hc["known_shape_categories"]), hc["known_shape_spacing_m"],
                                           hc["workspace_top_m"]]).encode()).hexdigest()[:10]
            path = Path(hc["cache_dir"]) / "known_shape" / f"{self.scene_name}_{key}.npz"
            if path.exists():
                with np.load(path) as z:
                    pts = z["points"]
            else:
                lo, hi = (np.asarray(b, dtype=np.float64) for b in self.header["navmesh_bounds"])
                pad = np.array([2.5, 0.5, 2.5])
                top = np.array([0.0, float(hc["workspace_top_m"]) + 0.5, 0.0])
                _, pts = self.client.surfaces(lo - pad, hi + pad + top, float(hc["known_shape_spacing_m"]),
                                              list(hc["known_shape_categories"]))
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(f".{os.getpid()}.tmp.npz")
                np.savez_compressed(tmp, points=pts.astype(np.float32))
                os.replace(tmp, path)
            self._known_points = hab_to_world(np.asarray(pts, dtype=np.float64))
        return self._known_points

    def true_points(self) -> np.ndarray:
        """[M, 3] world surface points of the true geometry the episode's search may know (mapping.occupancy_source =
        ground_truth, the world-model comparison): the stage and every object but this episode's goal instances. The
        scene's surfaces with their semantic ids are cached on disk; the goals are dropped per episode."""
        if getattr(self, "_true_points", None) is None:
            hc = self.hc
            key = hashlib.sha1(json.dumps(["*", hc["known_shape_spacing_m"], hc["workspace_top_m"]]).encode()).hexdigest()[:10]
            path = Path(hc["cache_dir"]) / "true_shape" / f"{self.scene_name}_{key}.npz"
            if path.exists():
                with np.load(path) as z:
                    pts, sem = z["points"], z["semantic_ids"]
            else:
                lo, hi = (np.asarray(b, dtype=np.float64) for b in self.header["navmesh_bounds"])
                pad = np.array([2.5, 0.5, 2.5])
                top = np.array([0.0, float(hc["workspace_top_m"]) + 0.5, 0.0])
                _, pts, sem = self.client.surfaces(lo - pad, hi + pad + top, float(hc["known_shape_spacing_m"]), ["*"], with_ids=True)
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(f".{os.getpid()}.tmp.npz")
                np.savez_compressed(tmp, points=pts.astype(np.float32), semantic_ids=sem.astype(np.int64))
                os.replace(tmp, path)
            keep = ~np.isin(np.asarray(sem), self.goal_ids)
            self._true_points = hab_to_world(np.asarray(pts, dtype=np.float64)[keep])
        return self._true_points

    def true_occupancy(self, grid: VoxelGrid) -> np.ndarray:
        """[n_vox] bool, C order: the grid's voxels holding a surface point of the true geometry (goals left out)."""
        return self._voxelize(self.true_points(), grid)

    @staticmethod
    def _voxelize(p: np.ndarray, grid: VoxelGrid) -> np.ndarray:
        ijk = np.floor((p - grid.lower) / grid.voxel_size).astype(np.int64)
        ok = np.all((ijk >= 0) & (ijk < np.asarray(grid.shape)), axis=1)
        occ = np.zeros(grid.n_voxels, dtype=bool)
        nx, ny, nz = grid.shape
        occ[(ijk[ok, 0] * ny + ijk[ok, 1]) * nz + ijk[ok, 2]] = True
        return occ

    def known_occupancy(self, grid: VoxelGrid) -> np.ndarray:
        """[n_vox] bool, C order: the grid's voxels holding a surface point of the known shape."""
        p = self.known_points()
        ijk = np.floor((p - grid.lower) / grid.voxel_size).astype(np.int64)
        ok = np.all((ijk >= 0) & (ijk < np.asarray(grid.shape)), axis=1)
        occ = np.zeros(grid.n_voxels, dtype=bool)
        nx, ny, nz = grid.shape
        occ[(ijk[ok, 0] * ny + ijk[ok, 1]) * nz + ijk[ok, 2]] = True
        return occ

    # ------------------------------------------------------------------------------------------------ sensing
    def observe_robot(self, pose_id: int, rng: np.random.Generator, step_index: int) -> PlannerObservation:
        pid = int(pose_id)
        ipos = int(self.pool.grid_index[pid, 0])
        b = self.base_hab[ipos]
        theta = float(self.pool.yaw[pid]) - np.pi / 2  # habitat yaw (start_yaw)
        r = self.client.render(np.array([[b[0], b[1], b[2], theta, float(self.pool.pitch[pid])]]))
        cam = self.robot_camera
        raw = np.asarray(r["depth"][0], dtype=np.float64)
        ignore = (raw > 0) & (raw < cam.near)  # clipped by the benchmark sensor: somewhere nearer, not where
        depth = np.where((raw <= 0) | (raw > cam.far), np.inf, raw)
        depth[ignore] = np.nan
        ids = np.asarray(r["ids"][0], dtype=np.int64)
        self.last_ids = ids  # calibration only (scripts/hssd/calibrate.py: which goal a reported mask lies on)
        valid = np.isfinite(depth)
        min_px = int(self.hc["oracle_min_px"])
        counts = {int(g): int(((ids == g) & valid).sum()) for g in self.goal_ids}
        kept = sorted([g for g, c in counts.items() if c >= min_px], key=lambda g: -counts[g])
        det = bool(kept)
        instance_masks = [ids == g for g in kept] if len(kept) > 1 else None
        mask = (np.logical_or.reduce(np.stack(instance_masks)) if instance_masks else (ids == kept[0])) if det \
            else np.zeros_like(valid)
        status = DetectionStatus.DET if det else DetectionStatus.NO_DET
        noise: dict = {}
        if self.detector.mode == "open_set" or self.detector.corrupts:
            gt_union = np.isin(ids, self.goal_ids)
            pose_key = f"hssd|{self.scene_name}|" + ",".join(f"{x:.3f}" for x in b) + f"|{theta:.4f}|{float(self.pool.pitch[pid]):.4f}"
            if self.detector.mode == "open_set":
                from bamos.environment.open_set import open_set_observe  # imports the Isaac client

                if self.open_set_client is None:
                    raise RuntimeError("sensor_detector.mode=open_set needs a detector client (scripts/hssd/run.py --detector-port)")
                status, mask, noise = open_set_observe(
                    self.open_set_client, self.open_set_query, pose_key, np.asarray(r["rgb"][0])[..., :3], depth, ids,
                    gt_union, det, cam, self.pool.world_from_camera[pid], self.detector.open_set_min_mask_px,
                    self.detector.open_set_eval_iou, self._fp_instance_ids, self.detector.open_set_mask_select,
                    self.detector.open_set_min_score, self.open_set_record_instances, self.detector.open_set_multi_mask)
                if mask is None:
                    mask = np.zeros_like(valid)
                instance_masks = noise.pop("instance_masks", None)
                instance_masks = instance_masks if instance_masks is not None and len(instance_masks) > 1 else None
                if instance_masks is not None:
                    mask = np.logical_or.reduce(np.stack(instance_masks))
            else:
                # one stream per (scene, episode, pose): every arm meets the same corruption at the same pose
                rng = np.random.default_rng([zlib.crc32(f"{self.episode_id}|{pose_key}".encode()) & 0xFFFFFFFF])
                reported, cmask, noise = corrupt_detection(det, mask, ids, self.detector, rng, self._fp_instance_ids)
                if noise.get("fn") or noise.get("fp"):
                    instance_masks = None
                    mask = cmask
                status = DetectionStatus.DET if reported else DetectionStatus.NO_DET
        best = counts[kept[0]] if det else max(counts.values(), default=0)
        sel = (ids == kept[0]) if det else np.zeros_like(valid)
        quality = {"visible_fraction": float("nan"), "area_px": int(best), "iso_area_px": -1,
                   "normalized_area": best / float(cam.n_pixels),
                   "truncated": bool(sel[0, :].any() or sel[-1, :].any() or sel[:, 0].any() or sel[:, -1].any()),
                   "gt_detected": det, "true_world_from_camera": self.pool.world_from_camera[pid],
                   "goal_px": {str(g): c for g, c in counts.items() if c}}
        self._capture_counter += 1
        cap = f"robot_{self._capture_counter:04d}"
        meta = {"depth_ignore": ignore, "gt_mask_area_px": int(best), "noise_applied": noise,
                "n_goal_instances_detected": len(kept)}
        if "wrong_instance_id" in noise:
            meta["wrong_instance_id"] = noise["wrong_instance_id"]
        obs = PlannerObservation("robot", "robot", cap, cam, self.pool.world_from_camera[pid], depth, mask,
                                 status, True,
                                 np.asarray(r["rgb"][0]), pid, meta, instance_masks)
        self._quality[cap] = quality
        return obs

    def ground_truth(self) -> HssdGroundTruth:
        return self._gt

    def gt_quality_of_observation(self, obs: PlannerObservation) -> dict:
        return self._quality[obs.capture_id]
