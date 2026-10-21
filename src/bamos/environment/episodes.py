"""Episode construction: layout -> target placement -> pool -> solvability check -> manifest.

Main-set episodes must have at least one reachable robot pose in the final discrete pool that yields DET and meets
the GT quality thresholds. Placement is retried (new target seed) until satisfied; the number of
rejected placements is recorded in the manifest.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bamos.environment.analytic_env import AnalyticEnvironment
from bamos.environment.base import EvaluatorGroundTruth
from bamos.environment.layouts import ARCHETYPES, LayoutBuilder, build_layout
from bamos.environment.pathing import NavigationOracle
from bamos.environment.scene import SceneDescription
from bamos.environment.sensor import DetectorConfig, threshold_detect
from bamos.mapping.camera import CameraModel
from bamos.planning.candidates import CandidatePool


@dataclass
class EpisodeSpec:
    episode_id: str
    layout_id: str
    archetype: str
    layout_seed: int
    episode_seed: int
    target_type: str
    placement_mode: str
    split: str


def make_pool(env_cfg: dict, W: float, D: float, nav: NavigationOracle, start_xy, keep_out=None) -> tuple[CandidatePool, np.ndarray, np.ndarray]:
    rc = env_cfg["cameras"]["robot"]
    clear = float(rc["robot_clearance_m"])
    xs = np.arange(clear + 0.25, W - clear - 0.25 + 1e-9, rc["x_step_m"])
    ys = np.arange(clear + 0.25, D - clear - 0.25 + 1e-9, rc["y_step_m"])
    xs = xs + float(rc.get("lattice_offset_m", 0.0))
    ys = ys + float(rc.get("lattice_offset_m", 0.0))
    yaw = np.deg2rad(np.asarray(rc["yaw_degrees"], dtype=np.float64))
    pitch = np.deg2rad(np.asarray(rc["pitch_degrees"], dtype=np.float64))
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    xy = np.stack([X.ravel(), Y.ravel()], axis=1)
    feas = np.array([nav.free(p) for p in xy]) & nav.reachable_from(start_xy, xy)
    if keep_out is not None:
        centre, radius = keep_out  # difficulty axis 2: the robot cannot get closer than `radius` to the target
        feas &= np.linalg.norm(xy - np.asarray(centre, dtype=float)[None, :2], axis=1) >= float(radius)
    feasible_xy = feas.reshape(len(xs), len(ys))
    pool = CandidatePool.build(xs, ys, yaw, pitch, float(rc["z_world_m"]), feasible_xy)
    return pool, xs, ys


def episode_metadata(cfg, spec, scene, tgt, surf, b, env, pool, start_pose_id, start_q, quals, ok, feas_ids, detector,
                     n_rejected: int, occluded: bool) -> dict:
    """Per-episode features for later analysis (evaluator only; the planner never reads them)."""
    from bamos.environment.layouts import TARGET_SPECS
    c = np.asarray(scene.gt_target_center, dtype=float)
    lo, hi = tgt.aabb()
    sx, sy, syaw, spitch = scene.robot_start_xy_yaw_pitch
    start = np.array([sx, sy, float(cfg["cameras"]["robot"]["z_world_m"])])
    step = cfg["mapping"]["voxel_size_m"] * cfg["target"]["center_stride_voxels"]
    base = -cfg["mapping"]["grid_offset_m"] + 0.5 * cfg["mapping"]["voxel_size_m"]
    nearest = base + np.round((c - base) / step) * step
    det = np.array([threshold_detect(q, detector) for q in quals])
    W, D = scene.workspace_hi[0], scene.workspace_hi[1]
    q_ids = feas_ids[ok]
    pos = pool.positions[q_ids] if len(q_ids) else np.zeros((0, 3))
    d_quality = np.linalg.norm(pos[:, :2] - start[None, :2], axis=1) if len(q_ids) else np.array([np.nan])
    spec_t = TARGET_SPECS[spec.target_type]
    return {
        "target_type": spec.target_type, "target_shape": spec_t["shape"], "target_size_scale_m": spec_t["s"],
        "target_extent_m": [float(x) for x in (hi - lo)], "target_center": c.tolist(), "target_height_m": float(c[2]),
        "placement_mode": surf.mode, "placement_requested": spec.placement_mode, "anchor": surf.anchor,
        "anchor_type": next((o.meta.get("type", o.name) for o in b.objects if o.name == surf.anchor), surf.anchor),
        "occluder_in_front": occluded,
        "dist_start_to_target_xy_m": float(np.linalg.norm(c[:2] - start[:2])),
        "dist_target_to_nearest_wall_m": float(min(c[0], W - c[0], c[1], D - c[1])),
        "lattice_offset_to_nearest_cell_m": float(np.linalg.norm(c - nearest)),
        "start_view_visible_fraction": float(start_q["visible_fraction"]), "start_view_area_px": int(start_q["area_px"]),
        "start_view_target_in_frame_unoccluded_px": int(start_q.get("iso_area_px", 0)),
        "n_feasible_poses": int(len(feas_ids)), "n_detecting_poses": int(det.sum()), "n_quality_poses": int(ok.sum()),
        "share_detecting_poses": float(det.mean()) if len(det) else 0.0,
        "dist_start_to_nearest_quality_pose_xy_m": float(np.nanmin(d_quality)),
        "n_objects": int(len(b.objects)), "n_clutter": int(sum(o.role == "clutter" for o in b.objects)),
        "n_rejected_placements": int(n_rejected), "room_size_m": [float(W), float(D), float(scene.workspace_hi[2])],
    }


def build_episode(spec: EpisodeSpec, cfg: dict, robot_camera: CameraModel, detector: DetectorConfig, max_placement_tries: int = 40) -> tuple[AnalyticEnvironment, dict]:
    arch = ARCHETYPES[spec.archetype]
    W, D = arch["size"]
    Hc = arch["ceiling"]
    rc = cfg["cameras"]["robot"]
    evc = cfg["evaluation"]
    n_rejected = 0
    for attempt in range(max_placement_tries):
        b: LayoutBuilder = build_layout(spec.archetype, spec.layout_seed)
        # target placement uses the episode RNG (layout objects are identical for the same layout seed)
        b.rng = np.random.default_rng(spec.episode_seed * 1000 + attempt)
        placed = b.place_target(spec.target_type, spec.placement_mode)
        if placed is None:
            # fall back to the other placement mode if this layout has no suitable surface
            alt = "inside" if spec.placement_mode == "on_top" else "on_top"
            placed = b.place_target(spec.target_type, alt)
            if placed is None:
                n_rejected += 1
                continue
        tgt, surf = placed
        diff = cfg["difficulty"]
        lat = float(diff["lattice_offset_m"])
        # difficulty axis 3: put the target centre half a candidate cell off the lattice, where the 0.2 m
        # quantisation of C is worst. The lattice is the one make_candidates() builds from the workspace.
        lo0, hi0 = tgt.aabb()
        c0 = 0.5 * (lo0[:2] + hi0[:2])
        step = cfg["mapping"]["voxel_size_m"] * cfg["target"]["center_stride_voxels"]
        base = -cfg["mapping"]["grid_offset_m"] + 0.5 * cfg["mapping"]["voxel_size_m"]
        snapped = base + np.round((c0 - base) / step) * step + lat
        yaw = float(tgt.prims[0].yaw) if hasattr(tgt.prims[0], "yaw") else 0.0
        cand = b.shift_target(tgt, spec.target_type, (float(snapped[0]), float(snapped[1])), surf.z_top, yaw)
        clo, chi = cand.aabb()
        if surf.x0 <= clo[0] and chi[0] <= surf.x1 and surf.y0 <= clo[1] and chi[1] <= surf.y1:
            tgt = cand
        else:
            b.objects[b.objects.index(cand)] = tgt  # shift would fall off the surface: keep the original
        # difficulty axis 1: an occluder in front of the target, between it and the room centre
        anchor = next((o for o in b.objects if o.name == surf.anchor), None)
        if anchor is not None:
            alo, ahi = anchor.aabb()
            f = np.array([W / 2, D / 2]) - 0.5 * (alo[:2] + ahi[:2])
            n = float(np.linalg.norm(f))
            if n > 1e-6:
                b.occlude_target(tgt, surf, f / n, float(diff["occluder_cover"]))
        lo, hi = tgt.aabb()
        # no fixed cameras (K = 0); the scene manifest keeps the empty list, as the Isaac render cache keys hash it
        scene = SceneDescription(
            spec.layout_id, spec.archetype, (0.0, 0.0, 0.0), (W, D, Hc), b.objects, [], float(rc["z_world_m"]), (0, 0, 0, 0),
            tgt.instance_id, tuple(0.5 * (lo + hi)), tuple(hi - lo), spec.target_type, surf.mode, surf.anchor,
            {"W": W, "D": D, "Hc": Hc, "layout_seed": spec.layout_seed, "episode_seed": spec.episode_seed},
        )
        nav = NavigationOracle((0, 0, 0), (W, D, Hc), scene.obstacles_for_navigation(), rc["robot_clearance_m"], rc["z_world_m"])
        # robot start: fixed entrance corner (target independent), nearest free lattice point looking at the room center
        start_pref = np.array([rc["robot_clearance_m"] + 0.25 + rc.get("lattice_offset_m", 0.0), rc["robot_clearance_m"] + 0.25 + rc.get("lattice_offset_m", 0.0)])
        # difficulty axis 2: no robot pose within min_robot_distance_m of the target
        keep_out = (np.asarray(scene.gt_target_center, dtype=float), float(diff["min_robot_distance_m"]))
        _, xs, ys = make_pool(cfg, W, D, nav, start_pref, keep_out)
        free_xy = np.array([nav.free(p) for p in np.stack([np.repeat(xs, len(ys)), np.tile(ys, len(xs))], 1)])
        xy_all = np.stack([np.repeat(xs, len(ys)), np.tile(ys, len(xs))], 1)
        if not free_xy.any():
            n_rejected += 1
            continue
        d = np.linalg.norm(xy_all - start_pref, axis=1)
        d[~free_xy] = np.inf
        start_xy = xy_all[int(np.argmin(d))]
        pool, xs, ys = make_pool(cfg, W, D, nav, start_xy, keep_out)
        center_yaw = np.arctan2(D / 2 - start_xy[1], W / 2 - start_xy[0])
        yaws = pool.yaw_values
        iyaw = int(np.argmin(np.abs((yaws - center_yaw + np.pi) % (2 * np.pi) - np.pi)))
        ipitch = int(np.argmin(np.abs(pool.pitch_values)))
        ix = int(np.argmin(np.abs(xs - start_xy[0])))
        iy = int(np.argmin(np.abs(ys - start_xy[1])))
        start_pose_id = int(np.nonzero((pool.grid_index == np.array([ix, iy, iyaw, ipitch])).all(axis=1))[0][0])
        scene.robot_start_xy_yaw_pitch = (float(xs[ix]), float(ys[iy]), float(yaws[iyaw]), float(pool.pitch_values[ipitch]))
        env = AnalyticEnvironment(scene, robot_camera, pool, nav, start_pose_id, detector, corruption_seed=spec.episode_seed)
        # difficulty axis 4: the oracle does not detect the target from the robot start view, so finding it needs the
        # active views. Checked first: it is cheap.
        start_q = env.gt_quality_many(np.array([start_pose_id]))[0]
        if threshold_detect(start_q, detector):
            n_rejected += 1
            continue
        # solvability over the FINAL feasible pool
        feas_ids = pool.pose_ids[pool.feasible]
        quals = env.gt_quality_many(feas_ids)
        ok = np.array([threshold_detect(q, detector) and q["visible_fraction"] >= evc["min_gt_visible_fraction"] and q["area_px"] >= evc["min_gt_area_px"] for q in quals])
        if not ok.any():
            n_rejected += 1
            continue
        score = np.array([q["visible_fraction"] * (q["area_px"] >= evc["min_gt_area_px"]) for q in quals])
        best = int(np.argmax(score))
        fixed_vis, vis_class = {}, "hidden"  # no fixed cameras, and the start view does not detect the target
        meta = episode_metadata(cfg, spec, scene, tgt, surf, b, env, pool, start_pose_id, start_q, quals, ok, feas_ids, detector,
                                n_rejected, occluded=True)
        gt = EvaluatorGroundTruth(np.asarray(scene.gt_target_center), tgt.instance_id, np.asarray(scene.gt_target_extent), spec.target_type, surf.mode, surf.anchor, feas_ids[ok], int(feas_ids[best]), quals[best], fixed_vis, vis_class)
        env._gt = gt
        manifest = {
            "episode_id": spec.episode_id,
            "split": spec.split,
            "layout_id": spec.layout_id,
            "archetype": spec.archetype,
            "layout_seed": spec.layout_seed,
            "episode_seed": spec.episode_seed,
            "target_type": spec.target_type,
            "placement_mode_requested": spec.placement_mode,
            "n_rejected_placements": n_rejected,
            "scene": scene.to_manifest(),
            "robot_start_pose_id": start_pose_id,
            "pool": {
                "x_values": xs.tolist(), "y_values": ys.tolist(), "yaw_values_rad": pool.yaw_values.tolist(), "pitch_values_rad": pool.pitch_values.tolist(),
                "z_robot": pool.z_robot, "n_total": int(pool.n), "n_feasible": int(pool.feasible.sum()), "feasible_pose_ids": pool.pose_ids[pool.feasible].tolist(),
            },
            "cameras": {
                "robot": robot_camera.__dict__,
            },
            "evaluator_only": {
                "reachable_quality_pose_ids": gt.reachable_quality_pose_ids.tolist(),
                "n_reachable_quality_poses": int(len(gt.reachable_quality_pose_ids)),
                "best_quality_pose_id": gt.best_quality_pose_id,
                "best_quality": gt.best_quality,
                "fixed_camera_visibility": fixed_vis,
                "initial_visibility_class": vis_class,
                "episode_meta": meta,
            },
        }
        return env, manifest
    raise RuntimeError(f"could not build a solvable episode for {spec} after {max_placement_tries} tries")
