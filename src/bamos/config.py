"""Configuration: one YAML (configs/isaac.yaml) merged over the defaults below, plus a content hash.

Every key a YAML or an override sets must exist in DEFAULT_CONFIG: a stale or misspelt key is an error, not a
silently ignored setting."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

DEFAULT_CONFIG: dict = {
    "experiment": {"include_robot_start_view": True, "additional_robot_view_budget": 60, "forbid_exact_revisit": True, "device": "cuda:0", "dtype": "float64"},
    "environment": {
        "backend": "isaac",  # isaac | analytic (exact ray caster, no render server; used for batch experiments) | habitat (HSSD)
        "archetypes": ["warehouse_aisles", "industrial_workcell", "dense_storage"],
        "isaac": {"host": "127.0.0.1", "port": 5555, "render_cache_dir": None, "validate_against_analytic": True},
    },
    "cameras": {
        "robot": {"z_world_m": 1.50, "x_step_m": 0.50, "y_step_m": 0.50, "lattice_offset_m": 0.0137, "yaw_degrees": [0, 45, 90, 135, 180, 225, 270, 315], "pitch_degrees": [-30, 0], "robot_clearance_m": 0.25, "width": 320, "height": 240, "hfov_deg": 70.0, "near_m": 0.1, "far_m": 15.0},
    },
    # the lattice the target-location candidates are drawn from (spec 5)
    # voxel_size_m: the target belief's voxel (the candidate lattice is drawn from it). The voxel belief of spec 4 has
    # its own finer grid over the same box; occupancy_model is its [p0, q_occ, q_free] (prior occupancy, the hit
    # probability of a ray entering an occupied / a free voxel), fitted on calib (stage 2)
    "mapping": {"voxel_size_m": 0.10, "grid_offset_m": 0.0037, "occupancy_voxel_m": 0.05,
                "occupancy_model": [0.03, 0.6, 0.005], "occupancy_pixel_stride": 2, "occupancy_ray_margin_m": 0.10,
                # "ground_truth": the true scene geometry instead of b_t (a perfect scene model; 2026-10-08)
                "occupancy_source": "belief",
                # a depth return nearer than the camera's near limit (clipped by the sensor, given as ignored pixels):
                # True, a hit at the near limit; False, neither hit nor pass (the Isaac benchmark has none)
                "near_hits": False},
    # each candidate cell's likelihood is the mean over the 8 octant centres of the cell (spec 7.3), the belief being over
    # cells, not their centres
    "target": {"center_stride_voxels": 2, "candidate_z_range_m": [0.15, 1.75], "candidate_wall_clearance_m": 0.15},
    # detection_gain is eta_t, the detection ceiling, and false_positive_rate is eta_f, the floor of r (spec 7.2).
    # h_area0 is the detection threshold a0 = (1/100)^2, the oracle's DET area threshold, and h_slope the log-area
    # slope k of the soft threshold a^k / (a^k + a0^k) (spec 7.1).
    "detection_likelihood": {"detection_gain": 0.95, "false_positive_rate": 0.01, "h_area0": 0.0001, "h_slope": 3.0},
    # sensor_detector: the oracle detection channel plus the controlled corruption applied to it before the
    # planner sees it (outline S5). fn_rate/fp_rate default to 0, i.e. the oracle channel unchanged.
    "sensor_detector": {"mode": "rendered_threshold", "min_visible_fraction": 0.01, "min_normalized_area": 0.0001,
                        "fn_rate": 0.0, "fp_rate": 0.0, "fp_mask_source": "instance",
                        # mode "open_set": a text-prompted segmentation model instead of the oracle (Isaac backend only)
                        "open_set_min_mask_px": 3, "open_set_eval_iou": 0.25, "open_set_mask_select": "largest",
                        "open_set_min_score": 0.0,  # > 0: raise a scoring detector's instance threshold (SAM 3)
                        # True: every kept instance enters the DET update, at most one of them the target, each with
                        # probability 1/M (formulation §8.2 extension, 2026-10-07); False: the one mask of open_set_mask_select
                        "open_set_multi_mask": False,
                        "open_set_cache_dir": "results/cache/open_set"},
    # Component ablations of BAMOS (outline D2-D5). All default to the full model.
    "ablation": {"use_kappa": True, "use_nodet": True, "use_voxel_belief": True},
    "planner": {
        "tie_tolerance": 1.0e-12, "ig_negative_tolerance": 1.0e-9,
        "utility_mode": "ig",  # | coverage | scene_coverage (policies.PolicyConfig)
        "motion_scale_m": 5.0,  # scale of the logged normalized path cost; never scored
        # size hypotheses S (spec 3) and kappa (spec 8.2): m_q = x_v + beta s n, Sigma = (lambda s)^2 I, a Gaussian
        # mixed with a uniform false-positive component (weight eta_f / r)
        "size_grid_m": [0.1, 0.2, 0.3],  # size scales sqrt(surface area / 4), not diameters
        "kappa_beta": 0.3333, "kappa_lambda": 0.2887,
    },
    # confirmations: S_hat = 1 steps at one location needed to declare (1 = spec 10; >1 is the stop ablation)
    # present_views: after S_hat = 1, this many views chosen to show the MAP cell (spec 10); a DET there with pi(v_hat) >=
    # p_succ (sum_c omega(c - v_hat) pi(c), the probability the declaration succeeds; formulation §10) >= tau_pi and the
    # MAP in place declares, otherwise the declaration is withdrawn
    "stop": {"success_radius_m": 0.2, "tau_pi": 0.5, "mode": "joint", "confirmations": 1, "present_views": 1,
             "rule": "belief"},
    # success: the MAP is within max_center_error_m of the target at the declaration (evaluation/evaluator.py). The
    # view thresholds only define solvability: an episode is built only if some pose has the oracle detecting the
    # target, mostly unoccluded and at least min_gt_area_px pixels, the smallest mask a detector may report
    # (environment/episodes.py). How well the declared view shows the target is not part of the test, since the
    # model only localises the target
    "evaluation": {"max_center_error_m": 0.20, "min_gt_visible_fraction": 0.50, "min_gt_area_px": 3},
    "benchmark": {"test_layouts_per_archetype": 5, "test_episodes_per_layout": 20},
    # The benchmark's difficulty (2026-10-01; environment/episodes.py), every axis on: a clutter box between the target
    # and the open side of its anchor, covering occluder_cover of it; robot poses closer than min_robot_distance_m to
    # the target are infeasible; the target centre lattice_offset_m off the candidate lattice; and the oracle does not
    # detect the target from the robot start view
    "difficulty": {"occluder_cover": 0.75, "min_robot_distance_m": 1.2, "lattice_offset_m": 0.1},
    # HSSD ObjectNav with teleport poses (environment.backend: habitat; environment/hssd_env.py, planning/hssd_runner.py,
    # servers/environment/hssd_server.py). Unused by the Isaac benchmark.
    "hssd": {
        "host": "127.0.0.1", "split": "val",
        # pose positions: navmesh points on this xz lattice, on the start's navmesh island; the camera is the benchmark's
        # (cameras.robot), sensor_height above the navmesh point
        "pose_lattice_m": 0.5, "sensor_height_m": 0.88,
        # the next pose: navmesh geodesic <= max_geodesic_m from the current one, and its floor seen in an earlier frame:
        # an up-facing depth point (below the camera) within floor_tol_m of the floor surface under it (floor_tol_sloped_m on
        # stairs and ramps), in the floor_seen_cell_m xy cell of the position or a neighbour of it
        "max_geodesic_m": 4.0, "floor_tol_m": 0.10, "floor_tol_sloped_m": 0.15, "floor_seen_cell_m": 0.1,
        "floor_seen_bin_m": 0.05,
        # candidate cells (option ii): the space above the floor of the start's navmesh island, within candidate_dilate_m of
        # a pose position horizontally (furniture stands where the navmesh is not), candidate_band_m above that floor, not
        # behind the known shape seen from the position, not inside it. On train (v0.2.4): 97.4% of the (scene, category)
        # episodes have a goal centre within 1.5 m and 0.0-2.3 m (99.4% within 2.0 m); 0.65% of goal centres lie outside
        # 0.0-2.3 m (0.15-1.85 m: 4.1%, TVs 11.9%)
        "candidate_dilate_m": 1.5, "candidate_band_m": [0.0, 2.3], "workspace_top_m": 3.0, "workspace_margin_m": 0.5,
        # the floors the cells stand on: "all" (every navmesh island's floor, all storeys, as the house's shape is known; the
        # robot still moves on its start island only) or "start_island" (option ii). "all" since 2026-10-09: with
        # "start_island" the 9 val episodes that start on a roof slab had no cell near their goal (SR 1/9 -> 7/9; a random
        # 60-episode sample 55 -> 53, its changes all budget-bound searches in both directions)
        "candidate_floors": "all",
        # the known shape: the stage and the objects whose objects.csv category contains one of these words, from ray sweeps
        # every known_shape_spacing_m (cached per scene in cache_dir)
        "known_shape_categories": ["window", "railing", "stair", "door"], "known_shape_spacing_m": 0.05,
        # the oracle detector: a goal instance with >= oracle_min_px pixels of valid depth is a detection
        "oracle_min_px": 150,
        # size hypotheses per category: the {p10, p50, p90} of the train goal instances' box size scale sqrt(area / 4)
        # (scripts/hssd/scene_stats.py, 122 train scenes, 3775 instances, v0.2.4 scene data)
        "size_grid_by_category": {"bed": [1.72, 2.06, 2.47], "chair": [0.72, 0.84, 1.23], "couch": [1.26, 1.58, 2.45],
                                  "potted_plant": [0.28, 0.64, 1.47], "toilet": [0.57, 0.73, 0.79], "tv": [0.63, 0.77, 1.21]},
        # success: the declared cell's centre within evaluation.max_center_error_m of a goal instance's box; also logged:
        # the declaring camera within viewpoint_success_m (horizontally) of a goal's box (the benchmark's view points)
        "viewpoint_success_m": 1.0,
        # the server's scene assets (the basename of its data/scene_datasets/hssd-hab): the episodes were generated on
        # hssd-hab v0.2.4 (stored geodesics and start heights reproduce exactly); v0.2.5's object colliders change the navmesh
        "scene_data": "hssd-hab-v0.2.4-meta",
        "cache_dir": "results/cache/hssd_v024",
    },
}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def check_keys(override: dict, base: dict = DEFAULT_CONFIG, where: str = "") -> None:
    """Raise KeyError for any key of `override` (recursively) that `base` does not define."""
    for k, v in (override or {}).items():
        path = f"{where}.{k}" if where else k
        if k not in base:
            raise KeyError(f"unknown config key {path!r}")
        if isinstance(v, dict) and isinstance(base[k], dict):
            check_keys(v, base[k], path)


def load_config(path: str | Path | None = "configs/isaac.yaml", overrides: list[dict] | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path is not None:
        with open(path) as f:
            y = yaml.safe_load(f) or {}
        check_keys(y)
        cfg = deep_merge(cfg, y)
    for o in overrides or []:
        check_keys(o)
        cfg = deep_merge(cfg, o)
    return cfg


def config_hash(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]
