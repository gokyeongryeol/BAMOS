"""Deterministic test-split episode specs and environment construction (Isaac backend)."""
from __future__ import annotations

import subprocess
from pathlib import Path

from bamos.environment.episodes import EpisodeSpec, build_episode
from bamos.environment.isaac_env import IsaacClient, IsaacEnvironment
from bamos.environment.open_set import OPEN_SET_QUERY
from bamos.environment.sensor import DetectorConfig
from bamos.planning.runner import camera_from_cfg

# Disjoint layout ranges. "calib" fits the detector constants and selects the settings; "test" is run once, from the
# frozen settings (2026-10-06 re-run: fresh calib and test layouts). "dev" (base 2000) and "dev2" (base 4000) are former
# test splits whose results were seen while the model was still changing, so they only serve development (smoke runs,
# figures).
SPLIT_LAYOUT_BASE = {"dev": 2000, "dev2": 4000, "calib": 10000, "test": 11000}
# every target type in both placements; consecutive entries alternate shape and size, and the calib stride (5) with the
# per-layout offset visits all twelve
TARGET_MODE_CYCLE = [("cube10", "on_top"), ("sphere20", "inside"), ("cube30", "on_top"), ("sphere10", "inside"),
                     ("cube20", "on_top"), ("sphere30", "inside"), ("cube10", "inside"), ("sphere20", "on_top"),
                     ("cube30", "inside"), ("sphere10", "on_top"), ("cube20", "inside"), ("sphere30", "on_top")]


def code_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, cwd=Path(__file__).parent).decode().strip()
    except Exception:
        return "no-git"


def make_episode_specs(cfg: dict, split: str = "test") -> list[EpisodeSpec]:
    """Layout-index-major order (identical to the benchmark that produced the frames)."""
    b = cfg["benchmark"]
    specs = []
    for ai, arch in enumerate(cfg["environment"]["archetypes"]):
        layouts, per = b["test_layouts_per_archetype"], b["test_episodes_per_layout"]
        for li in range(layouts):
            layout_seed = SPLIT_LAYOUT_BASE[split] + 100 * ai + li
            layout_id = f"{arch}_{split}_L{li}"
            for ei in range(per):
                tt, mode = TARGET_MODE_CYCLE[(ei + li) % len(TARGET_MODE_CYCLE)]
                ep_seed = layout_seed * 1000 + ei
                specs.append((li, ai, ei, EpisodeSpec(f"{layout_id}_E{ei}", layout_id, arch, layout_seed, ep_seed, tt, mode, split)))
    specs.sort(key=lambda t: (t[0], t[1], t[2]))
    return [t[3] for t in specs]


def detector_from_cfg(cfg: dict) -> DetectorConfig:
    sd = cfg["sensor_detector"]
    return DetectorConfig(
        sd["mode"], sd["min_visible_fraction"], sd["min_normalized_area"],
        float(sd.get("fn_rate", 0.0)), float(sd.get("fp_rate", 0.0)),
        str(sd.get("fp_mask_source", "instance")),
        int(sd.get("open_set_min_mask_px", 3)), float(sd.get("open_set_eval_iou", 0.25)),
        str(sd.get("open_set_mask_select", "largest")), float(sd.get("open_set_min_score", 0.0)),
        bool(sd.get("open_set_multi_mask", False)),
    )


def build_env_for_spec(spec: EpisodeSpec, cfg: dict, manifest_dir: Path | None, isaac_env: IsaacClient | None,
                       detector_client=None):
    import json

    from bamos.evaluation.records import _jsonable

    rc = camera_from_cfg(cfg["cameras"]["robot"])
    env, manifest = build_episode(spec, cfg, rc, detector_from_cfg(cfg))
    if manifest_dir is not None:
        manifest_dir.mkdir(parents=True, exist_ok=True)
        with open(manifest_dir / f"{spec.episode_id}.json", "w") as f:
            json.dump(_jsonable(manifest), f)
    if cfg["environment"]["backend"] == "isaac":
        env = IsaacEnvironment.wrap(env, cfg, isaac_env)
    if env.detector.mode == "open_set":
        if detector_client is None:
            raise RuntimeError("sensor_detector.mode=open_set needs a detector client (--detector-ports)")
        # the text query is the task specification handed to the robot, not ground truth
        env.open_set_client, env.open_set_query = detector_client, OPEN_SET_QUERY[spec.target_type]
    return env, manifest
