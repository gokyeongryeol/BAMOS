"""Re-run stage 1: what a detection channel reports as a function of the target's apparent area.

    # an oracle-family channel (analytic renderer, no servers)
    uv run python scripts/calibrate_channel.py --channel oracle --out results/raw/calib/oracle --gpu 0
    uv run python scripts/calibrate_channel.py --channel oracle --out results/raw/calib/s5_fp020 \
        --override sensor_detector.fp_rate=0.2
    # an open-set channel (Isaac RGB + a detector server)
    uv run python scripts/calibrate_channel.py --channel open_set --out results/raw/calib/falcon \
        --isaac-port 5555 --detector-port 5600 --shard 0/2

Frames come from the "calib" split, whose layouts are disjoint from the test split, so nothing fitted or chosen
on them sees a test episode (README.md, "Isaac Sim pipeline"). Every channel meets the same episodes and the same poses:
robot poses stratified over apparent area, drawn from the episode seed. Each frame goes
through the channel exactly as the planner receives it, so its status is the Z the belief update gets. The
apparent area is the target's unoccluded silhouette, the a_hat of spec 6 (a shape constant is absorbed in a0).

Open-set frames also record every instance the model returned ([area_px, score or -1, IoU with the GT mask, world
centroid]), so a score threshold can be applied offline and the channel's output under it reconstructed exactly.
Every frame records the reported mask's world centroid, the GT target centre and the camera position, from which
kappa's beta and lambda are fitted (scripts/fit_channel_constants.py). Each episode's frames also keep their depth and
reported mask (arrays/<episode>.npz, every mapping.occupancy_pixel_stride-th pixel), in the order they were observed,
so that stage 2 can rebuild the voxel belief frame by frame (scripts/calib_voxel_belief.py).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

FIELDS = ["episode_id", "archetype", "target_type", "camera_type", "camera_id", "pose_id", "a_iso",
          "visible_fraction", "in_view", "oracle_det", "status", "fp", "iou", "n_masks", "n_kept", "instances",
          # for kappa's beta and lambda: the reported mask's world centroid (spec 8.2, n_min = 3), the GT target
          # centre and the camera position, all [x, y, z] in metres
          "centroid", "gt_center", "cam_pos"]


def stratified(a: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Indices spread over log-area, a sixth of them from the zero-area (target out of view) stratum."""
    zero, pos = np.nonzero(a <= 0)[0], np.nonzero(a > 0)[0]
    out = []
    if len(zero):
        out.append(rng.choice(zero, size=min(len(zero), max(1, n // 6)), replace=False))
    if len(pos):
        edges = np.quantile(np.log10(a[pos]), np.linspace(0, 1, 9))
        per = max(1, (n - sum(len(o) for o in out)) // 8)
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = pos[(np.log10(a[pos]) >= lo) & (np.log10(a[pos]) <= hi)]
            if len(sel):
                out.append(rng.choice(sel, size=min(len(sel), per), replace=False))
    return np.unique(np.concatenate(out)) if out else np.zeros(0, dtype=int)


def parse_overrides(items: list[str]) -> dict:
    """dotted.key=value pairs, literal values only (as scripts/run_experiment.py)."""
    out: dict = {}
    for ov in items:
        key, _, val = ov.partition("=")
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        try:
            node[parts[-1]] = eval(val, {"__builtins__": {}}, {})
        except Exception:
            node[parts[-1]] = val
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", choices=["oracle", "open_set"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/normal.yaml")
    ap.add_argument("--override", action="append", default=[], help="dotted.key=value config override")
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--stride", type=int, default=5, help="take every stride-th calib spec, to cover all layouts")
    ap.add_argument("--poses", type=int, default=30, help="robot poses per episode")
    ap.add_argument("--isaac-port", type=int, default=5555)
    ap.add_argument("--detector-port", type=int, default=5600)
    ap.add_argument("--gpu", type=int, default=None, help="oracle channel: CUDA device of the ray caster")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    k, n = (int(x) for x in a.shard.split("/"))
    if a.gpu is not None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(a.gpu))

    from bamos.config import load_config
    from bamos.environment.analytic_renderer import set_cast_backend
    from bamos.environment.open_set import _world_centroid
    from bamos.experiment.benchmark import build_env_for_spec, make_episode_specs

    base = {"environment": {"backend": "isaac" if a.channel == "open_set" else "analytic",
                            "isaac": {"render_cache_dir": "results/cache/isaac"}},
            "sensor_detector": {"mode": "open_set" if a.channel == "open_set" else "rendered_threshold"}}
    cfg = load_config(a.config, [base, parse_overrides(a.override)])
    isaac = det = None
    if a.channel == "open_set":
        from bamos.environment.isaac_client import IsaacClient
        from bamos.environment.open_set import DetectorClient

        isaac = IsaacClient(cfg["environment"]["isaac"]["host"], a.isaac_port)
        det = DetectorClient(cfg["environment"]["isaac"]["host"], a.detector_port,
                             cache_dir=cfg["sensor_detector"]["open_set_cache_dir"])
    else:
        set_cast_backend("cuda:0")
    specs = make_episode_specs(cfg, "calib")[:: a.stride][: a.episodes][k::n]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "channel.json").write_text(json.dumps({"channel": a.channel, "overrides": a.override,
                                                   "detector": getattr(det, "model", "oracle")}, indent=1))
    rows = []
    rng_sensor = np.random.default_rng(0)
    for spec in specs:
        try:
            env, _ = build_env_for_spec(spec, cfg, None, isaac, det)
        except RuntimeError as e:  # unsolvable placement, as in the test split
            print(f"skip {spec.episode_id}: {str(e)[:60]}", flush=True)
            continue
        env.open_set_record_instances = True
        rng = np.random.default_rng([a.seed, spec.episode_seed])
        pool_ids = env.pool.pose_ids[env.pool.feasible]
        iso = np.array([q["iso_area_px"] for q in env.gt_quality_many(pool_ids)], dtype=float) / env.robot_camera.n_pixels
        obs = [env.observe_robot(int(pool_ids[i]), rng_sensor, 0) for i in stratified(iso, a.poses, rng)]
        gt_c = [round(float(x), 4) for x in env.ground_truth().target_center]
        st = int(cfg["mapping"]["occupancy_pixel_stride"])
        (out / "arrays").mkdir(exist_ok=True)
        np.savez_compressed(
            out / "arrays" / f"{spec.episode_id}.npz",
            depth=np.stack([np.nan_to_num(o.depth[::st, ::st], nan=np.inf, posinf=np.inf).astype(np.float32) for o in obs]),
            mask=np.stack([(o.target_mask[::st, ::st] if (o.detection.value == "DET" and o.target_mask is not None)
                            else np.zeros_like(o.depth[::st, ::st], dtype=bool)) for o in obs]),
            camera_id=np.array([o.camera_id for o in obs]), pose_id=np.array([-1 if o.pose_id is None else o.pose_id for o in obs]),
            stride=st)
        for o in obs:
            q = env.gt_quality_of_observation(o)
            nz = o.meta.get("noise_applied", {})
            cen = None
            if o.detection.value == "DET" and o.target_mask is not None and o.depth is not None:
                cen = _world_centroid(o.target_mask, o.depth, o.camera, o.world_from_camera)
            rows.append({
                "episode_id": spec.episode_id, "archetype": spec.archetype, "target_type": spec.target_type,
                "camera_type": o.camera_type, "camera_id": o.camera_id, "pose_id": -1 if o.pose_id is None else o.pose_id,
                "a_iso": q["iso_area_px"] / float(o.camera.n_pixels), "visible_fraction": q["visible_fraction"],
                "in_view": q["iso_area_px"] > 0, "oracle_det": q["gt_detected"], "status": o.detection.value,
                "fp": nz.get("fp", ""), "iou": nz.get("iou", ""), "n_masks": nz.get("n_masks", ""),
                "n_kept": nz.get("n_kept", ""), "instances": json.dumps(nz["instances"]) if "instances" in nz else "",
                "centroid": "" if cen is None else json.dumps([round(float(x), 4) for x in cen]),
                "gt_center": json.dumps(gt_c),
                "cam_pos": json.dumps([round(float(x), 4) for x in o.world_from_camera[:3, 3]]),
            })
        print(f"{spec.episode_id}: {len(obs)} frames (total {len(rows)})", flush=True)
        with open(out / f"frames_{k}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
    print(f"wrote {out / f'frames_{k}.csv'} ({len(rows)} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
