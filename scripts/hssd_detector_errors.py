"""What an open-set detector's masks lay on, in the episodes a val run declared wrongly (or all declared ones).

    uv run python scripts/hssd_detector_errors.py --run results/hssd/val/falcon_bamos --port 5940 --detector-port 5600 \
        --fit results/hssd/calib/falcon --out results/hssd/analysis/falcon_bamos_wrong.jsonl [--limit 150] [--all]

For each episode (wrong declarations: declared, not a success), every DET frame of its step log is rendered again and its
mask read back from the detection cache (the same frame key, so the same mask the run saw). Per frame: the mask's
normalized area, the share of its pixels on goal instances, the instance holding most of its pixels and that object's
category (the scene's objects.csv category; "stage" for the walls and floors). Per episode: the frames, the declared
step, and whether the declaring frames' masks lay on one non-goal instance throughout.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default="configs/hssd.yaml")
    ap.add_argument("--fit", required=True)
    ap.add_argument("--port", type=int, default=5940)
    ap.add_argument("--detector-port", type=int, default=5600)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--all", action="store_true", help="every declared episode, not only the wrong ones")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import yaml

    from bamos.config import load_config
    from bamos.environment.habitat_env import HabitatEnvironment, TeleportClient
    from bamos.environment.open_set import DetectorClient
    from bamos.planning.runner import camera_from_cfg
    from run_hssd import parse_value

    fit = Path(a.fit)
    overrides = [yaml.safe_load((fit / "fitted.yaml").read_text())]
    for o in json.loads((fit / "overrides.json").read_text()):
        k, v = o.split("=", 1)
        d: dict = {}
        cur = d
        for p in k.split(".")[:-1]:
            cur = cur.setdefault(p, {})
        cur[k.split(".")[-1]] = parse_value(v)
        overrides.append(d)
    cfg = load_config(a.config, overrides)
    cam = camera_from_cfg(cfg["cameras"]["robot"])
    client = TeleportClient(cfg["hssd"]["host"], a.port, timeout=7200.0)
    det = DetectorClient(cfg["hssd"]["host"], a.detector_port, cache_dir=cfg["sensor_detector"].get("open_set_cache_dir"))
    run = Path(a.run)
    rows = [json.loads(l) for l in open(run / "episodes.jsonl")]
    pick = [r for r in rows if r["status"] == "ok" and r["declared_stop"] and (a.all or not r["success"])]
    pick.sort(key=lambda r: (r["scene"], int(r["episode_id"])))
    if a.limit:
        pick = pick[:: max(1, len(pick) // a.limit)][: a.limit]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {(json.loads(l)["scene"], json.loads(l)["episode_id"]) for l in open(out)} if out.exists() else set()
    for r in pick:
        if (r["scene"], r["episode_id"]) in done:
            continue
        env = HabitatEnvironment(client, cfg, r["scene"], r["episode_id"], cam)
        env.open_set_client = det
        cats = {int(i): str(c) for i, c in env.header.get("objects", [])}
        gids = set(int(g) for g in env.goal_ids)
        steps = [json.loads(l) for l in open(run / "steps" / f"{r['scene']}_{r['episode_id']}.jsonl")]
        frames = []
        for k, s in enumerate(steps):
            if s.get("detection_status") != "DET":
                continue
            obs = env.observe_robot(int(s["selected_pose_id"]), None, k)
            m = np.asarray(obs.target_mask, dtype=bool) if obs.target_mask is not None else None
            if m is None or not m.any():
                frames.append({"step": k, "mask": False})
                continue
            ids = np.asarray(env.last_ids)[m]
            cnt = Counter(ids.tolist())
            top, n_top = cnt.most_common(1)[0]
            frames.append({"step": k, "mask": True, "area": float(m.mean()), "on_goal": float(np.isin(ids, list(gids)).mean()),
                           "top_id": int(top), "top_share": n_top / len(ids), "top_is_goal": int(top) in gids,
                           "top_cat": "goal" if int(top) in gids else cats.get(int(top), "stage" if int(top) == 0 else f"id{int(top)}")})
        decl = next((k for k, s in enumerate(steps) if s.get("declared_stop")), None)
        last = [f for f in frames if f["mask"] and (decl is None or f["step"] <= decl)][-3:]
        row = {"scene": r["scene"], "episode_id": r["episode_id"], "category": r["category"], "success": r["success"],
               "declared_step": decl, "n_det": len(frames), "n_det_on_goal": sum(1 for f in frames if f.get("top_is_goal")),
               "declaring_top_cats": [f["top_cat"] for f in last], "declaring_same_instance": len({f["top_id"] for f in last}) == 1,
               "frames": frames}
        with open(out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"[errors] {r['scene']} {r['episode_id']} {r['category']}: {len(frames)} DET, on goal {row['n_det_on_goal']}, "
              f"declaring masks on {row['declaring_top_cats']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
