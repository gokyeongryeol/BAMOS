"""Why did an HSSD val episode fail? Replays the episode's trajectory (its step log) against the pose-candidate rules.

    uv run python scripts/hssd/analyze_failures.py --run results/hssd/val_c14bfcf --port 5820 --shard 0/8 \
        --out results/hssd/val_c14bfcf/failures.jsonl

For each failed episode (status ok, success false):
- D: the pose positions from which some yaw / pitch gives an oracle detection of a goal instance. Poses are rendered
  when their view frustum holds a goal box's centre or corner within the depth range, one detection per position ends
  that position's search.
- geodesic from the start to the nearest position of D, and from each visited position.
- the trajectory replayed with the runner's rule (navmesh geodesic <= max_geodesic_m, floor seen in an earlier frame;
  the floor map rebuilt from re-rendered frames): the first step a D position was within the geodesic range, and the
  first step one was also floor-seen (a feasible candidate the planner could have picked).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/hssd.yaml")
    ap.add_argument("--run", required=True)
    ap.add_argument("--port", type=int, default=5820)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-renders", type=int, default=6000)
    a = ap.parse_args()
    import numpy as np

    from bamos.config import load_config
    from bamos.environment.hssd_env import HabitatEnvironment, TeleportClient
    from bamos.mapping.camera import inverse_transform
    from bamos.planning.hssd_runner import FloorSeenMap
    from bamos.planning.runner import camera_from_cfg

    cfg = load_config(a.config)
    hc = cfg["hssd"]
    client = TeleportClient(hc["host"], a.port)
    cam = camera_from_cfg(cfg["cameras"]["robot"])
    run = Path(a.run)
    rows = [json.loads(l) for l in open(run / "episodes.jsonl")]
    fails = sorted([r for r in rows if r["status"] == "ok" and not r["success"]], key=lambda r: (r["scene"], int(r["episode_id"])))
    i, n = (int(x) for x in a.shard.split("/"))
    fails = fails[i::n]
    out = Path(a.out)
    done = {(json.loads(l)["scene"], json.loads(l)["episode_id"]) for l in open(out)} if out.exists() else set()
    R = float(hc["max_geodesic_m"])
    for r in fails:
        if (r["scene"], r["episode_id"]) in done:
            continue
        t0 = time.time()
        env = HabitatEnvironment(client, cfg, r["scene"], r["episode_id"], cam)
        gi, T = env.pool.grid_index, env.pool.world_from_camera
        b = env.base_world
        steps = [json.loads(l) for l in open(run / "steps" / f"{r['scene']}_{r['episode_id']}.jsonl")]
        traj = [int(s["selected_pose_id"]) for s in steps]
        assert np.allclose(T[traj[0]], np.asarray(steps[0]["camera_pose"]), atol=1e-4), "pool differs from the run's"
        # ---- D: positions with a detecting pose
        goals = env.ground_truth().goals
        pts = np.concatenate([np.stack([g["center"]] + [np.array([x, y, z]) for x in (g["lo"][0], g["hi"][0])
                                                         for y in (g["lo"][1], g["hi"][1]) for z in (g["lo"][2], g["hi"][2])])
                              for g in goals])
        Ti = np.stack([inverse_transform(t) for t in T])
        pc = np.einsum("pij,nj->pni", Ti[:, :3, :3], pts) + Ti[:, None, :3, 3]
        z = pc[..., 2]
        u = cam.fx * pc[..., 0] / np.maximum(z, 1e-6) + cam.cx
        v = cam.fy * pc[..., 1] / np.maximum(z, 1e-6) + cam.cy
        inview = ((z >= cam.near) & (z <= cam.far) & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)).any(axis=1)
        cand = np.nonzero(inview)[0]
        cand = cand[np.argsort(np.linalg.norm(pc[cand], axis=2).min(axis=1))]
        D, n_r = set(), 0
        for pid in cand:
            p = int(gi[pid, 0])
            if p in D:
                continue
            if n_r >= a.max_renders:
                break
            n_r += 1
            if env.gt_quality_of_observation(env.observe_robot(int(pid), None, 0))["gt_detected"]:
                D.add(p)
        D = np.array(sorted(D), dtype=np.int64)
        # ---- trajectory replay
        fs = FloorSeenMap(hc["floor_seen_cell_m"], hc["floor_seen_bin_m"])
        first_in_range = first_feasible = None
        n_steps_range_blocked = 0
        visited_pos = []
        min_geo_visit = np.inf
        for t, pid in enumerate(traj):
            obs = env.observe_robot(pid, None, 0)
            fs.add(obs)
            cur = int(gi[pid, 0])
            visited_pos.append(cur)
            if not len(D):
                continue
            g = env.geodesic(cur, D)
            min_geo_visit = min(min_geo_visit, float(g.min()))
            inr = D[g <= R]
            if len(inr):
                first_in_range = t if first_in_range is None else first_in_range
                tol = np.where(env.sloped[inr], float(hc["floor_tol_sloped_m"]), float(hc["floor_tol_m"]))
                seen = fs.seen(b[inr, :2], env.floor_z[inr], tol) | (inr == 0) | (inr == cur)
                if seen.any():
                    first_feasible = t if first_feasible is None else first_feasible
                else:
                    n_steps_range_blocked += 1
        geo_start = float(env.geodesic(0, D).min()) if len(D) else None
        # straight-line distance from the start to the nearest goal box, and the trajectory's spread
        vp = b[np.array(visited_pos)]
        row = {"scene": r["scene"], "episode_id": r["episode_id"], "category": r["category"],
               "detected_ever": r["first_true_detection_step"] is not None, "n_views": len(traj),
               "n_positions": int(len(b)), "n_D_positions": int(len(D)), "n_rendered": n_r,
               "n_inview_poses": int(inview.sum()), "D_truncated": bool(n_r >= a.max_renders),
               "geo_start_to_D": geo_start, "min_geo_visited_to_D": None if not np.isfinite(min_geo_visit) else min_geo_visit,
               "first_step_D_in_range": first_in_range, "first_step_D_feasible": first_feasible,
               "n_steps_D_in_range_floor_blocked": n_steps_range_blocked,
               "n_unique_positions_visited": len(set(visited_pos)),
               "visited_spread_m": float(np.linalg.norm(vp[:, :2] - vp[0, :2], axis=1).max()),
               "path_length_m": r["path_length_m"], "episode_geodesic_m": r["geodesic_distance_m"],
               "final_gt_bin_posterior_mass": r["final_gt_bin_posterior_mass"], "s": round(time.time() - t0, 1)}
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[failures] {r['scene']} {r['episode_id']}: D {len(D)} positions, geo start->D {geo_start}, "
              f"in range at {first_in_range}, feasible at {first_feasible} ({row['s']} s)", flush=True)


if __name__ == "__main__":
    main()
