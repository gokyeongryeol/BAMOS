"""Which HSSD val episodes can the oracle detector solve at all from the teleport pose pool? (the Isaac benchmark's
episode solvability check, environment/episodes.py, for HSSD)

    uv run python scripts/hssd/solvability.py --port 5800 --out results/hssd/solvability.jsonl [--scenes ...]

An episode is solvable when some pose of its pool (the pose lattice on the start's navmesh island, every yaw and pitch)
gives an oracle detection of a goal instance (>= hssd.oracle_min_px pixels of valid depth). Episodes of one scene with
the same category and start island share the answer, so each such group is checked once. Poses are tried nearest-first
among those whose view frustum holds a corner or the centre of a goal box within the depth range; the first detection
ends the search. One JSON row per episode: solvable, the detecting pose, the number of poses rendered.
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
    ap.add_argument("--port", type=int, default=5800)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", default="")
    ap.add_argument("--max-renders", type=int, default=3000, help="per group; unsolved after this many: 'undecided'")
    a = ap.parse_args()
    import numpy as np

    from bamos.config import load_config
    from bamos.environment.hssd_env import HabitatEnvironment, TeleportClient
    from bamos.mapping.camera import inverse_transform
    from bamos.planning.runner import camera_from_cfg

    cfg = load_config(a.config)
    client = TeleportClient(cfg["hssd"]["host"], a.port)
    cam = camera_from_cfg(cfg["cameras"]["robot"])
    eps = sorted(client.episodes(), key=lambda e: (e["scene"], int(e["episode_id"])))
    if a.scenes:
        keep = set(a.scenes.split(","))
        eps = [e for e in eps if e["scene"] in keep]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {(json.loads(l)["scene"], json.loads(l)["episode_id"]) for l in open(out)} if out.exists() else set()
    groups: dict = {}
    for e in eps:
        if (e["scene"], e["episode_id"]) in done:
            continue
        t0 = time.time()
        env = HabitatEnvironment(client, cfg, e["scene"], e["episode_id"], cam)
        key = (e["scene"], e["category"], env.island)
        if key not in groups:
            goals = env.ground_truth().goals
            pts = np.concatenate([np.stack([g["center"]] + [np.array([x, y, z]) for x in (g["lo"][0], g["hi"][0])
                                                             for y in (g["lo"][1], g["hi"][1]) for z in (g["lo"][2], g["hi"][2])])
                                  for g in goals])  # [9 G, 3] centres and corners
            T = env.pool.world_from_camera
            Ti = np.stack([inverse_transform(t) for t in T])
            pc = np.einsum("pij,nj->pni", Ti[:, :3, :3], pts) + Ti[:, None, :3, 3]  # [P, 9G, 3]
            z = pc[..., 2]
            u = cam.fx * pc[..., 0] / np.maximum(z, 1e-6) + cam.cx
            v = cam.fy * pc[..., 1] / np.maximum(z, 1e-6) + cam.cy
            inview = ((z >= cam.near) & (z <= cam.far) & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)).any(axis=1)
            cand = np.nonzero(inview)[0]
            dist = np.linalg.norm(pc[cand], axis=2).min(axis=1)
            cand = cand[np.argsort(dist)][: a.max_renders]
            found, n = None, 0
            for pid in cand:
                n += 1
                q = env.gt_quality_of_observation(env.observe_robot(int(pid), None, 0))
                if q["gt_detected"]:
                    found = int(pid)
                    break
            status = "solvable" if found is not None else ("undecided" if len(np.nonzero(inview)[0]) > a.max_renders else "unsolvable")
            groups[key] = {"status": status, "pose_id": found, "n_rendered": n, "n_candidate_poses": int(inview.sum()),
                           "pose": None if found is None else T[found][:3, 3].round(3).tolist()}
        row = {"scene": e["scene"], "episode_id": e["episode_id"], "category": e["category"], "island": env.island,
               **groups[key], "s": round(time.time() - t0, 1)}
        with open(out, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[solvability] {e['scene']} {e['episode_id']} {e['category']}: {row['status']} (rendered {row['n_rendered']}, "
              f"{row['s']} s)", flush=True)


if __name__ == "__main__":
    main()
