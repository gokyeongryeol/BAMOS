"""Train-split statistics that set the HSSD world constants (hssd.* in configs/hssd.yaml), on the server's scene data.

    uv run python scripts/hssd/scene_stats.py --port 5810 --out results/hssd/scene_stats/goals_0.jsonl
    uv run python scripts/hssd/scene_stats.py --summarize results/hssd/scene_stats

One episode per (scene, category) of the scenes the server holds (the goal set is the category's instances in the scene;
the pose pool is that episode's start island). One JSON row per goal instance:
- size: the box size scale sqrt(surface area / 4) (formulation §3), for size_grid_by_category {p10, p50, p90};
- d_xy: the horizontal distance from the box centre to the nearest pose position whose floor is below the centre and
  within 2.5 m of it, and h: the centre's height above that position's floor (candidate_dilate_m, candidate_band_m);
- the pose positions' navmesh height above the floor surface under them (the floor tolerance's reference).
"""
from __future__ import annotations

import argparse
import os
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def size_scale(lo, hi) -> float:
    a, b, c = np.asarray(hi, dtype=float) - np.asarray(lo, dtype=float)
    return float(np.sqrt((a * b + b * c + c * a) / 2.0))


def collect(a):
    from bamos.config import load_config
    from bamos.environment.hssd_env import HssdClient

    cfg = load_config(a.config)
    hc = cfg["hssd"]
    client = HssdClient(hc["host"], a.port)
    by: dict = {}
    for e in client.episodes():
        by.setdefault((e["scene"], e["category"]), []).append(e)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {(json.loads(l)["scene"], json.loads(l)["category"]) for l in open(out)} if out.exists() else set()
    for (scene, cat), eps in sorted(by.items()):
        if (scene, cat) in done:
            continue
        e = sorted(eps, key=lambda e: int(e["episode_id"]))[0]
        h = client.load(scene, e["episode_id"])
        if os.path.basename(str(h.get("scene_data", ""))) != hc["scene_data"]:
            raise SystemExit(f"server scene data {h.get('scene_data')} != {hc['scene_data']}")
        _, la = client.lattice(float(hc["pose_lattice_m"]), h["start_position"])
        P = np.asarray(la["positions"], dtype=np.float64).reshape(-1, 3)
        fy = np.asarray(la["floor_y"], dtype=np.float64)
        rows = []
        for g in h["goals"]:
            lo, hi = np.asarray(g["aabb_lo"], float), np.asarray(g["aabb_hi"], float)  # Habitat frame, y up
            c = 0.5 * (lo + hi)
            ok = np.isfinite(fy) & (fy <= c[1]) & (c[1] - fy <= 2.5)
            if ok.any():
                d = np.hypot(P[ok, 0] - c[0], P[ok, 2] - c[2])
                j = int(np.argmin(d))
                d_xy, hgt = float(d[j]), float(c[1] - fy[ok][j])
            else:
                d_xy, hgt = None, None
            rows.append({"scene": scene, "category": cat, "episode_id": e["episode_id"], "semantic_id": g["semantic_id"],
                         "size": size_scale(lo, hi), "extent": (hi - lo).round(3).tolist(), "d_xy": d_xy, "h": hgt})
        float_above = (P[:, 1] - fy)[np.isfinite(fy)]
        with open(out, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.write(json.dumps({"scene": scene, "category": cat, "navmesh_above_floor": np.round(float_above, 3).tolist()}) + "\n")
        print(f"[stats] {scene} {cat}: {len(rows)} goals, {len(P)} positions", flush=True)


def summarize(root: Path):
    G, F = [], []
    for p in sorted(root.glob("goals_*.jsonl")):
        for l in open(p):
            r = json.loads(l)
            (F if "navmesh_above_floor" in r else G).append(r)
    print(f"{len(G)} goal instances, {len({r['scene'] for r in G})} scenes")
    print("size_grid_by_category {p10, p50, p90} of the size scale:")
    for cat in sorted({r["category"] for r in G}):
        s = np.array([r["size"] for r in G if r["category"] == cat])
        print(f"  {cat:13s} n {len(s):4d}  {np.round(np.percentile(s, [10, 50, 90]), 2).tolist()}")
    Gh = [r for r in G if r["d_xy"] is not None]
    print(f"goal centres with a pose position below them within 2.5 m: {len(Gh)} of {len(G)}")
    d = np.array([r["d_xy"] for r in Gh])
    hh = np.array([r["h"] for r in Gh])
    print("horizontal distance to the nearest such pose position: quantiles 50/90/95/99",
          np.round(np.percentile(d, [50, 90, 95, 99]), 2).tolist())
    for D in (1.0, 1.5, 2.0):
        print(f"  share within {D} m: {np.mean(d <= D):.3f}")
    print("centre height above that position's floor: quantiles 0.5/1/50/99/99.5",
          np.round(np.percentile(hh, [0.5, 1, 50, 99, 99.5]), 2).tolist())
    for b0, b1 in ((0.0, 2.3), (0.15, 1.85)):
        out = (hh < b0) | (hh > b1)
        print(f"  outside [{b0}, {b1}]: {out.mean():.4f} of all;", {c: round(float(np.mean(out[[r['category'] == c for r in Gh]])), 3)
                                                              for c in sorted({r['category'] for r in Gh})})
    fa = np.concatenate([np.asarray(r["navmesh_above_floor"]) for r in F])
    print("navmesh above the floor surface under it: quantiles 1/5/50/95/99", np.round(np.percentile(fa, [1, 5, 50, 95, 99]), 3).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/hssd.yaml")
    ap.add_argument("--port", type=int, default=5810)
    ap.add_argument("--out", default="")
    ap.add_argument("--summarize", default="")
    a = ap.parse_args()
    if a.summarize:
        summarize(Path(a.summarize))
    else:
        collect(a)


if __name__ == "__main__":
    main()
