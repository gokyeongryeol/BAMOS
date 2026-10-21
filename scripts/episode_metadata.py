"""Build every episode spec of a split and record each solvable episode's features (no planner, no result).

    uv run python scripts/episode_metadata.py --split calib [--out results/raw/episode_meta_calib.csv]

The features (environment/episodes.py `episode_metadata`) are evaluator-only: target type, shape and size, placement,
anchor, occluder, distances (start to target, target to the nearest wall, target to the nearest candidate cell),
the start view's visibility, how many feasible poses detect the target or meet the quality bar, and the layout's
object counts. Unsolvable specs are listed with the reason.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bamos.config import load_config  # noqa: E402
from bamos.environment.analytic_renderer import set_cast_backend  # noqa: E402
from bamos.experiment.benchmark import build_env_for_spec, make_episode_specs  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--config", default="configs/normal.yaml")
    ap.add_argument("--out", default=None)
    ap.add_argument("--stride", type=int, default=1, help="every stride-th spec, as run_experiment --stride (calib: 5)")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    cfg = load_config(a.config, [{"environment": {"backend": "analytic"}}])
    set_cast_backend(cfg["experiment"]["device"])
    rows, bad = [], []
    specs = make_episode_specs(cfg, a.split)[:: a.stride][: a.limit]
    for s in specs:
        try:
            _, m = build_env_for_spec(s, cfg, None, None, None)
        except RuntimeError as e:
            bad.append((s.episode_id, str(e)[:80]))
            continue
        meta = m["evaluator_only"]["episode_meta"]
        row = {"episode_id": s.episode_id, "layout_id": s.layout_id, "archetype": s.archetype}
        for k, v in meta.items():
            row[k] = json.dumps(v) if isinstance(v, (list, dict)) else v
        rows.append(row)
    out = Path(a.out or f"results/raw/episode_meta_{a.split}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{a.split}: {len(specs)} specs, solvable {len(rows)}, unsolvable {len(bad)}; wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
