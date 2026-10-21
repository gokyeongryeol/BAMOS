"""SR and SPL (bootstrap 95% CIs over episodes) of HSSD val runs (run_hssd.py outputs).

    uv run python scripts/hssd_summary.py results/hssd/val/oracle_bamos results/hssd/val/oracle_scene_only ...

SR: the declared cell within evaluation.max_center_error_m of a goal box. Also the 9 val episodes whose start island
lies on top of a roof slab (a defect of the benchmark's episodes), apart.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOF_START = {("105515379_173104395", e) for e in ("9", "18", "19", "20", "21", "22", "23")} | \
    {("104348328_171513363", e) for e in ("14", "16")}


def ci(x, b=2000, seed=0):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (b, len(x)))].mean(axis=1)
    return np.percentile(m, [2.5, 97.5])


def main() -> int:
    print(f"{'run':32s} {'n':>5s} {'err':>4s}  {'SR [95% CI]':>22s}  {'SPL [95% CI]':>22s}  {'SR roof-start 9':>16s}")
    for d in map(Path, sys.argv[1:]):
        rows = [json.loads(l) for l in open(d / "episodes.jsonl")]
        ok = [r for r in rows if r["status"] == "ok"]
        if not ok:
            print(f"{d.name:32s} {len(rows):5d} (no ok rows)")
            continue
        s = [bool(r["success"]) for r in ok]
        p = [float(r["spl"]) for r in ok]
        a, b = ci(s)
        c, e = ci(p)
        roof = [bool(r["success"]) for r in ok if (r["scene"], r["episode_id"]) in ROOF_START]
        print(f"{d.name:32s} {len(rows):5d} {len(rows) - len(ok):4d}  {100 * np.mean(s):5.1f} [{100 * a:5.1f}, {100 * b:5.1f}]  "
              f"{100 * np.mean(p):5.1f} [{100 * c:5.1f}, {100 * e:5.1f}]  {sum(roof):>8d} of {len(roof)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
