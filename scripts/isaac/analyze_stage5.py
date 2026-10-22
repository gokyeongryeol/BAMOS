"""Stage 6 of the 2x2 re-run (results doc R): tables, the missed-detection sweep, breakdowns by room and target size,
and the numbers the registered predictions R-* are judged on. Reads the stage 5 runs of scripts/isaac/stage5_test.sh.

    python3 scripts/isaac/analyze_stage5.py [--root results/raw/test] [--meta results/raw/episode_meta_test.csv]

Every comparison is paired over the episodes all arms completed; intervals are paired bootstrap 95% (10000 draws).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis as W  # noqa: E402

ORACLE = [("a_bamos", "BAMOS"), ("a_d5_novoxel", "target belief only (D5)"), ("a_scene", "scene model only"),
          ("a_reactive", "reactive (neither)"), ("a_d2_nonodet", "no NO_DET update (D2)"), ("a_d3_nok", "kappa = 1 (D3)")]
SAM3 = [("bp_bamos", "BAMOS"), ("bp_d5_novoxel", "target belief only (D5)"),
        ("bp_scene", "scene model only"), ("bp_reactive", "reactive (neither)"), ("bp_d2_nonodet", "no NO_DET update (D2)")]


def arr(data, tag, eps, key):
    if key == "auc":
        return np.array([data[tag][e]["anytime"].mean() for e in eps])
    if key == "none":
        return np.array([not data[tag][e]["declared"] for e in eps], dtype=float)
    return np.array([data[tag][e][key] for e in eps], dtype=float)


def diff(data, ref, tag, eps, key="success"):
    a, b = arr(data, ref, eps, key), arr(data, tag, eps, key)
    lo, hi = W.boot_diff(a, b)
    return b.mean() - a.mean(), lo, hi


def fmt(d):
    return f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/raw/test")
    ap.add_argument("--meta", default="results/raw/episode_meta_test.csv")
    a = ap.parse_args()
    root = Path(a.root)
    tags = sorted({t for t, _ in ORACLE + SAM3})
    data = {t: W.load_arm(root, t) for t in tags}
    missing = [t for t, v in data.items() if v is None]
    if missing:
        print("missing arms:", missing)
    data = {t: v for t, v in data.items() if v is not None}
    eps = sorted(set.intersection(*[set(v) for v in data.values()]))
    meta = {r["episode_id"]: r for r in csv.DictReader(open(a.meta))}
    size = {e: round(100 * float(meta[e]["target_size_scale_m"])) for e in eps}
    large = {e: meta[e]["archetype"] == "large_warehouse" for e in eps}
    print(f"arms {len(data)}; paired episodes {len(eps)} (large_warehouse {sum(large.values())})")

    W.table("A. oracle (analytic): the 2x2 and the ablations", ORACLE, data, eps)
    W.table("B. SAM 3 (threshold 0.5)", SAM3, data, eps)
    print("\nSAM 3 arms against SAM 3 BAMOS (success@B, AUC):")
    for t, n in SAM3[1:]:
        print(f"  {n:<26} {fmt(diff(data, 'bp_bamos', t, eps))}   AUC {fmt(diff(data, 'bp_bamos', t, eps, 'auc'))}")
    print("oracle arms against oracle BAMOS, AUC:")
    for t, n in ORACLE[1:]:
        print(f"  {n:<26} {fmt(diff(data, 'a_bamos', t, eps, 'auc'))}")

    print("\n## D. by room and target size: failure rate (1 - success@B)")
    groups = [("small rooms", lambda e: not large[e]), ("large_warehouse", lambda e: large[e])] + \
             [(f"{s} cm", (lambda s: lambda e: size[e] == s)(s)) for s in (10, 20, 30)]
    print(f"{'arm':<28}" + "".join(f"{g:>17}" for g, _ in groups))
    for t, n in [("a_bamos", "oracle BAMOS"), ("a_reactive", "oracle reactive"), ("a_scene", "oracle scene only"),
                 ("bp_bamos", "SAM 3 BAMOS"), ("bp_reactive", "SAM 3 reactive"), ("bp_scene", "SAM 3 scene only")]:
        cells = []
        for _, f in groups:
            sel = [e for e in eps if f(e)]
            cells.append(f"{1 - np.mean([data[t][e]['success'] for e in sel]):.3f} ({len(sel)})")
        print(f"{n:<28}" + "".join(f"{c:>17}" for c in cells))
    for ref, tag, n in [("bp_bamos", "bp_reactive", "SAM 3 reactive - BAMOS"), ("bp_bamos", "bp_scene", "SAM 3 scene - BAMOS"),
                        ("a_bamos", "a_reactive", "oracle reactive - BAMOS")]:
        for g, f in groups[:2]:
            sel = [e for e in eps if f(e)]
            print(f"  {n:<26} {g:<16} {fmt(diff(data, ref, tag, sel))}")

    print("\n## E. failures of BAMOS: outcome and target size")
    for t in ("a_bamos", "bp_bamos"):
        fails = [e for e in eps if not data[t][e]["success"]]
        nd = [e for e in fails if not data[t][e]["declared"]]
        print(f"{t}: failures {len(fails)}, no declaration {len(nd)} (10 cm {sum(size[e] == 10 for e in nd)}), "
              f"wrong declaration {len(fails) - len(nd)}; failure sizes {sorted(size[e] for e in fails)}")
        errs = sorted(round(float(data[t][e]['row'].get('final_gt_center_error_m') or 'nan'), 3) for e in fails if data[t][e]["declared"])
        print(f"   wrong-declaration errors at the end (m): {errs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
