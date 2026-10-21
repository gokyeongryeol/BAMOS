"""Stage 8: the tables of the extra test arms and the check of each prediction
written down before the runs. Same success definition and paired bootstrap as scripts/analyze_workshop.py.

    uv run python scripts/analyze_stage8.py [--root results/raw/test]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import analyze_workshop as aw

S2X2 = [("bamos", "BAMOS"), ("d5_novoxel", "no scene model (D5)"), ("scene", "scene model only"),
        ("reactive", "reactive (neither)")]
SYM = [("scene_bstop", "scene coverage + belief stop"), ("cov_bstop", "coverage + belief stop"),
       ("ig_fdet", "IG views + first-DET stop")]
TABLES = [
    ("E1. oracle: the view rule and the stop rule crossed", [("a_" + t, n) for t, n in S2X2[:1] + SYM + S2X2[2:]]),
    ("E1. SAM 3: the view rule and the stop rule crossed", [("bp_" + t, n) for t, n in S2X2[:1] + SYM + S2X2[2:]]),
    ("E2. Falcon Perception: the 2x2 and D2",
     [("fa_" + t, n) for t, n in S2X2] + [("fa_d2_nonodet", "no NO_DET update (D2)")]),
    ("X. multi-mask DET update (exploratory, not pre-registered)",
     [("bp_bamos", "SAM 3 BAMOS"), ("bp_bamos_multimask", "SAM 3 multi-mask"), ("fa_bamos", "Falcon BAMOS"),
      ("fa_bamos_multimask", "Falcon multi-mask")]),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/raw/test")
    ap.add_argument("--frozen", default="configs/conditions/frozen.json")
    a = ap.parse_args()
    frozen = json.loads(Path(a.frozen).read_text())
    aw.CONFIRM = int(frozen.get("confirmations", 1))
    tags = list(dict.fromkeys(t for _, arms in TABLES for t, _ in arms))
    data = {t: v for t in tags if (v := aw.load_arm(Path(a.root), t)) is not None}
    missing = [t for t in tags if t not in data]
    eps = sorted(set.intersection(*[set(v) for v in data.values()]))
    print(f"arms loaded: {len(data)}, missing: {missing or 'none'}; episodes common to all: {len(eps)}")
    for title, arms in TABLES:
        aw.table(title, arms, data, eps)

    s = {t: np.array([data[t][e]["success"] for e in eps], dtype=float) for t in data}
    fd = {t: np.mean([data[t][e]["false_decl"] for e in eps]) for t in data}

    def diff(hi: str, lo: str) -> tuple[float, float, float]:
        d0, d1 = aw.boot_diff(s[lo], s[hi])
        return s[hi].mean() - s[lo].mean(), d0, d1

    def fmt(d: tuple[float, float, float]) -> str:
        return f"{d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}]"

    checks = []
    d = diff("bp_bamos", "bp_scene_bstop")
    checks.append(("S-V", f"SAM 3 BAMOS - scene coverage + belief stop {fmt(d)}", d[1] > 0))
    d = diff("bp_scene_bstop", "bp_scene")
    checks.append(("S-S", f"SAM 3 scene coverage + belief stop - scene model only {fmt(d)}", d[0] > 0))
    ok = fd["a_ig_fdet"] > fd["a_bamos"] and fd["bp_ig_fdet"] > fd["bp_bamos"]
    checks.append(("S-I", f"false decl IG + first DET vs BAMOS: oracle {fd['a_ig_fdet']:.3f} vs {fd['a_bamos']:.3f}, "
                          f"SAM 3 {fd['bp_ig_fdet']:.3f} vs {fd['bp_bamos']:.3f}", ok))
    d = diff("bp_cov_bstop", "bp_reactive")
    checks.append(("S-C", f"SAM 3 coverage + belief stop - reactive {fmt(d)}", d[1] > 0))
    d = diff("fa_bamos", "fa_reactive")
    checks.append(("D-R", f"Falcon BAMOS - reactive {fmt(d)}", d[1] > 0))
    m = [s[f"fa_{t}"].mean() for t in ("bamos", "d5_novoxel", "scene")]
    checks.append(("D-O", "Falcon BAMOS >= no scene model >= scene model only: " + " / ".join(f"{x:.3f}" for x in m),
                   m[0] >= m[1] >= m[2]))
    d = diff("fa_bamos", "fa_d2_nonodet")
    checks.append(("D-2", f"Falcon BAMOS - D2 {fmt(d)}", d[1] > 0))
    # the false-positive predictions F-1..F-3 are read off stage 13's grid (analyze_stage13.py)
    print("\n## R.5 predictions")
    for k, txt, ok in checks:
        print(f"{k:<5}{'confirmed' if ok else 'REFUTED':<11}{txt}")
    print(f"{sum(ok for *_, ok in checks)} of {len(checks)} confirmed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
