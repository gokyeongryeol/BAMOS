"""Stages 9 and 10: the perfect scene model (true geometry in place of the
occupancy belief) and the Falcon arms SAM 3 had, with the check of each prediction written down before the runs.
Same success definition and paired bootstrap as scripts/analyze_workshop.py; an arm not run yet is left out.

    uv run python scripts/analyze_stage9_10.py [--root results/raw/test]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import analyze_workshop as aw

GT = [("bamos", "BAMOS"), ("gt_bamos", "BAMOS, true geometry"), ("scene", "scene model only"),
      ("scene_bstop", "scene coverage + belief stop")]
FA = [("fa_bamos", "BAMOS"), ("fa_scene_bstop", "scene coverage + belief stop"), ("fa_cov_bstop", "coverage + belief stop"),
      ("fa_ig_fdet", "IG views + first-DET stop"), ("fa_scene", "scene model only"), ("fa_reactive", "reactive"),
      ("fa_gt_bamos", "BAMOS, true geometry")]
TABLES = [("S9. oracle: true geometry", [("a_" + t, n) for t, n in GT]),
          ("S9. SAM 3: true geometry", [("bp_" + t, n) for t, n in GT]),
          ("S10. Falcon Perception: the missing arms", FA)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/raw/test")
    ap.add_argument("--frozen", default="configs/conditions/frozen.json")
    a = ap.parse_args()
    aw.CONFIRM = int(json.loads(Path(a.frozen).read_text()).get("confirmations", 1))
    tags = list(dict.fromkeys(t for _, arms in TABLES for t, _ in arms))
    data = {t: v for t in tags if (v := aw.load_arm(Path(a.root), t)) is not None}
    full = max(len(v) for v in data.values())
    for t in [t for t, v in data.items() if len(v) < full]:
        print(f"note: {t} has {len(data[t])}/{full} episodes and is left out")
        del data[t]
    eps = sorted(set.intersection(*[set(v) for v in data.values()]))
    print(f"arms loaded: {len(data)}, missing: {[t for t in tags if t not in data] or 'none'}; common episodes: {len(eps)}")
    for title, arms in TABLES:
        aw.table(title, arms, data, eps)

    s = {t: np.array([data[t][e]["success"] for e in eps], dtype=float) for t in data}
    fd = {t: np.mean([data[t][e]["false_decl"] for e in eps]) for t in data}

    def diff(hi: str, lo: str) -> tuple[float, float, float]:
        d0, d1 = aw.boot_diff(s[lo], s[hi])
        return s[hi].mean() - s[lo].mean(), d0, d1

    def fmt(d: tuple[float, float, float]) -> str:
        return f"{d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}]"

    def have(*ts: str) -> bool:
        return all(t in s for t in ts)

    checks = []
    if have("a_bamos", "a_gt_bamos", "bp_bamos", "bp_gt_bamos"):
        d1, d2 = diff("a_gt_bamos", "a_bamos"), diff("bp_gt_bamos", "bp_bamos")
        checks.append(("W-3", f"BAMOS true geometry - BAMOS: oracle {fmt(d1)}, SAM 3 {fmt(d2)}",
                       abs(d1[0]) <= 0.02 and abs(d2[0]) <= 0.02))
    if have("fa_bamos", "fa_scene_bstop"):
        d = diff("fa_bamos", "fa_scene_bstop")
        checks.append(("FV-1", f"Falcon BAMOS - scene coverage + belief stop {fmt(d)}", d[1] > 0))
    if have("fa_scene_bstop", "fa_scene"):
        d = diff("fa_scene_bstop", "fa_scene")
        checks.append(("FV-2", f"Falcon scene coverage + belief stop - scene model only {fmt(d)}", d[1] > 0))
    if have("fa_ig_fdet", "fa_bamos"):
        checks.append(("FV-3", f"Falcon false decl IG + first DET {fd['fa_ig_fdet']:.3f} vs BAMOS {fd['fa_bamos']:.3f}",
                       fd["fa_ig_fdet"] > fd["fa_bamos"]))
    print("\n## R.7 / R.8 predictions")
    for k, txt, ok in checks:
        print(f"{k:<6}{'confirmed' if ok else 'REFUTED':<11}{txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
