"""Stage 13: BAMOS's robustness to errors injected into the oracle detector, on
the levels of stages 5, 8 and 13 (constants fitted per level), paired with the clean oracle run, and the check of
predictions E-2, E-5 and E-6. Same success definition and bootstrap as scripts/analyze_workshop.py.

    uv run python scripts/analyze_stage13.py [--root results/raw/test]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import analyze_workshop as aw

FN = [(0.0, "a_bamos"), (0.1, "s13_fn010_bamos"), (0.2, "s13_fn020_bamos"), (0.3, "s13_fn030_bamos")]
FP = [(0.0, "a_bamos"), (0.1, "s8_fp010_bamos"), (0.2, "s13_fp020_bamos"), (0.3, "s13_fp030_bamos")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/raw/test")
    ap.add_argument("--frozen", default="configs/conditions/frozen.json")
    a = ap.parse_args()
    aw.CONFIRM = int(json.loads(Path(a.frozen).read_text()).get("confirmations", 1))
    root = Path(a.root)
    data = {t: aw.load_arm(root, t) for _, t in FN + FP}
    eps = sorted(set.intersection(*[set(v) for v in data.values()]))
    ref = np.array([data["a_bamos"][e]["success"] for e in eps], dtype=float)
    print(f"{len(eps)} paired episodes; BAMOS on the oracle with injected errors, constants fitted per level")
    res = {}
    for kind, levels in (("missed detections", FN), ("false masks", FP)):
        print(f"\n## {kind}")
        print(f"{'rate':>6}{'success@B [95% CI]':>24}{'loss [95% CI]':>24}{'AUC':>7}{'wrong':>7}{'undecl':>8}{'views':>6}")
        for rate, t in levels:
            d = [data[t][e] for e in eps]
            s = np.array([x["success"] for x in d], dtype=float)
            lo, hi = aw.boot(s)
            dlo, dhi = aw.boot_diff(s, ref)
            views = np.median([x["fire_step"] for x in d if x["declared"]])
            res[(kind, rate)] = s.mean()
            print(f"{rate:>6}{s.mean():>8.3f} [{lo:.3f},{hi:.3f}]{ref.mean() - s.mean():>+10.3f} [{dlo:+.3f},{dhi:+.3f}]"
                  f"{np.mean([x['anytime'].mean() for x in d]):>7.3f}{np.mean([x['false_decl'] for x in d]):>7.3f}"
                  f"{np.mean([not x['declared'] for x in d]):>8.3f}{views:>6.0f}")
    fn = [res[("missed detections", r)] for r, _ in FN]
    fp = [res[("false masks", r)] for r, _ in FP]
    print("\n## R.13 predictions")
    ok = all(b <= a_ for a_, b in zip(fn, fn[1:]))
    print(f"E-2 {'confirmed' if ok else 'REFUTED':<11}BAMOS over missed detections " + " / ".join(f"{x:.3f}" for x in fn))
    lfn = {r: fn[0] - res[("missed detections", r)] for r in (0.2, 0.3)}
    lfp = {r: fp[0] - res[("false masks", r)] for r in (0.2, 0.3)}
    ok = all(lfp[r] > lfn[r] for r in (0.2, 0.3))
    print(f"E-5 {'confirmed' if ok else 'REFUTED':<11}loss at 0.2 / 0.3: false masks {lfp[0.2]:+.3f} / {lfp[0.3]:+.3f}, "
          f"missed detections {lfn[0.2]:+.3f} / {lfn[0.3]:+.3f}")
    ok = all(b <= a_ for a_, b in zip(fp, fp[1:])) and fp[-1] >= 0.9
    print(f"E-6 {'confirmed' if ok else 'REFUTED':<11}BAMOS over false masks " + " / ".join(f"{x:.3f}" for x in fp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
