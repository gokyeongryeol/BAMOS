"""Re-run stage 3 selection (README.md, "Isaac Sim pipeline"): choose SAM 3's score threshold t from the calib closed-loop
runs, never from the test split.

    uv run python scripts/isaac/select_threshold.py --root results/raw/stage3

The stop rule's confirmation count is N = 1, the rule of spec 10 itself. A larger N needs the target seen again at
the same place, which an information-seeking planner does not do once the belief is sure of it, so N > 1 would
measure how often the planner happens to look back rather than how reliable a declaration is (and would favour
U = P+, which keeps looking). The sweep over N is still printed, as the STOP-N analysis.

t is the threshold with the highest calib success@B of SAM 3 itself (read at the first declaration, as
scripts/isaac/analysis.py does); ties go to the lower no-declaration rate (the threshold that finds more targets,
its extra false declarations being reported, not hidden), then to the lower t, closer to SAM 3's own default.
Writes <root>/selection.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis as W  # noqa: E402

N = 1  # the stop rule's confirmation count (spec 10)
BASE = ["oracle"]
THRESH = {f"t{round(100 * t):03d}": t for t in (0.5, 0.6, 0.7, 0.8, 0.9)}  # as scripts/isaac/stage2_fit.sh / scripts/isaac/stage3_select.sh


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/raw/stage3")
    ap.add_argument("--n-max", type=int, default=6, help="largest N of the printed STOP-N sweep")
    a = ap.parse_args()
    root = Path(a.root)
    data = {t: W.load_arm(root, t) for t in BASE + [f"sam3_{k}" for k in THRESH]}
    missing = [t for t, v in data.items() if v is None]
    if missing:
        raise SystemExit(f"stage 3 runs missing: {missing}")
    eps = sorted(set.intersection(*[set(v) for v in data.values()]))
    ns = range(1, a.n_max + 1)
    succ = {(t, n): float(np.mean([W.outcome(data[t][e], n)[0] for e in eps])) for t in data for n in ns}
    fdecl = {(t, n): float(np.mean([W.outcome(data[t][e], n)[1] for e in eps])) for t in data for n in ns}
    nodecl = {(t, n): float(np.mean([not W.outcome(data[t][e], n)[2] for e in eps])) for t in data for n in ns}

    print(f"calib episodes common to every run: {len(eps)}")
    print(f"\nsuccess@B (false declarations) by confirmation count N; the selection uses N = {N}")
    print(f"{'condition':<16}" + "".join(f"{'N=' + str(n):>16}" for n in ns))
    for t in data:
        print(f"{t:<16}" + "".join(f"{succ[(t, n)]:>9.3f} ({fdecl[(t, n)]:.2f})" for n in ns))

    best = max(THRESH, key=lambda k: (succ[(f"sam3_{k}", N)], -nodecl[(f"sam3_{k}", N)], -THRESH[k]))
    sel = {"sam3_threshold": THRESH[best], "confirmations": N, "sam3_success": succ[(f"sam3_{best}", N)],
           "sam3_false_declarations": fdecl[(f"sam3_{best}", N)], "sam3_no_declaration": nodecl[(f"sam3_{best}", N)],
           "n_calib_episodes": len(eps), "success": {f"{t}|N={n}": v for (t, n), v in succ.items()}}
    (root / "selection.json").write_text(json.dumps(sel, indent=1))
    print(f"\nselected: SAM 3 threshold {sel['sam3_threshold']} (calib success@B {sel['sam3_success']:.3f}), N = {N}; "
          f"wrote {root / 'selection.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
