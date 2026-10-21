"""SAM 3's score threshold as Isaac chose it: the highest train closed-loop success (results/hssd/calib_runs/sam3_t*),
ties to the lower no-declaration rate, then to the lower threshold. Writes results/hssd/calib/sam3_threshold.txt.

    uv run python scripts/hssd_sam3_select.py
"""
from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    rows = []
    for t in ("050", "060", "070", "080", "090"):
        p = Path(f"results/hssd/calib_runs/sam3_t{t}/episodes.jsonl")
        r = [json.loads(l) for l in open(p)]
        ok = [x for x in r if x["status"] == "ok"]
        sr = sum(x["success"] for x in ok) / len(ok)
        nd = sum(not x["declared_stop"] for x in ok) / len(ok)
        rows.append((t, sr, nd))
        print(f"t{t}: SR {sr:.4f}, no-declaration {nd:.4f} ({len(ok)} ok of {len(r)})")
    best = sorted(rows, key=lambda x: (-x[1], x[2], x[0]))[0]
    Path("results/hssd/calib/sam3_threshold.txt").write_text(f"sam3_t{best[0]}\n")
    print(f"chosen: sam3_t{best[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
