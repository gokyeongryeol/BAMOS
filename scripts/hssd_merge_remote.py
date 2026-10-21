"""Merge <arm>_remote (rows and step logs written by the second machine) into <arm>, then remove <arm>_remote.

    uv run python scripts/hssd_merge_remote.py results/hssd/val/sam3_no_nodet
"""
import json
import shutil
import sys
from pathlib import Path

arm = Path(sys.argv[1])
rem = Path(str(arm) + "_remote")
have = {(r["scene"], r["episode_id"]) for r in map(json.loads, open(arm / "episodes.jsonl"))}
add, seen = [], set(have)
for r in map(json.loads, open(rem / "episodes.jsonl")):
    k = (r["scene"], r["episode_id"])
    if k not in seen and r["status"] == "ok":
        add.append(r)
        seen.add(k)
with open(arm / "episodes.jsonl", "a") as f:
    for r in add:
        f.write(json.dumps(r) + "\n")
        src = rem / "steps" / f"{r['scene']}_{r['episode_id']}.jsonl"
        if src.exists():
            shutil.copy(src, arm / "steps" / src.name)
shutil.rmtree(rem)
print(f"{arm}: merged {len(add)} rows; now {len(have) + len(add)}")
