"""Split an arm's unfinished val episodes between two machines (after its workers were stopped).

    uv run python scripts/hssd_split_remaining.py results/hssd/val/sam3_no_nodet 0.5 \
        results/hssd/logs/a_local.txt results/hssd/logs/a_remote.txt

Unfinished: the val episodes (any complete arm's list) not in <arm>/episodes.jsonl nor in <arm>_remote/episodes.jsonl.
Their part-written step logs are removed. The first <share> of them (sorted) go to the first list, the rest to the
second; the second machine writes to <arm>_remote, merged afterwards (scripts/hssd_merge_remote.py).
"""
import json
import sys
from pathlib import Path

arm, share, out_a, out_b = Path(sys.argv[1]), float(sys.argv[2]), sys.argv[3], sys.argv[4]
allk = [(r["scene"], r["episode_id"]) for r in map(json.loads, open("results/hssd/val/oracle_bamos/episodes.jsonl"))]
done = set()
for d in (arm, Path(str(arm) + "_remote")):
    if (d / "episodes.jsonl").exists():
        done |= {(r["scene"], r["episode_id"]) for r in map(json.loads, open(d / "episodes.jsonl"))}
left = sorted(set(allk) - done)
for d in (arm, Path(str(arm) + "_remote")):
    for s, e in left:
        p = d / "steps" / f"{s}_{e}.jsonl"
        if p.exists():
            p.unlink()
k = int(round(share * len(left)))
Path(out_a).write_text(",".join(f"{s}:{e}" for s, e in left[:k]))
Path(out_b).write_text(",".join(f"{s}:{e}" for s, e in left[k:]))
print(f"{arm}: {len(done)} done, {len(left)} left -> {k} / {len(left) - k}")
