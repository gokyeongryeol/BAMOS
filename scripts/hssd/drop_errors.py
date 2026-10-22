"""Remove rows of a scripts/hssd/run.py output (and their step logs) so that a rerun redoes only those episodes.

    uv run python scripts/hssd/drop_errors.py results/hssd/val/oracle_scene_only
    uv run python scripts/hssd/drop_errors.py --exhausted-below 200 results/hssd/val/oracle_bamos

Error rows always; with --exhausted-below B also the rows that ran out of a budget smaller than B
(termination budget_exhausted, fewer than B active views). The budget only bounds the view loop, so a rerun with budget B
replays such an episode's views and continues it: the same as having run it with B from the start (checked on 63
episodes, 2026-10-09). The removed rows are kept in <run>/errors.jsonl or <run>/exhausted.jsonl (appended).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--exhausted-below", type=int, default=0)
    a = ap.parse_args()
    for run in map(Path, a.runs):
        rows = [json.loads(l) for l in open(run / "episodes.jsonl")]
        err = [r for r in rows if r["status"] != "ok"]
        exh = [r for r in rows if r["status"] == "ok" and a.exhausted_below and r["termination_reason"] == "budget_exhausted"
               and int(r["active_views"]) < a.exhausted_below]
        drop = {id(r) for r in err + exh}
        keep = [r for r in rows if id(r) not in drop]
        for name, part in (("errors.jsonl", err), ("exhausted.jsonl", exh)):
            if part:
                with open(run / name, "a") as f:
                    for r in part:
                        f.write(json.dumps(r) + "\n")
        (run / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in keep))
        n_steps = 0
        for r in err + exh:
            p = run / "steps" / f"{r['scene']}_{r['episode_id']}.jsonl"
            if p.exists():
                p.unlink()
                n_steps += 1
        print(f"{run}: kept {len(keep)}, removed {len(err)} error rows, {len(exh)} budget-exhausted rows, {n_steps} step logs")


if __name__ == "__main__":
    main()
