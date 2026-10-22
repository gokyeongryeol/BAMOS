#!/usr/bin/env bash
# Re-run the episodes of a test arm that failed for an infrastructure reason (CUDA out of memory, a dropped server
# connection), not because no solvable episode could be built, and merge them into the arm: the rows replace the
# failed ones in <out>/<tag>/episodes.csv and the step logs join <out>/<tag>/steps/. Same frozen condition and
# arguments as the arm (scripts/isaac/stage5_test.sh, scripts/isaac/stage8_extra.sh).
#
#   bash scripts/isaac/retry_failed.sh <tag> <condition> <extra args...>     (e.g. --gpus 6,7 --workers 2 --override ...)
set -u
PY=${PY:-uv run python}
cd "$(dirname "$0")/../.."
C=${C:-configs/isaac_conditions}
OUT=${OUT:-results/raw/test}
tag=$1; cond=$2; shift 2
ids=$(python3 - "$OUT/$tag/episodes.csv" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
print(",".join(r["episode_id"] for r in rows if r["status"] != "ok" and "solvable" not in r["error"]))
PY
)
[ -n "$ids" ] || { echo "$tag: nothing to retry"; exit 0; }
echo "$(date +%H:%M) retry $tag: $ids"
COMMON="--methods bamos --config configs/isaac.yaml --steps --override stop.mode=joint $(cat "$C/common.args")"
R="$OUT/${tag}_retry"
# shellcheck disable=SC2046,SC2086
$PY scripts/isaac/run.py $COMMON --out "$R" --experiment-id "test_$tag" --episode-ids "$ids" \
  $(cat "$C/$cond.channel") $(cat "$C/$cond.constants") "$@" > "$R.log" 2>&1
python3 - "$OUT/$tag" "$R" <<'PY'
import csv, shutil, sys
from pathlib import Path
arm, retry = Path(sys.argv[1]), Path(sys.argv[2])
new = {r["episode_id"]: r for r in csv.DictReader(open(retry / "episodes.csv"))}
rows = list(csv.DictReader(open(arm / "episodes.csv")))
fields = list(rows[0])
out = [new.get(r["episode_id"], r) if r["status"] != "ok" else r for r in rows]
with open(arm / "episodes.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(out)
for ep, r in new.items():
    if r["status"] == "ok":
        shutil.copy(retry / "steps" / f"{ep}.jsonl", arm / "steps" / f"{ep}.jsonl")
print(f"merged {sum(r['status'] == 'ok' for r in new.values())}/{len(new)} retried episodes into {arm}")
PY
