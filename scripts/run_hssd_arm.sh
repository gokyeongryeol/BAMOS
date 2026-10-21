#!/usr/bin/env bash
# One arm on all of val: N shards of scripts/run_hssd.py against the teleport servers $BASE.., GPU i for shard i.
#   bash scripts/run_hssd_arm.sh <out dir> <base port> <n shards> [run_hssd.py args...]
set -u
cd "$(dirname "$0")/.."
OUT=$1; BASE=$2; N=$3; shift 3
mkdir -p "$OUT"
for ((i = 0; i < N; i++)); do
  ${PY:-.venv/bin/python} scripts/run_hssd.py --port $((BASE + i)) --gpu $((i % 8)) --shard $i/$N --steps --out "$OUT" "$@" > "$OUT/run_$i.log" 2>&1 &
done
wait
echo "$OUT: $(wc -l < "$OUT/episodes.jsonl") rows, $(grep -c '"status": "error"' "$OUT/episodes.jsonl") errors"
