#!/usr/bin/env bash
# run_hssd_arm.sh with the open-set detector servers spread over the shards (a separate file: run_hssd_arm.sh is read
# by running chains, and bash reads a script as it goes).
#   DETECTOR_PORTS=5610,5611 bash scripts/run_hssd_openset_arm.sh <out dir> <base port> <n shards> [run_hssd.py args...]
# DETECTOR_PORTS=5610,5611,...: shard i uses the open-set detector server ports[i mod n] (--detector-port).
set -u
cd "$(dirname "$0")/.."
OUT=$1; BASE=$2; N=$3; shift 3
mkdir -p "$OUT"
IFS=, read -ra DP <<< "${DETECTOR_PORTS:-}"
for ((i = 0; i < N; i++)); do
  det=()
  [ ${#DP[@]} -gt 0 ] && det=(--detector-port "${DP[$((i % ${#DP[@]}))]}")
  ${PY:-.venv/bin/python} scripts/run_hssd.py --port $((BASE + i)) --gpu $((i % 8)) --shard $i/$N --steps --out "$OUT" "${det[@]}" "$@" > "$OUT/run_$i.log" 2>&1 &
done
wait
echo "$OUT: $(wc -l < "$OUT/episodes.jsonl") rows, $(grep -c '"status": "error"' "$OUT/episodes.jsonl") errors"
