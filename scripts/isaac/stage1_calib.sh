#!/usr/bin/env bash
# Re-run stage 1 (README.md, "Isaac Sim pipeline"): calibration frames of every detector channel on the calib split.
#
#   bash scripts/isaac/stage1_calib.sh
#
# The oracle family runs on the analytic renderer, one channel per GPU. The open-set channels need running servers
# (scripts/isaac/launch_servers.sh): SAM3 pairs Isaac and SAM 3 ports, one shard per
# pair. Output: results/raw/calib/<channel>/frames_<shard>.csv.
set -u
# PY: the Python that runs the experiment code (default: this repo's uv venv)
PY=${PY:-uv run python}
cd "$(dirname "$0")/../.."
OUT=${OUT:-results/raw/calib}
EPISODES=${EPISODES:-80}  # calib episodes per channel: stride 5 over 4 archetypes x 5 layouts x 20 (fewer only for a smoke test)
SAM3=${SAM3:-"5555:5610 5556:5611 5557:5612 5558:5613 5559:5614 5560:5615"}   # isaac:detector
# stage 2 reads every frames_*.csv under a channel: frames left from an earlier collection would mix in
if ls "$OUT"/*/frames_*.csv > /dev/null 2>&1; then
  echo "$OUT already holds calib frames: move or delete them before stage 1"; exit 1
fi
mkdir -p "$OUT"

oracle_channel () {  # oracle_channel <name> <gpu> [overrides...]
  name=$1; gpu=$2; shift 2
  ov=(); for o in "$@"; do ov+=(--override "$o"); done
  mkdir -p "$OUT/$name"
  $PY scripts/isaac/calibrate.py --channel oracle --out "$OUT/$name" --episodes "$EPISODES" --gpu "$gpu" "${ov[@]}" \
    > "$OUT/$name/shard_0.log" 2>&1
  echo "$(date +%H:%M) done $name: $(grep -o 'wrote.*' "$OUT/$name/shard_0.log")"
}

open_set_channel () {  # open_set_channel <name> "<isaac:detector> ..."
  name=$1; read -ra pairs <<< "$2"; n=${#pairs[@]}
  mkdir -p "$OUT/$name"
  for i in "${!pairs[@]}"; do
    IFS=: read -r ip dp <<< "${pairs[$i]}"
    $PY scripts/isaac/calibrate.py --channel open_set --out "$OUT/$name" --episodes "$EPISODES" --isaac-port "$ip" \
      --detector-port "$dp" --shard "$i/$n" > "$OUT/$name/shard_$i.log" 2>&1 &
  done
  wait
  echo "$(date +%H:%M) done $name: $(cat "$OUT/$name"/shard_*.log | grep -c '^wrote')/$n shards"
}

echo "$(date +%H:%M) stage 1 start"
# 2026-10-06 re-run: the oracle (analytic, GPU 6), SAM 3 (GPUs 0-5); the injected-error channels are stages 8 and 13
oracle_channel oracle        6 &
open_set_channel sam3 "$SAM3" &
wait
echo "STAGE1_DONE"
