#!/usr/bin/env bash
# Stage 13 (2026-10-11): oracle error levels on one grid, {0.1, 0.2, 0.3} for both
# missed detections and false masks. New levels: missed detections 0.1, 0.2, 0.3 and false masks 0.2, 0.3 (0.1 is
# stage 8's s8_fp010), each calibrated and fitted as stage 8's s8_fp levels, then tested with BAMOS. The test reads
# each level's fitted constants straight from its calib directory.
#   GROUP=calib|fit|test bash scripts/rerun_stage13_errors.sh
set -u
PY=${PY:-uv run python}
cd "$(dirname "$0")/.."
C=${C:-configs/conditions}
CAL=${CAL:-results/raw/calib}
OUT=${OUT:-results/raw/test}
GROUP=${GROUP:?calib|fit|test}
EPISODES=${EPISODES:-80}
LOG=results/raw/stage13.log
mkdir -p results/raw
echo "$(date +%H:%M) stage 13 $GROUP at $(git rev-parse --short HEAD)" | tee -a "$LOG"
FN="s13_fn010:0.1 s13_fn020:0.2 s13_fn030:0.3"
FP="s13_fp020:0.2 s13_fp030:0.3"
case "$GROUP" in
calib)
  g=0
  for spec in $FN $FP; do
    name=${spec%%:*}; rate=${spec##*:}; key=fn_rate; [[ $name == *_fp* ]] && key=fp_rate
    if ls "$CAL/$name"/frames_*.csv > /dev/null 2>&1; then echo "skip calib $name"; g=$((g + 1)); continue; fi
    mkdir -p "$CAL/$name"
    ( $PY scripts/calibrate_channel.py --channel oracle --out "$CAL/$name" --episodes "$EPISODES" --gpu "$g" \
        --override "sensor_detector.$key=$rate" > "$CAL/$name/shard_0.log" 2>&1
      echo "$(date +%H:%M) calib $name: $(grep -o 'wrote.*' "$CAL/$name/shard_0.log")" | tee -a "$LOG" ) &
    g=$((g + 1))
  done
  wait ;;
fit)
  names=$(for spec in $FN $FP; do echo -n "${spec%%:*} "; done)
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=${FIT_GPU:-7} $PY scripts/calib_voxel_belief.py --calib "$CAL" --reuse-occupancy --channels $names \
    > "$CAL/voxel_belief_stage13.log" 2>&1 || { echo "replay failed"; exit 1; }
  export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
  for n in $names; do
    ( $PY scripts/fit_channel_constants.py --root "$CAL/$n" --a0 0.0001 > "$CAL/$n/fit.log" 2>&1
      echo "$(date +%H:%M) fitted $n: $(cat "$CAL/$n/overrides.txt" 2>/dev/null)" | tee -a "$LOG" ) &
  done
  wait ;;
test)
  if ! git diff --quiet HEAD -- configs src scripts || [ -n "$(git ls-files --others --exclude-standard configs src scripts | grep -v '/\.nfs')" ]; then
    echo "configs/, src/ or scripts/ differ from HEAD: commit the frozen state before the test runs"; exit 1
  fi
  COMMON="--methods bamos --config configs/normal.yaml --steps --override stop.mode=joint $(cat "$C/common.args")"
  arm () {  # arm <tag> <level> <key> <rate> <gpus> <extra args...>
    tag=$1; lvl=$2; key=$3; rate=$4; g=$5; shift 5
    if [ -s "$OUT/$tag/episodes.csv" ] && grep -q "^wrote" "$OUT/$tag.log" 2>/dev/null; then echo "skip $tag"; return; fi
    [ -s "$CAL/$lvl/overrides.txt" ] || { echo "no constants for $lvl: skip $tag"; return; }
    # shellcheck disable=SC2046,SC2086
    $PY scripts/run_experiment.py $COMMON --out "$OUT/$tag" --experiment-id "test_$tag" \
      --override "sensor_detector.$key=$rate" $(cat "$CAL/$lvl/overrides.txt") --gpus "$g" --workers 6 "$@" > "$OUT/$tag.log" 2>&1
    echo "$(date +%H:%M) done $tag: $(awk -F, 'NR>1{print $8}' "$OUT/$tag/episodes.csv" | sort | uniq -c | tr '\n' ' ')" | tee -a "$LOG"
  }
  # BAMOS at every level, one level per pair of GPUs
  arm s13_fn010_bamos s13_fn010 fn_rate 0.1 0,1 &
  arm s13_fn020_bamos s13_fn020 fn_rate 0.2 2,3 &
  ( arm s13_fn030_bamos s13_fn030 fn_rate 0.3 4,5; arm s13_fp030_bamos s13_fp030 fp_rate 0.3 4,5 ) &
  arm s13_fp020_bamos s13_fp020 fp_rate 0.2 6,7 &
  wait ;;
esac
echo "STAGE13_${GROUP}_DONE" | tee -a "$LOG"
