#!/usr/bin/env bash
# Stage 9 (2026-10-08): a perfect scene model, the true geometry in place of the
# occupancy belief (mapping.occupancy_source=ground_truth), with and without the target belief; oracle and SAM 3, the
# frozen constants, the stage 5 test split.
#   GROUP=test_a bash scripts/isaac/stage9_worldmodel.sh   # oracle arms (analytic)
#   GROUP=test_o bash scripts/isaac/stage9_worldmodel.sh   # SAM 3 arms (Isaac + SAM 3 servers up)
set -u
PY=${PY:-uv run python}
cd "$(dirname "$0")/../.."
C=${C:-configs/isaac_conditions}
OUT=${OUT:-results/raw/test}
GROUP=${GROUP:?test_a|test_o}
LIMIT=${LIMIT:-}
A_GPUS=${A_GPUS:-6,7}
SAM3_PORTS=${SAM3_PORTS:-"--isaac-ports 5555,5556,5557,5558,5559,5560 --detector-ports 5610,5611,5612,5613,5614,5615"}
LOG=results/raw/stage9.log
mkdir -p results/raw
echo "$(date +%H:%M) stage 9 $GROUP at $(git rev-parse --short HEAD)" | tee -a "$LOG"
if [ -z "$LIMIT" ] && { ! git diff --quiet HEAD -- configs src scripts || [ -n "$(git ls-files --others --exclude-standard configs src scripts | grep -v '/\.nfs')" ]; }; then
  echo "configs/, src/ or scripts/ differ from HEAD: commit the frozen state before the test runs"; exit 1
fi
COMMON="--methods bamos --config configs/isaac.yaml --steps --override stop.mode=joint $(cat "$C/common.args")${LIMIT:+ --limit $LIMIT}"
ISAAC="--backend isaac --override environment.isaac.render_cache_dir=results/cache/isaac"
GT="--override mapping.occupancy_source=ground_truth"
arm () {  # arm <tag> <condition> <extra args...>, as scripts/isaac/stage8_extra.sh
  tag=$1; cond=$2; shift 2
  if [ -s "$OUT/$tag/episodes.csv" ] && grep -q "^wrote" "$OUT/$tag.log" 2>/dev/null; then echo "skip $tag"; return; fi
  # shellcheck disable=SC2046,SC2086
  $PY scripts/isaac/run.py $COMMON --out "$OUT/$tag" --experiment-id "test_$tag" \
    $(cat "$C/$cond.channel") $(cat "$C/$cond.constants") "$@" > "$OUT/$tag.log" 2>&1
  echo "$(date +%H:%M) done $tag: $(awk -F, 'NR>1{print $8}' "$OUT/$tag/episodes.csv" | sort | uniq -c | tr '\n' ' ')" | tee -a "$LOG"
}
if [ "$GROUP" = test_a ]; then
  A="--gpus $A_GPUS --workers ${A_WORKERS:-8}"
  arm a_gt_bamos        oracle $A $GT
else
  SAM3=$(python3 -c "import json;print(json.load(open('$C/frozen.json'))['sam3_condition'])")
  S="$ISAAC $SAM3_PORTS --workers 6 --gpus 0,1,2,3,4,5"
  arm bp_gt_bamos       "$SAM3" $S $GT
fi
echo "STAGE9_${GROUP}_DONE" | tee -a "$LOG"
