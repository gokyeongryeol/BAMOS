#!/usr/bin/env bash
# Stage 10 (2026-10-08): the arms SAM 3 has and Falcon Perception lacked, on the
# frozen falcon condition and the stage 5 test split (Isaac + Falcon servers up).
#   FALCON_PORTS="--isaac-ports ... --detector-ports ..." GPUS=0,...,7 WORKERS=8 bash scripts/isaac/stage10_falcon.sh
set -u
PY=${PY:-uv run python}
cd "$(dirname "$0")/../.."
C=${C:-configs/isaac_conditions}
OUT=${OUT:-results/raw/test}
LIMIT=${LIMIT:-}
FALCON_PORTS=${FALCON_PORTS:-"--isaac-ports 5555,5556,5557 --detector-ports 5620,5621,5622"}
GPUS=${GPUS:-0,1,2}; WORKERS=${WORKERS:-3}
LOG=results/raw/stage10.log
mkdir -p results/raw
echo "$(date +%H:%M) stage 10 at $(git rev-parse --short HEAD)" | tee -a "$LOG"
if [ -z "$LIMIT" ] && { ! git diff --quiet HEAD -- configs src scripts || [ -n "$(git ls-files --others --exclude-standard configs src scripts | grep -v '/\.nfs')" ]; }; then
  echo "configs/, src/ or scripts/ differ from HEAD: commit the frozen state before the test runs"; exit 1
fi
COMMON="--methods bamos --config configs/isaac.yaml --steps --override stop.mode=joint $(cat "$C/common.args")${LIMIT:+ --limit $LIMIT}"
S="--backend isaac --override environment.isaac.render_cache_dir=results/cache/isaac $FALCON_PORTS --workers $WORKERS --gpus $GPUS"
arm () {  # arm <tag> <extra args...>, the falcon condition, as scripts/isaac/stage8_extra.sh
  tag=$1; shift
  if [ -s "$OUT/$tag/episodes.csv" ] && grep -q "^wrote" "$OUT/$tag.log" 2>/dev/null; then echo "skip $tag"; return; fi
  # shellcheck disable=SC2046,SC2086
  $PY scripts/isaac/run.py $COMMON --out "$OUT/$tag" --experiment-id "test_$tag" \
    $(cat "$C/falcon.channel") $(cat "$C/falcon.constants") $S "$@" > "$OUT/$tag.log" 2>&1
  echo "$(date +%H:%M) done $tag: $(awk -F, 'NR>1{print $8}' "$OUT/$tag/episodes.csv" | sort | uniq -c | tr '\n' ' ')" | tee -a "$LOG"
}
GT="--override mapping.occupancy_source=ground_truth"
arm fa_scene_bstop --override planner.utility_mode=scene_coverage
arm fa_ig_fdet     --override stop.rule=first_det
arm fa_cov_bstop   --override planner.utility_mode=coverage
arm fa_gt_bamos    $GT
echo "STAGE10_DONE" | tee -a "$LOG"
