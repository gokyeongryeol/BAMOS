#!/usr/bin/env bash
# One HSSD val arm with the true geometry as the scene model (mapping.occupancy_source=ground_truth), retried once on
# error rows. variant: bamos (IG views, belief stop) | scene_only (scene coverage, first mask).
#   bash scripts/hssd_gt_arm.sh <out name> <channel dir> <variant> <base port> <n> [detector ports]
set -u
cd "$(dirname "$0")/.."
name=$1 fit=$2 variant=$3 base=$4 n=$5 ports=${6:-}
PY=${PY:-.venv/bin/python}
export PY PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
V=results/hssd/val
L=results/hssd/logs
case $variant in
  bamos) ov=() ;;
  scene_only) ov=(--override planner.utility_mode=scene_coverage --override stop.rule=first_det) ;;
  *) echo "unknown variant $variant"; exit 1 ;;
esac
run() {
  if [ -n "$ports" ]; then
    DETECTOR_PORTS=$ports bash scripts/run_hssd_openset_arm.sh $V/$name $base $n --fit $fit --override mapping.occupancy_source=ground_truth "${ov[@]}"
  else
    bash scripts/run_hssd_arm.sh $V/$name $base $n --fit $fit --override mapping.occupancy_source=ground_truth "${ov[@]}"
  fi
}
echo "[$(date -u +%H:%M:%S)] arm $name" >> $L/arms_gt.log
run >> $L/arms_gt.log 2>&1
if grep -q '"status": "error"' $V/$name/episodes.jsonl; then
  $PY scripts/hssd_drop_errors.py $V/$name >> $L/arms_gt.log 2>&1
  run >> $L/arms_gt.log 2>&1
fi
echo "[$(date -u +%H:%M:%S)] arm $name: $(wc -l < $V/$name/episodes.jsonl) rows, $(grep -c '"status": "error"' $V/$name/episodes.jsonl) errors" >> $L/arms_gt.log
