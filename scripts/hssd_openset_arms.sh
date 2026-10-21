#!/usr/bin/env bash
# The HSSD open-set val arms (200 views) named on the command line, for one detector channel, in order.
#   bash scripts/hssd_openset_arms.sh <label> <channel dir> <base port> <n shards> <detector ports> <arm>...
#   e.g. bash scripts/hssd_openset_arms.sh sam3 results/hssd/calib/sam3_t090 5950 8 5610,5611 scene_only kappa1
# Arms: bamos scene_only target_only no_nodet kappa1.
# Outputs results/hssd/val/<label>_<arm>; an arm's error rows are retried once. PY selects the Python.
set -u
cd "$(dirname "$0")/.."
label=$1 fit=$2 base=$3 n=$4 ports=$5
shift 5
L=results/hssd/logs
V=results/hssd/val
PY=${PY:-.venv/bin/python}
export PY PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
stamp() { echo "[$(date -u +%H:%M:%S)] $*"; }
for a in "$@"; do
  case $a in
    bamos) ov=() ;;
    scene_only) ov=(--override planner.utility_mode=scene_coverage --override stop.rule=first_det) ;;
    target_only) ov=(--override ablation.use_voxel_belief=false) ;;
    no_nodet) ov=(--override ablation.use_nodet=false) ;;
    kappa1) ov=(--override ablation.use_kappa=false) ;;
    *) echo "unknown arm $a"; exit 1 ;;
  esac
  name=${label}_$a
  stamp "arm $name"
  DETECTOR_PORTS=$ports bash scripts/run_hssd_openset_arm.sh $V/$name $base $n --fit $fit "${ov[@]}" >> $L/arms_${label}_$base.log 2>&1
  if grep -q '"status": "error"' $V/$name/episodes.jsonl; then
    $PY scripts/hssd_drop_errors.py $V/$name >> $L/arms_${label}_$base.log 2>&1
    DETECTOR_PORTS=$ports bash scripts/run_hssd_openset_arm.sh $V/$name $base $n --fit $fit "${ov[@]}" >> $L/arms_${label}_$base.log 2>&1
  fi
  stamp "arm $name: $(wc -l < $V/$name/episodes.jsonl) rows, $(grep -c '"status": "error"' $V/$name/episodes.jsonl) errors"
done
stamp "ARMS_DONE $label $*"
