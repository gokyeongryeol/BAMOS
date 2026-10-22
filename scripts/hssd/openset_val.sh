#!/usr/bin/env bash
# Local machine: the HSSD val arms with the open-set detectors (200 views), SAM 3 at the threshold of
# calib/sam3_threshold.txt and Falcon Perception, each with its channel's train fit (scripts/hssd/run.py --fit).
# Needs val teleport servers 5950..5965 (8 per detector) and the detector servers SAM 3 5610..5613, Falcon 5600..5603.
set -u
cd "$(dirname "$0")/../.."
C=${HSSD_CALIB:-results/hssd/calib}  # configs/hssd_conditions: the fits shipped with the repository
L=results/hssd/logs
V=results/hssd/val
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
stamp() { echo "[$(date -u +%H:%M:%S)] $*"; }
SAM3=$(tr -d '\n' < $C/sam3_threshold.txt)
SCENE="--override planner.utility_mode=scene_coverage --override stop.rule=first_det"

chain() {  # chain <label> <channel dir> <base port> <detector ports>
  local label=$1 fit=$2 base=$3 ports=$4
  arm() {
    local name=${label}_$1
    shift
    stamp "arm $name"
    DETECTOR_PORTS=$ports bash scripts/hssd/run_openset_arm.sh $V/$name $base 8 --fit $fit "$@" >> $L/arms_${label}.log 2>&1
    if grep -q '"status": "error"' $V/$name/episodes.jsonl; then
      .venv/bin/python scripts/hssd/drop_errors.py $V/$name >> $L/arms_${label}.log 2>&1
      DETECTOR_PORTS=$ports bash scripts/hssd/run_openset_arm.sh $V/$name $base 8 --fit $fit "$@" >> $L/arms_${label}.log 2>&1
    fi
    stamp "arm $name: $(wc -l < $V/$name/episodes.jsonl) rows, $(grep -c '"status": "error"' $V/$name/episodes.jsonl) errors"
  }
  arm bamos
  arm scene_only $SCENE
  arm target_only --override ablation.use_voxel_belief=false
  arm no_nodet --override ablation.use_nodet=false
  arm kappa1 --override ablation.use_kappa=false
}
chain sam3 $C/$SAM3 5950 5610,5611,5612,5613 &
chain falcon $C/falcon 5958 5600,5601,5602,5603 &
wait
stamp "OPENSET_VAL_DONE"
