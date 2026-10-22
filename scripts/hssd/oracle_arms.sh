#!/usr/bin/env bash
# HSSD oracle-channel val arms named <channel>_<variant> (channel: oracle, fn010-fn030, fp010-fp030, the fit in
# $HSSD_CALIB/<channel>: default results/hssd/calib, or configs/hssd_conditions for the fits shipped with the
# repository; variant: bamos, scene_only, no_nodet), in order, on teleport servers <base>..<base+n-1>; an arm's error
# rows are retried once. PY selects the Python.
#   bash scripts/hssd/oracle_arms.sh <base> <n> <arm>...    e.g. bash scripts/hssd/oracle_arms.sh 5828 8 fp010_bamos
set -u
cd "$(dirname "$0")/../.."
base=$1 n=$2
shift 2
PY=${PY:-.venv/bin/python}
export PY PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=results/hssd/logs
V=results/hssd/val
stamp() { echo "[$(date -u +%H:%M:%S)] $*"; }
for name in "$@"; do
  ch=${name%%_*}
  case ${name#*_} in
    bamos) ov=() ;;
    scene_only) ov=(--override planner.utility_mode=scene_coverage --override stop.rule=first_det) ;;
    no_nodet) ov=(--override ablation.use_nodet=false) ;;
    *) echo "unknown arm $name"; exit 1 ;;
  esac
  stamp "arm $name"
  bash scripts/hssd/run_arm.sh $V/$name $base $n --fit ${HSSD_CALIB:-results/hssd/calib}/$ch "${ov[@]}" >> $L/arms_oracle_$base.log 2>&1
  if grep -q '"status": "error"' $V/$name/episodes.jsonl; then
    $PY scripts/hssd/drop_errors.py $V/$name >> $L/arms_oracle_$base.log 2>&1
    bash scripts/hssd/run_arm.sh $V/$name $base $n --fit ${HSSD_CALIB:-results/hssd/calib}/$ch "${ov[@]}" >> $L/arms_oracle_$base.log 2>&1
  fi
  stamp "arm $name: $(wc -l < $V/$name/episodes.jsonl) rows, $(grep -c '"status": "error"' $V/$name/episodes.jsonl) errors"
done
stamp "ARMS_DONE $*"
