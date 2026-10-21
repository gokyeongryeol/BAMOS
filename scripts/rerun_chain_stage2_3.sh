#!/usr/bin/env bash
# After stage 1: fit every channel (stage 2), run stage 3 (calib closed loop) for all conditions, then select t and N.
set -u
# PY: the Python that runs the experiment code (default: this repo's uv venv)
PY=${PY:-uv run python}
cd "$(dirname "$0")/.."
CAL=results/raw/calib
until grep -q STAGE1_DONE results/raw/stage1.log; do sleep 60; done
echo "$(date +%H:%M) stage 1 done"
bash scripts/rerun_stage2_fit.sh all
echo "$(date +%H:%M) stage 2 fits done"
bash scripts/rerun_stage3_select.sh oracle > results/raw/stage3_oracle.log 2>&1 &
bash scripts/rerun_stage3_select.sh open_set
wait
$PY scripts/select_stage3.py --root results/raw/stage3 > results/raw/stage3/selection.log 2>&1
echo "$(date +%H:%M) selection done: $(tail -n 1 results/raw/stage3/selection.log)"
echo "CHAIN_2_3_DONE"
