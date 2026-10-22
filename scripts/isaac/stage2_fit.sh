#!/usr/bin/env bash
# Re-run stage 2 (README.md, "Isaac Sim pipeline"): fit every channel's constants from its stage 1 frames.
#   eta_f, eta_t, k jointly by maximum likelihood over all frames, on the model's own area a(v*, s*) (eta_f from the out-of-view frames is kept as a
#   diagnostic); a0 fitted for the open-set channels and held at
#   the DET area threshold for the oracle family; kappa's beta, lambda jointly with them (the closed form is a diagnostic). The same
#   procedure for every channel, the oracle included. Writes results/raw/calib/<channel>/overrides*.txt and fit*.json.
#
#   bash scripts/isaac/stage2_fit.sh [oracle|open_set|all]
set -u
# PY: the Python that runs the experiment code (default: this repo's uv venv)
PY=${PY:-uv run python}
cd "$(dirname "$0")/../.."
WHAT=${1:-all}
# SAM 3 score thresholds to fit (the server serves instances scoring >= 0.5); tag t<100 x threshold>
SAM3_THRESHOLDS=${SAM3_THRESHOLDS:-"0.5 0.6 0.7 0.8 0.9"}
CAL=${CAL:-results/raw/calib}
# the Bernoulli voxel belief's p0, q_occ, q_free, and every channel's frames at the true hypothesis: the model's own area a(v*, s*) and
# the belief's front-face transmittance T(v*, s*) before the frame, the fit's inputs
# SKIP_REPLAY=1: the replay's outputs are already there (a re-fit only)
if [ "${SKIP_REPLAY:-0}" != 1 ]; then
  $PY scripts/isaac/calib_voxel_belief.py --calib "$CAL" > "$CAL/voxel_belief.log" 2>&1 || { echo "voxel belief failed"; exit 1; }
fi
# the fits run side by side on small tensors: cap each one's threads, or they fight over the cores
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
fit () {  # fit <channel> <log name> <args...>
  ch=$1; log=$2; shift 2
  $PY scripts/isaac/fit_constants.py --root "$CAL/$ch" "$@" > "$CAL/$ch/$log" 2>&1
  echo "$(date +%H:%M) fitted $ch ($log): $(grep -c 'wrote' "$CAL/$ch/$log")"
}
if [ "$WHAT" = oracle ] || [ "$WHAT" = all ]; then
  fit oracle fit.log --a0 0.0001
fi
if [ "$WHAT" = open_set ] || [ "$WHAT" = all ]; then
  for th in $SAM3_THRESHOLDS; do
    tag=$(awk -v t="$th" 'BEGIN{printf "t%03d", t * 100 + 0.5}')
    fit sam3 "fit_$tag.log" --a0 fit --min-score "$th" --mask-select score --out-name "$tag" &
  done
  wait
fi
echo "STAGE2_${WHAT}_DONE"
