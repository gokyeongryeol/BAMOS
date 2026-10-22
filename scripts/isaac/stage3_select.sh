#!/usr/bin/env bash
# Re-run stage 3 (README.md, "Isaac Sim pipeline"): BAMOS in closed loop on the calib episodes of stage 1, with each
# condition's fitted constants, to choose SAM 3's score threshold and the stop rule's confirmation count N
# (scripts/isaac/select_threshold.py). Nothing here touches the test split.
#
#   bash scripts/isaac/stage3_select.sh oracle          # the oracle (needs its stage 2 fit)
#   bash scripts/isaac/stage3_select.sh open_set        # SAM 3 (needs servers and its fits)
set -u
# PY: the Python that runs the experiment code (default: this repo's uv venv)
PY=${PY:-uv run python}
cd "$(dirname "$0")/../.."
WHAT=${1:?oracle|open_set}
CAL=${CAL:-results/raw/calib}
OUT=${OUT:-results/raw/stage3}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
SAM3_PORTS=${SAM3_PORTS:-"--isaac-ports 5555,5556,5557,5558,5559,5560 --detector-ports 5610,5611,5612,5613,5614,5615"}
SAM3_THRESHOLDS=${SAM3_THRESHOLDS:-"0.5 0.6 0.7 0.8 0.9"}  # as in scripts/isaac/stage2_fit.sh
LIMIT=${LIMIT:-80}  # calib episodes, as in stage 1 (fewer only for a smoke test)
# the voxel belief's constants fitted in stage 2 (scripts/isaac/calib_voxel_belief.py), shared by every condition: the detector
# constants were fitted with T_bar under them, so the closed loop must use them too (stage 4 freezes the same values)
OCC=$(python3 -c "import json; m = json.load(open('$CAL/occupancy_model.json')); print(f\"[{m['p0']:.6g},{m['q_occ']:.6g},{m['q_free']:.6g}]\")") \
  || { echo "missing $CAL/occupancy_model.json"; exit 1; }
COMMON="--methods bamos --config configs/isaac.yaml --split calib --stride 5 --limit $LIMIT --steps --override stop.mode=joint --override mapping.occupancy_model=$OCC"  # N = 1 is fixed, so stopping at the declaration (as the test runs do) selects the same threshold
mkdir -p "$OUT"

run () {  # run <tag> <overrides file> <extra args...>
  tag=$1; ovf=$2; shift 2
  [ -f "$ovf" ] || { echo "missing $ovf"; return 1; }
  # shellcheck disable=SC2046,SC2086
  $PY scripts/isaac/run.py $COMMON --out "$OUT/$tag" --experiment-id "stage3_$tag" \
    $(cat "$ovf") "$@" > "$OUT/$tag.log" 2>&1
  echo "$(date +%H:%M) done $tag: $(awk -F, 'NR>1{print $8}' "$OUT/$tag/episodes.csv" | sort | uniq -c | tr '\n' ' ')"
}

case "$WHAT" in
  oracle)
    run oracle        "$CAL/oracle/overrides.txt"        --gpus "$GPUS" --workers 8 ;;
  open_set)
    OS="--backend isaac --override sensor_detector.mode=open_set --override environment.isaac.render_cache_dir=results/cache/isaac"
    # shellcheck disable=SC2086
    (for th in $SAM3_THRESHOLDS; do
      t=$(awk -v t="$th" 'BEGIN{printf "t%03d", t * 100 + 0.5}')
      # shellcheck disable=SC2086
      run "sam3_$t" "$CAL/sam3/overrides_$t.txt" $OS $SAM3_PORTS --workers 6 --gpus 0,1,2,3,4,5
    done) &
    wait ;;
esac
echo "STAGE3_${WHAT}_DONE"
