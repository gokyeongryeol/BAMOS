#!/usr/bin/env bash
# Re-run stage 5 (README.md, "Isaac Sim pipeline"): every test arm on the solvable test episodes, from the frozen
# conditions of stage 4 (configs/conditions/). All arms run with the stop rule acting (stop.mode=joint): an episode
# ends at its declaration, the present view that confirms S_hat = 1 (formulation §10).
#
#   bash scripts/rerun_stage5_test.sh            # all groups side by side
#   GROUP=analytic bash scripts/rerun_stage5_test.sh   # analytic | sam3
#   DRY_RUN=1 bash scripts/rerun_stage5_test.sh        # print every command, run nothing
#
# It refuses to start unless configs/conditions/frozen.json exists and configs/, src/ and scripts/ have no
# uncommitted change, so the test split only ever sees a committed, frozen configuration.
set -u
# PY: the Python that runs the experiment code (default: this repo's uv venv)
PY=${PY:-uv run python}
cd "$(dirname "$0")/.."
C=${C:-configs/conditions}
OUT=${OUT:-results/raw/test}
GROUP=${GROUP:-all}
LIMIT=${LIMIT:-}  # set only for a smoke test: the first LIMIT test episodes
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
SAM3_PORTS=${SAM3_PORTS:-"--isaac-ports 5555,5556,5557,5558,5559,5560 --detector-ports 5610,5611,5612,5613,5614,5615"}

[ -f "$C/frozen.json" ] || { echo "no $C/frozen.json: run stage 4 first"; exit 1; }
# (.nfs* are the NFS client's placeholders for files deleted while open, not part of the tree)
if ! git diff --quiet HEAD -- configs src scripts || [ -n "$(git ls-files --others --exclude-standard configs src scripts | grep -v '/\.nfs')" ]; then
  echo "configs/, src/ or scripts/ differ from HEAD: commit the frozen state before the test runs"; exit 1
fi
SAM3=$(python3 -c "import json;print(json.load(open('$C/frozen.json'))['sam3_condition'])")
N=$(python3 -c "import json;print(json.load(open('$C/frozen.json'))['confirmations'])")
COMMON="--methods bamos --config configs/normal.yaml --steps --override stop.mode=joint $(cat "$C/common.args")${LIMIT:+ --limit $LIMIT}"
ISAAC="--backend isaac --override environment.isaac.render_cache_dir=results/cache/isaac"
mkdir -p "$OUT"
echo "$(date +%H:%M) stage 5 at $(git rev-parse --short HEAD): SAM 3 condition $SAM3, N = $N" | tee -a "$OUT/stage5.log"

arm () {  # arm <tag> <condition> <constants: fit|spec> <extra args...>
  tag=$1; cond=$2; consts=$3; shift 3
  if [ -s "$OUT/$tag/episodes.csv" ] && grep -q "^wrote" "$OUT/$tag.log" 2>/dev/null; then echo "skip $tag"; return; fi
  cst=""; [ "$consts" = fit ] && cst=$(cat "$C/$cond.constants")
  if [ -n "${DRY_RUN:-}" ]; then  # print the command only
    echo "$tag: run_experiment.py $COMMON --out $OUT/$tag $(cat "$C/$cond.channel") $cst $*"; return
  fi
  # shellcheck disable=SC2046,SC2086
  $PY scripts/run_experiment.py $COMMON --out "$OUT/$tag" --experiment-id "test_$tag" \
    $(cat "$C/$cond.channel") $cst "$@" > "$OUT/$tag.log" 2>&1
  echo "$(date +%H:%M) done $tag: $(awk -F, 'NR>1{print $8}' "$OUT/$tag/episodes.csv" | sort | uniq -c | tr '\n' ' ')" | tee -a "$OUT/stage5.log"
}

# 2026-10-06 re-run: the 2x2 of a scene model (the occupancy belief) and a target belief, and the model ablations. Arms that read no target belief choose views without it and declare at the first robot DET
# (stop.rule=first_det, the cell nearest the mask centroid).
REACTIVE="--override planner.utility_mode=coverage --override stop.rule=first_det"     # neither model
SCENE="--override planner.utility_mode=scene_coverage --override stop.rule=first_det"   # scene model only
NOVOX="--override ablation.use_voxel_belief=False"                                       # target belief only

analytic () {
  A="--gpus $GPUS --workers ${A_WORKERS:-16}"
  # A: oracle, fitted constants
  arm a_bamos       oracle fit $A
  arm a_d5_novoxel  oracle fit $A $NOVOX
  arm a_reactive    oracle fit $A $REACTIVE
  arm a_scene       oracle fit $A $SCENE
  arm a_d2_nonodet  oracle fit $A --override ablation.use_nodet=False
  arm a_d3_nok      oracle fit $A --override ablation.use_kappa=False
}

sam3 () {
  # shellcheck disable=SC2086
  S="$ISAAC $SAM3_PORTS --workers ${S_WORKERS:-6} --gpus 0,1,2,3,4,5"
  arm bp_bamos      "$SAM3" fit $S
  arm bp_d5_novoxel "$SAM3" fit $S $NOVOX
  arm bp_reactive   "$SAM3" fit $S $REACTIVE
  arm bp_scene      "$SAM3" fit $S $SCENE
  arm bp_d2_nonodet "$SAM3" fit $S --override ablation.use_nodet=False
}

case "$GROUP" in
  analytic) analytic ;;
  sam3) sam3 ;;
  all) analytic; sam3 ;;   # the analytic arms take all eight GPUs first, then the open-set arms GPUs 0-5
esac
echo "STAGE5_${GROUP}_DONE" | tee -a "$OUT/stage5.log"
