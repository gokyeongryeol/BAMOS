#!/usr/bin/env bash
# Stage 8 (2026-10-07, after the paper's trial_2 audit): extra experiments on the same calib and test splits.
#   1. The 2x2 made symmetric in its stop rule: each view rule without a target belief (scene coverage, coverage) with
#      BAMOS's belief stop, and BAMOS's IG views with the first-DET stop; oracle and SAM 3.
#   2. Another detector: Falcon Perception (Apache-2.0) through the same calibrate -> fit -> test path as SAM 3
#      (Grounding DINO + SAM 2.1 was planned too and dropped on 2026-10-07); and a false-positive sweep on the oracle (a non-target
#      instance reported as the target in a frame without the target's DET, rate 0.05 or 0.1).
#
#   GROUP=calib    bash scripts/isaac/stage8_extra.sh   # calib frames of the new channels (servers up)
#   GROUP=fit      bash scripts/isaac/stage8_extra.sh   # replay + constants of the new channels
#   GROUP=test_a   bash scripts/isaac/stage8_extra.sh   # analytic test arms (experiment 1 oracle, false-positive sweep)
#   GROUP=test_o   bash scripts/isaac/stage8_extra.sh   # open-set test arms (experiment 1 SAM 3, Falcon)
#
# The test groups read configs/isaac_conditions/ (frozen) and refuse to run on uncommitted configs/, src/ or scripts/.
set -u
PY=${PY:-uv run python}
cd "$(dirname "$0")/../.."
C=${C:-configs/isaac_conditions}
CAL=${CAL:-results/raw/calib}
OUT=${OUT:-results/raw/test}
GROUP=${GROUP:?calib|fit|test_a|test_o}
LIMIT=${LIMIT:-}
EPISODES=${EPISODES:-80}
A_GPUS=${A_GPUS:-6,7}
# isaac:detector pairs; SAM 3 5610-, Falcon 5620-
FALCON_PAIRS=${FALCON_PAIRS:-"5561:5620 5562:5621 5563:5622"}
SAM3_PORTS=${SAM3_PORTS:-"--isaac-ports 5555,5556,5557,5558,5559,5560 --detector-ports 5610,5611,5612,5613,5614,5615"}
FALCON_PORTS=${FALCON_PORTS:-"--isaac-ports 5561,5562,5563 --detector-ports 5620,5621,5622"}
LOG=results/raw/stage8.log
mkdir -p results/raw
echo "$(date +%H:%M) stage 8 $GROUP at $(git rev-parse --short HEAD)" | tee -a "$LOG"

oracle_channel () {  # oracle_channel <name> <gpu> [overrides...]
  name=$1; gpu=$2; shift 2
  if ls "$CAL/$name"/frames_*.csv > /dev/null 2>&1; then echo "skip calib $name"; return; fi
  ov=(); for o in "$@"; do ov+=(--override "$o"); done
  mkdir -p "$CAL/$name"
  $PY scripts/isaac/calibrate.py --channel oracle --out "$CAL/$name" --episodes "$EPISODES" --gpu "$gpu" "${ov[@]}" \
    > "$CAL/$name/shard_0.log" 2>&1
  echo "$(date +%H:%M) calib $name: $(grep -o 'wrote.*' "$CAL/$name/shard_0.log")" | tee -a "$LOG"
}
open_set_channel () {  # open_set_channel <name> "<isaac:detector> ..."
  name=$1; read -ra pairs <<< "$2"; n=${#pairs[@]}
  if ls "$CAL/$name"/frames_*.csv > /dev/null 2>&1; then echo "skip calib $name"; return; fi
  mkdir -p "$CAL/$name"
  for i in "${!pairs[@]}"; do
    IFS=: read -r ip dp <<< "${pairs[$i]}"
    $PY scripts/isaac/calibrate.py --channel open_set --out "$CAL/$name" --episodes "$EPISODES" --isaac-port "$ip" \
      --detector-port "$dp" --shard "$i/$n" > "$CAL/$name/shard_$i.log" 2>&1 &
  done
  wait
  echo "$(date +%H:%M) calib $name: $(cat "$CAL/$name"/shard_*.log | grep -c '^wrote')/$n shards" | tee -a "$LOG"
}
fit () {  # fit <channel> <log name> <args...>
  ch=$1; log=$2; shift 2
  $PY scripts/isaac/fit_constants.py --root "$CAL/$ch" "$@" > "$CAL/$ch/$log" 2>&1
  echo "$(date +%H:%M) fitted $ch ($log): $(grep -c 'wrote' "$CAL/$ch/$log")" | tee -a "$LOG"
}

OCC=$(python3 -c "import json; m = json.load(open('$CAL/occupancy_model.json')); print(f\"[{m['p0']:.6g},{m['q_occ']:.6g},{m['q_free']:.6g}]\")")
OS="--backend isaac --override sensor_detector.mode=open_set --override environment.isaac.render_cache_dir=results/cache/isaac"

case "$GROUP" in
calib)
  oracle_channel s8_fp010 7 sensor_detector.fp_rate=0.1 &
  open_set_channel falcon "$FALCON_PAIRS" &
  wait ;;
fit)
  CUDA_VISIBLE_DEVICES=${FIT_GPU:-7} $PY scripts/isaac/calib_voxel_belief.py --calib "$CAL" --reuse-occupancy \
    --channels s8_fp010 falcon > "$CAL/voxel_belief_stage8.log" 2>&1 \
    || { echo "replay failed"; exit 1; }
  export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
  fit s8_fp010 fit.log --a0 0.0001 &
  fit falcon fit.log --a0 fit --mask-select largest &
  wait ;;
test_a|test_o)
  if ! git diff --quiet HEAD -- configs src scripts || [ -n "$(git ls-files --others --exclude-standard configs src scripts | grep -v '/\.nfs')" ]; then
    echo "configs/, src/ or scripts/ differ from HEAD: commit the frozen state before the test runs"; exit 1
  fi
  COMMON="--methods bamos --config configs/isaac.yaml --steps --override stop.mode=joint $(cat "$C/common.args")${LIMIT:+ --limit $LIMIT}"
  ISAAC="--backend isaac --override environment.isaac.render_cache_dir=results/cache/isaac"
  arm () {  # arm <tag> <condition> <extra args...>, as scripts/isaac/stage5_test.sh
    tag=$1; cond=$2; shift 2
    # a condition frozen later (the new channels) is skipped until it exists; a re-run picks it up
    [ -f "$C/$cond.constants" ] || { echo "no condition $cond yet: skip $tag"; return; }
    if [ -s "$OUT/$tag/episodes.csv" ] && grep -q "^wrote" "$OUT/$tag.log" 2>/dev/null; then echo "skip $tag"; return; fi
    # shellcheck disable=SC2046,SC2086
    $PY scripts/isaac/run.py $COMMON --out "$OUT/$tag" --experiment-id "test_$tag" \
      $(cat "$C/$cond.channel") $(cat "$C/$cond.constants") "$@" > "$OUT/$tag.log" 2>&1
    echo "$(date +%H:%M) done $tag: $(awk -F, 'NR>1{print $8}' "$OUT/$tag/episodes.csv" | sort | uniq -c | tr '\n' ' ')" | tee -a "$LOG"
  }
  REACTIVE="--override planner.utility_mode=coverage --override stop.rule=first_det"
  SCENE="--override planner.utility_mode=scene_coverage --override stop.rule=first_det"
  NOVOX="--override ablation.use_voxel_belief=False"
  # experiment 1: the view rule and the stop rule crossed
  SCENE_B="--override planner.utility_mode=scene_coverage"   # scene coverage views, belief stop
  COV_B="--override planner.utility_mode=coverage"           # coverage views, belief stop
  IG_F="--override stop.rule=first_det"                       # BAMOS's IG views, first-DET stop
  SAM3=$(python3 -c "import json;print(json.load(open('$C/frozen.json'))['sam3_condition'])")
  if [ "$GROUP" = test_a ]; then
    A="--gpus $A_GPUS --workers ${A_WORKERS:-8}"
    arm a_scene_bstop oracle $A $SCENE_B
    arm a_cov_bstop   oracle $A $COV_B
    arm a_ig_fdet     oracle $A $IG_F
    arm s8_fp010_bamos s8_fp010 $A
  else
    sam3_arms () {
      S="$ISAAC $SAM3_PORTS --workers 6 --gpus 0,1,2,3,4,5"
      arm bp_scene_bstop "$SAM3" $S $SCENE_B
      arm bp_cov_bstop   "$SAM3" $S $COV_B
      arm bp_ig_fdet     "$SAM3" $S $IG_F
    }
    det_arms () {  # det_arms <prefix> <condition> <ports> <gpus>
      p=$1; cond=$2; ports=$3; g=$4
      S="$ISAAC $ports --workers 3 --gpus $g"
      arm "${p}_bamos"      "$cond" $S
      arm "${p}_d5_novoxel" "$cond" $S $NOVOX
      arm "${p}_scene"      "$cond" $S $SCENE
      arm "${p}_reactive"   "$cond" $S $REACTIVE
      arm "${p}_d2_nonodet" "$cond" $S --override ablation.use_nodet=False
    }
    sam3_arms &
    det_arms fa falcon "$FALCON_PORTS" 0,1,2 &
    wait
  fi ;;
esac
echo "STAGE8_${GROUP}_DONE" | tee -a "$LOG"
