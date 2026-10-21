#!/usr/bin/env bash
# Stage 12 (2026-10-11): BAMOS without the mask-position likelihood (kappa = 1)
# with SAM 3 and Falcon, the frozen conditions, the stage 5 test split (Isaac + detector servers up), two lanes.
set -u
PY=${PY:-uv run python}
cd "$(dirname "$0")/.."
C=${C:-configs/conditions}
OUT=${OUT:-results/raw/test}
SAM3_PORTS=${SAM3_PORTS:-"--isaac-ports 5555,5556,5557,5558 --detector-ports 5610,5611,5612,5613"}
FALCON_PORTS=${FALCON_PORTS:-"--isaac-ports 5559,5560,5561,5562 --detector-ports 5620,5621,5622,5623"}
LOG=results/raw/stage12.log
mkdir -p results/raw
echo "$(date +%H:%M) stage 12 at $(git rev-parse --short HEAD)" | tee -a "$LOG"
if ! git diff --quiet HEAD -- configs src scripts || [ -n "$(git ls-files --others --exclude-standard configs src scripts | grep -v '/\.nfs')" ]; then
  echo "configs/, src/ or scripts/ differ from HEAD: commit the frozen state before the test runs"; exit 1
fi
COMMON="--methods bamos --config configs/normal.yaml --steps --override stop.mode=joint $(cat "$C/common.args")"
ISAAC="--backend isaac --override environment.isaac.render_cache_dir=results/cache/isaac"
arm () {  # arm <tag> <condition> <extra args...>
  tag=$1; cond=$2; shift 2
  if [ -s "$OUT/$tag/episodes.csv" ] && grep -q "^wrote" "$OUT/$tag.log" 2>/dev/null; then echo "skip $tag"; return; fi
  # shellcheck disable=SC2046,SC2086
  $PY scripts/run_experiment.py $COMMON --out "$OUT/$tag" --experiment-id "test_$tag" \
    $(cat "$C/$cond.channel") $(cat "$C/$cond.constants") $ISAAC "$@" --override ablation.use_kappa=False > "$OUT/$tag.log" 2>&1
  echo "$(date +%H:%M) done $tag: $(awk -F, 'NR>1{print $8}' "$OUT/$tag/episodes.csv" | sort | uniq -c | tr '\n' ' ')" | tee -a "$LOG"
}
SAM3=$(python3 -c "import json;print(json.load(open('$C/frozen.json'))['sam3_condition'])")
# shellcheck disable=SC2086
arm bp_d3_nok "$SAM3" $SAM3_PORTS --workers 4 --gpus 0,1,2,3 &
# shellcheck disable=SC2086
arm fa_d3_nok falcon $FALCON_PORTS --workers 4 --gpus 4,5,6,7 &
wait
echo STAGE12_DONE | tee -a "$LOG"
