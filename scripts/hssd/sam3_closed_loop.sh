#!/usr/bin/env bash
# SAM 3's train closed-loop runs on the calibration episodes, one per score threshold named (Isaac's choice of the
# threshold; scripts/hssd/select_threshold.py picks it once all five exist). Train teleport servers <base>..<base+n-1>.
#   bash scripts/hssd/sam3_closed_loop.sh <base port> <n> <detector ports> <t>...    e.g. 5910 7 5610,5611 050 060
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-.venv/bin/python}
export PY PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
base=$1 n=$2 ports=$3
shift 3
IFS=, read -ra DP <<< "$ports"
E=$($PY - <<'EOF'
import glob, json
e = []
for p in sorted(glob.glob("results/hssd/calib/oracle/episodes_*.json")):
    e += json.load(open(p))
print(",".join(":".join(x.rsplit("_", 1)) for x in e))
EOF
)
for t in "$@"; do
  O=results/hssd/calib_runs/sam3_t$t
  mkdir -p $O
  for ((i = 0; i < n; i++)); do
    $PY scripts/hssd/run.py --split train --port $((base + i)) --gpu $((i % 8)) --shard $i/$n --episodes "$E" --fit results/hssd/calib/sam3_t$t \
      --detector-port ${DP[$((i % ${#DP[@]}))]} --out $O > $O/run_$i.log 2>&1 &
  done
  wait
  echo "[$(date -u +%H:%M:%S)] sam3_t$t: $(wc -l < $O/episodes.jsonl) rows"
done
