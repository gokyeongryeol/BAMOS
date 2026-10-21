#!/usr/bin/env bash
# Start the servers an open-set run needs: one Isaac render server per experiment worker and Falcon detector
# servers the workers share (scripts/run_experiment.py --backend isaac --detector-ports ...).
#
#   bash scripts/launch_open_set.sh start 4 0,1,2,3 2 4,5   # 4 Isaac servers on GPUs 0-3, 2 Falcon on GPUs 4,5
#   bash scripts/launch_open_set.sh stop 4 2
#
# Isaac ports are $ISAAC_BASE.. (5555), Falcon ports $FALCON_BASE.. (5600); set the bases to run a second set of
# servers beside the first. Prints the two port lists for run_experiment.py.
set -u
# PY: the Python that runs the experiment code (default: this repo's uv venv)
PY=${PY:-uv run python}
cd "$(dirname "$0")/.."
CMD=${1:?start|stop}; NI=${2:-4}
LOG=results/server_logs
# FALCON_DIR, SAM3_DIR: checkouts of the detectors, each with its own environment; ISAAC_PY: Isaac Sim's Python
IB=${ISAAC_BASE:-5555}; FB=${FALCON_BASE:-5600}
# DETECTOR=falcon (default) | sam3: which open-set model the detector servers run
# FALCON_PY: run Falcon with this Python instead of `uv run python`
DETECTOR=${DETECTOR:-falcon}
ROOT=$(pwd)
if [ "$CMD" = stop ]; then
  NF=${3:-2}
  $PY - "$NI" "$NF" "$IB" "$FB" <<'EOF' > /dev/null 2>&1
import sys
from bamos.environment.isaac_client import IsaacClient
ni, nf, ib, fb = (int(a) for a in sys.argv[1:5])
for p in [ib + i for i in range(ni)] + [fb + i for i in range(nf)]:
    try:
        IsaacClient("127.0.0.1", p, timeout=30).call({"cmd": "shutdown"})
    except Exception:
        pass
EOF
  echo STOPPED; exit 0
fi
IGPUS=${3:-0,1,2,3}; NF=${4:-2}; FGPUS=${5:-4,5}
[ -n "${ISAAC_PY:-}" ] || { echo "set ISAAC_PY to the Python of Isaac Sim"; exit 1; }
if [ "$NF" -gt 0 ]; then
  if [ "$DETECTOR" = sam3 ]; then [ -n "${SAM3_DIR:-}" ] || { echo "set SAM3_DIR"; exit 1; }
  else [ -n "${FALCON_DIR:-}" ] || { echo "set FALCON_DIR"; exit 1; }; fi
fi
IFS=, read -ra IG <<< "$IGPUS"; IFS=, read -ra FG <<< "$FGPUS"
mkdir -p "$LOG"
iports=(); fports=()
for ((i = 0; i < NI; i++)); do
  p=$((IB + i)); g=${IG[$((i % ${#IG[@]}))]}; iports+=("$p")
  : > "$LOG/isaac_$p.log"  # truncate first: a stale "ready on" line must not pass the wait below
  (unset CUDA_VISIBLE_DEVICES; ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES nohup "$ISAAC_PY" \
     isaac_server/server.py --port "$p" --gpu "$g" > "$LOG/isaac_$p.log" 2>&1 &)
done
for ((i = 0; i < NF; i++)); do
  p=$((FB + i)); g=${FG[$((i % ${#FG[@]}))]}; fports+=("$p")
  : > "$LOG/detector_$p.log"
  # the redirection sits on the subshell so no descendant keeps this script's stdout open
  if [ "$DETECTOR" = sam3 ]; then
    (cd "$SAM3_DIR" && HF_HUB_OFFLINE=1 exec nohup ${SAM3_PY:-.venv/bin/python} \
       "$ROOT/detector_server/sam3_server.py" --port "$p" --device "cuda:$g") > "$LOG/detector_$p.log" 2>&1 < /dev/null &
  else
    # shellcheck disable=SC2086
    (cd "$FALCON_DIR" && exec nohup ${FALCON_PY:-uv run python} \
       "$ROOT/detector_server/falcon_server.py" --port "$p" --device "cuda:$g") > "$LOG/detector_$p.log" 2>&1 < /dev/null &
  fi
done
for p in "${iports[@]}"; do until grep -q "ready on" "$LOG/isaac_$p.log" 2>/dev/null; do sleep 3; done; done
for p in "${fports[@]}"; do until grep -q "ready on" "$LOG/detector_$p.log" 2>/dev/null; do
  if grep -q "Traceback" "$LOG/detector_$p.log" 2>/dev/null; then echo "detector server $p failed: $LOG/detector_$p.log"; exit 1; fi
  sleep 3; done; done
echo "--isaac-ports $(IFS=,; echo "${iports[*]}") --detector-ports $(IFS=,; echo "${fports[*]}")"
