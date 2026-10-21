#!/usr/bin/env bash
# Start / stop the HSSD teleport render servers (habitat_server/teleport_server.py, in the habitat env), one per worker of
# scripts/run_hssd.py.
#
#   bash scripts/launch_hssd.sh start 4 0,1,2,3      # 4 servers on GPUs 0-3, ports $HSSD_BASE.. (5800)
#   bash scripts/launch_hssd.sh stop 4
#   # then one worker per server, each its shard of the episodes:
#   for i in 0 1 2 3; do uv run python scripts/run_hssd.py --port $((5800 + i)) --gpu $i --shard $i/4 --out results/hssd/run & done
#
# Servers are stopped with their shutdown command (not by process name).
set -u
cd "$(dirname "$0")/.."
CMD=${1:?start|stop}; N=${2:-4}
HAB_PY=${HAB_PY:-}  # the Python of a habitat-sim environment (start only)
PY=${PY:-.venv/bin/python}
BASE=${HSSD_BASE:-5800}
# a data root whose data/scene_datasets/hssd-hab is the v0.2.4 assets (scripts/make_hssd_root.sh makes it)
DATA_ROOT=${HSSD_DATA_ROOT:-results/cache/hssd_v024/root}
LOG=results/server_logs
if [ "$CMD" = stop ]; then
  $PY - "$N" "$BASE" <<'EOF'
import sys
from bamos.environment.isaac_client import IsaacClient
n, base = (int(a) for a in sys.argv[1:3])
for p in range(base, base + n):
    try:
        IsaacClient("127.0.0.1", p, timeout=30).call({"cmd": "shutdown"})
        print("stopped", p)
    except Exception as e:
        print("not running", p, e)
EOF
  exit 0
fi
[ -n "$HAB_PY" ] || { echo "set HAB_PY to the Python of a habitat-sim environment"; exit 1; }
GPUS=${3:-0,1,2,3}
IFS=, read -ra G <<< "$GPUS"
mkdir -p "$LOG"
for ((i = 0; i < N; i++)); do
  port=$((BASE + i)); gpu=${G[$((i % ${#G[@]}))]}
  rm -f "$LOG/teleport_$port.log"
  CUDA_VISIBLE_DEVICES=$gpu nohup "$HAB_PY" habitat_server/teleport_server.py --port "$port" --data-root "$(realpath "$DATA_ROOT")" ${HSSD_SERVER_ARGS:-} > "$LOG/teleport_$port.log" 2>&1 &
  echo "server $port on GPU $gpu (pid $!)"
done
for ((i = 0; i < N; i++)); do
  port=$((BASE + i))
  until grep -q "listening on" "$LOG/teleport_$port.log" 2>/dev/null; do
    if grep -q -E "Traceback|Error" "$LOG/teleport_$port.log" 2>/dev/null; then echo "server $port failed: $LOG/teleport_$port.log"; exit 1; fi
    sleep 2
  done
done
echo "READY ports $BASE..$((BASE + N - 1))"
