#!/usr/bin/env bash
# The HSSD open-set channels' calibration frames and fits (multi-mask, configs/hssd.yaml), on the local machine.
# Needs train teleport servers 5910..5916 and the detector servers (Falcon 5600, 5601; SAM 3 5610, 5611), and the oracle
# replay (calib/oracle/model_area.csv). Then scripts/hssd/sam3_closed_loop.sh per threshold (split across machines) and
# scripts/hssd/select_threshold.py.
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-.venv/bin/python}
C=results/hssd/calib
L=results/hssd/logs
mkdir -p $L
stamp() { echo "[$(date -u +%H:%M:%S)] $*"; }
G=(0 1 2 3 0 1 2)
OS="--override sensor_detector.mode=open_set"
stamp "collect open-set channels"
$PY scripts/hssd/calibrate.py collect --out $C --port 5910 --gpu 0 --shard 0/2 --channel falcon --detector-port 5600 $OS \
  > $L/collect_falcon_0.log 2>&1 &
$PY scripts/hssd/calibrate.py collect --out $C --port 5911 --gpu 1 --shard 1/2 --channel falcon --detector-port 5601 $OS \
  > $L/collect_falcon_1.log 2>&1 &
i=2
for t in 050 060 070 080 090; do
  $PY scripts/hssd/calibrate.py collect --out $C --port $((5910 + i)) --gpu ${G[$i]} --channel sam3_t$t \
    --detector-port $((5610 + i % 2)) $OS --override sensor_detector.open_set_mask_select=score \
    --override sensor_detector.open_set_min_score=0.${t:1:1} > $L/collect_sam3_t$t.log 2>&1 &
  i=$((i + 1))
done
wait
[ -f $C/oracle/model_area.csv ] || { echo "no oracle replay"; exit 1; }
stamp "fits"
for ch in falcon sam3_t050 sam3_t060 sam3_t070 sam3_t080 sam3_t090; do
  OMP_NUM_THREADS=32 $PY scripts/hssd/fit_constants.py --root $C/$ch --a0 fit --boot 20 > $L/fit_$ch.log 2>&1 &
done
wait
for ch in falcon sam3_t050 sam3_t060 sam3_t070 sam3_t080 sam3_t090; do stamp "fit $ch: $(head -n 1 $C/$ch/fitted.yaml)"; done
stamp "OPENSET_CALIB_DONE"
