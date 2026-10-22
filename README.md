# BAMOS — Bayesian Active Mask-based Object Search

BAMOS searches for an object with an explicit posterior over the target's location and size, updated from detections
and non-detections; each view maximizes the mutual information between the detection event and the target. The code
covers the paper's two benchmarks: Isaac Sim warehouse rooms and HSSD ObjectNav (habitat-sim).

## Setup
```bash
uv sync
```
Simulators, data and detectors are installed separately and set by environment variables:
`ISAAC_PY` (Isaac Sim 6.1.0's Python), `SAM3_DIR` / `FALCON_DIR` (SAM 3, Falcon Perception), `HAB_PY` (a habitat-sim
Python) and `OBJECTNAV_DATA` (HSSD ObjectNav episodes and hssd-hab v0.2.4 scenes).

## Isaac Sim
```bash
C=configs/isaac_conditions
uv run python scripts/isaac/run.py --methods bamos --config configs/isaac.yaml --split test --steps \
  --override stop.mode=joint $(cat $C/common.args) $(cat $C/oracle.channel) $(cat $C/oracle.constants) \
  --gpus 0 --workers 4 --out results/raw/test/a_bamos
```
For SAM 3 or Falcon Perception, start the render and detector servers with `scripts/isaac/launch_servers.sh` and pass
`--backend isaac` with the printed ports. The full pipeline runs `scripts/isaac/stage*` in order (calibration, fit,
selection, freeze, test); `configs/isaac_conditions/` holds the frozen constants.

## HSSD ObjectNav
```bash
bash scripts/hssd/make_data_root.sh
bash scripts/hssd/launch_servers.sh start 1 0
uv run python scripts/hssd/run.py --port 5800 --gpu 0 --fit configs/hssd_conditions/oracle --out results/hssd/val/oracle_bamos
bash scripts/hssd/launch_servers.sh stop 1
```
`configs/hssd_conditions/` holds each detector channel's fit on train. `scripts/hssd/` runs the calibration and the val
arms, and `scripts/hssd/summary.py` summarizes them.
