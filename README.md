# BAMOS — Bayesian Active Mask-based Object Search

BAMOS searches for an object with an explicit posterior over the target's location and size, updated from detections
and non-detections; each view maximizes the mutual information between the detection event and the target. The code
covers the paper's two benchmarks: Isaac Sim warehouse rooms and HSSD ObjectNav (habitat-sim).

## Layout
```
src/bamos/                  the method: beliefs, planning, environments, evaluation
configs/                    isaac.yaml, hssd.yaml
  isaac_conditions/         Isaac Sim: frozen constants of each detector condition
  hssd_conditions/          HSSD: each detector channel's fit on train
servers/environment/        render servers: isaac_server.py (Isaac Sim), hssd_server.py (habitat-sim)
servers/detector/           open-vocabulary detector servers: sam3_server.py, falcon_server.py
scripts/isaac/              Isaac Sim: run.py, launch_servers.sh, stage*.sh pipeline, analyze_stage*.py
scripts/hssd/               HSSD: run.py, launch_servers.sh, calibration, arm runners, summary.py
scripts/figures/            the paper's figures: method_loop.py, anytime_both.py
```
Each server runs in its own Python environment; `scripts/*/launch_servers.sh` start and stop them.

## Setup
```bash
uv sync
```
Simulators, data and detectors are installed separately and set by environment variables:
`ISAAC_PY` (Isaac Sim 6.1.0's Python), `SAM3_DIR` / `FALCON_DIR` (SAM 3, Falcon Perception), `HAB_PY` (a habitat-sim
Python) and `OBJECTNAV_DATA` (HSSD ObjectNav episodes and hssd-hab v0.2.4 scenes).

## Isaac Sim
Oracle detector, analytic renderer:
```bash
C=configs/isaac_conditions
uv run python scripts/isaac/run.py --methods bamos --config configs/isaac.yaml --split test --steps \
  --override stop.mode=joint $(cat $C/common.args) $(cat $C/oracle.channel) $(cat $C/oracle.constants) \
  --gpus 0 --workers 4 --out results/raw/test/a_bamos
```
SAM 3 or Falcon Perception: start the render and detector servers, then run with `--backend isaac`:
```bash
DETECTOR=sam3 bash scripts/isaac/launch_servers.sh start 1 0 1 1    # 1 Isaac server on GPU 0, 1 SAM 3 server on GPU 1
uv run python scripts/isaac/run.py --methods bamos --config configs/isaac.yaml --split test --steps \
  --override stop.mode=joint $(cat $C/common.args) $(cat $C/sam3_t050.channel) $(cat $C/sam3_t050.constants) \
  --backend isaac --isaac-ports 5555 --detector-ports 5600 --gpus 0 --workers 1 --out results/raw/test/bp_bamos
bash scripts/isaac/launch_servers.sh stop 1 1
```
The paper's arms come from `scripts/isaac/stage*` in order: calibration, fit, threshold selection and freeze
(stages 1-4, already frozen in `configs/isaac_conditions/`), then the test arms (stages 5-13). `analyze_stage5.py`,
`analyze_stage8.py`, `analyze_stage9_10.py` and `analyze_stage13.py` print the tables; `analyze_stage5.py` also reads
the episode features that `episode_metadata.py --split test` writes.

## HSSD ObjectNav
```bash
bash scripts/hssd/make_data_root.sh
bash scripts/hssd/launch_servers.sh start 1 0                       # 1 teleport server on GPU 0, port 5800
uv run python scripts/hssd/run.py --port 5800 --gpu 0 --fit configs/hssd_conditions/oracle \
  --out results/hssd/val/oracle_bamos
bash scripts/hssd/launch_servers.sh stop 1
```
The calibration on train is `calibrate.py`, `fit_constants.py`, `openset_calib.sh` and `sam3_closed_loop.sh` with
`select_threshold.py`. The val arms are `oracle_arms.sh`, `openset_arms.sh` and `gt_arm.sh`; set
`HSSD_CALIB=configs/hssd_conditions` to run them with the shipped fits. `summary.py` prints the results.

## Figures
`scripts/figures/method_loop.py` and `scripts/figures/anytime_both.py` write the paper's figures to `results/figures/`.
`scripts/isaac/render_strip.py` and `scripts/hssd/render_strip.py` render an episode's views and beliefs.
