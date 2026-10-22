"""Batch experiment runner: runs the test-split (or, with --split calib, the calib-split) episodes on the analytic
backend (default) or through Isaac.

    uv run python scripts/isaac/run.py --methods bamos --out results/raw/my_run

Episodes are independent, so they are distributed over worker processes, each pinned to one CUDA device
(the visibility engine and the ray caster both run on GPU). Every row is written to `<out>/episodes.csv`.

The open-set detector needs RGB, hence Isaac: one Isaac server per worker (a server holds one scene) and
Falcon servers that several workers may share (`scripts/isaac/launch_servers.sh` starts both):

    uv run python scripts/isaac/run.py --methods bamos --out results/raw/open_set --backend isaac \
        --isaac-ports 5555,5556,5557,5558 --detector-ports 5600,5601 --workers 4 \
        --override sensor_detector.mode=open_set
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

ROW_FIELDS = [
    "experiment_id", "episode_id", "layout_id", "archetype", "method_id", "planner_seed", "sensor_seed", "status",
    "termination_reason", "declared_stop", "declared_stop_step", "actual_success", "wrong_instance_stop",
    "fixed_initial_view_count", "robot_initial_view_count", "active_robot_view_count", "total_view_count",
    "total_path_length_m", "first_true_detection_observation_step", "first_confidence_step",
    "first_correct_localization_step", "first_detection_source", "initial_visibility_class", "target_type",
    "placement_mode", "runtime_s", "planning_latency_p50_s", "planning_latency_p95_s", "n_model_inconsistency",
    "n_negative_ig_steps", "final_gt_center_error_m", "final_gt_bin_posterior_mass", "final_gt_bin_nll",
    "final_gt_map_center_correct", "final_gt_visible_fraction", "final_gt_area_px", "final_gt_normalized_area",
    "final_gt_truncated", "final_gt_detected", "gt_bin_offset_m", "n_candidates_C",
    # step-derived diagnostics
    "brier_p_plus", "n_det_views", "min_robot_distance_to_gt_m", "final_robot_distance_to_gt_m", "max_observed_area",
    "final_max_pi", "tie_rate_active",
    "n_nodet_views", "nodet_entropy_drop_mean", "det_entropy_drop_mean", "entropy_after_init", "expected_s_final",
    # size belief (spec 3): the MAP size hypothesis and its marginal mass
    "final_map_size", "final_map_size_posterior",
    # realized perception corruption (outline S5); all zero on the oracle channel
    "n_fn_views", "n_fp_views", "n_fp_instance_views",
    # open-set channel: frames the detector did not run on, and true detections (IoU >= open_set_eval_iou)
    "n_missing_views", "n_open_set_hit_views", "n_multi_mask_views",
    # failure analysis (outline §5.3): poses meeting the task quality thresholds, and the final location
    # marginal's mass within the success radius of the GT centre and of the MAP candidate
    "n_reachable_quality_poses", "final_gt_mass_within_tau", "final_map_mass_within_tau", "final_map_prob", "error",
]


class _Capture:
    """Collects StepRecords in memory so the worker can derive per-episode diagnostics."""

    def __init__(self):
        self.rows = []

    def write(self, rec):
        self.rows.append(rec)

_STATE: dict = {}


def _init(config_path: str, overrides: dict, gpus: list[int], counter, backend: str = "analytic",
          isaac_ports: list[int] | None = None, detector_ports: list[int] | None = None):
    with counter.get_lock():
        wid = counter.value
        counter.value += 1
    gpu = gpus[wid % len(gpus)]
    import torch

    from bamos.config import load_config
    from bamos.environment.analytic_renderer import set_cast_backend

    torch.cuda.set_device(gpu)
    set_cast_backend(f"cuda:{gpu}")
    cfg = load_config(config_path, [overrides])
    cfg["experiment"]["device"] = f"cuda:{gpu}"
    cfg["environment"]["backend"] = backend
    _STATE["cfg"] = cfg
    _STATE["gpu"] = gpu
    _STATE["isaac"] = _STATE["detector"] = None
    if backend == "isaac":
        from bamos.environment.isaac_env import IsaacClient

        _STATE["isaac"] = IsaacClient(cfg["environment"]["isaac"]["host"], isaac_ports[wid])
    if detector_ports:
        from bamos.environment.open_set import DetectorClient

        sd = cfg["sensor_detector"]
        _STATE["detector"] = DetectorClient(cfg["environment"]["isaac"]["host"], detector_ports[wid % len(detector_ports)],
                                            cache_dir=sd.get("open_set_cache_dir"))


def _run_one(task):
    spec, method_id, experiment_id, steps_dir = task
    from bamos.config import config_hash
    from bamos.experiment.benchmark import build_env_for_spec, code_commit
    from bamos.planning.runner import EpisodeContext, EpisodeRunner

    cfg = dict(_STATE["cfg"])
    cfg["planner"] = {**cfg["planner"], "method": method_id}
    row = {f: "" for f in ROW_FIELDS}
    row.update(experiment_id=experiment_id, episode_id=spec.episode_id, layout_id=spec.layout_id, archetype=spec.archetype, method_id=method_id, planner_seed=0, sensor_seed=0)
    t0 = time.time()
    try:
        env, _ = build_env_for_spec(spec, cfg, None, _STATE["isaac"], _STATE["detector"])
        ctx = EpisodeContext(experiment_id, spec.episode_id, spec.layout_id, spec.layout_seed, spec.episode_seed, method_id, 0, 0, config_hash(cfg), code_commit())
        cap = _Capture()
        runner = EpisodeRunner(env, cfg, ctx, step_writer=cap)
        res = runner.run()
        if steps_dir:
            from bamos.evaluation.records import JsonlWriter

            w = JsonlWriter(Path(steps_dir) / f"{spec.episode_id}.jsonl")
            for rec in cap.rows:
                w.write(rec)
            w.close()
        import numpy as np

        gt_c = np.asarray(env.ground_truth().target_center, dtype=float)
        rob = [r for r in cap.rows if r.camera_type == "robot"]
        d = [float(np.linalg.norm(np.asarray(r.camera_pose, dtype=float)[:3, 3] - gt_c)) for r in rob]
        pp = np.array([r.pre_observation_detection_probability for r in cap.rows], dtype=float)
        y = np.array([r.detection_status == "DET" for r in cap.rows], dtype=float)
        m = np.isfinite(pp)
        act = [r for r in cap.rows if r.initialization_or_active == "active"]
        row.update(
            brier_p_plus=float(np.mean((pp[m] - y[m]) ** 2)) if m.any() else "",
            n_det_views=int(y.sum()),
            min_robot_distance_to_gt_m=min(d) if d else "",
            final_robot_distance_to_gt_m=d[-1] if d else "",
            max_observed_area=float(max((r.observed_normalized_area for r in cap.rows), default=0.0)),
            final_max_pi=float(cap.rows[-1].posterior_max) if cap.rows else "",
            tie_rate_active=float(np.mean([r.utility_tie_count > 1 for r in act])) if act else "",
            expected_s_final=float(np.sqrt(runner.expected_s2())),
        )
        ps = np.asarray(runner.state.location.probs_size(), dtype=float)
        row.update(final_map_size=float(runner.sizes[int(np.argmax(ps))]), final_map_size_posterior=float(ps.max()))
        # negative-evidence efficiency: entropy removed per NO_DET view (and per DET view for reference)
        H0 = float(np.log(len(runner.candidates)))
        prev, dn, dd = H0, [], []
        for r in cap.rows:
            drop = prev - float(r.posterior_entropy)
            if r.detection_status == "NO_DET":
                dn.append(drop)
            elif r.detection_status == "DET":
                dd.append(drop)
            prev = float(r.posterior_entropy)
        noise = [(r.extra.get("noise_applied") or {}) for r in cap.rows]
        row.update(
            n_fn_views=int(sum(bool(n.get("fn")) for n in noise)),
            n_fp_views=int(sum(bool(n.get("fp")) for n in noise)),
            n_fp_instance_views=int(sum("wrong_instance_id" in n for n in noise)),
            n_missing_views=int(sum(r.detection_status == "MISSING" for r in cap.rows)),
            n_open_set_hit_views=int(sum(bool(n.get("open_set")) and "fp" in n and not n["fp"] for n in noise)),
            # frames where the mask-selection rule had a choice to make
            n_multi_mask_views=int(sum(int(n.get("n_kept", 0)) > 1 for n in noise)),
        )
        # failure analysis: is the truth still among several live hypotheses, or has the mass left it?
        from bamos.planning.stopping import mass_within_radius

        tau = float(cfg["stop"]["success_radius_m"])
        pi_loc = runner.state.location.probs_loc()
        near_gt = np.linalg.norm(runner.state.candidate_centers() - gt_c[None, :], axis=1) <= tau + 1e-9
        row.update(
            n_reachable_quality_poses=int(len(env.ground_truth().reachable_quality_pose_ids)),
            final_gt_mass_within_tau=float(pi_loc[near_gt].sum()),
            final_map_mass_within_tau=mass_within_radius(runner.state, tau),
            final_map_prob=float(runner.state.location.max_prob()),  # the stop rule's pi(v_hat)
        )
        init_rows = [r for r in cap.rows if r.initialization_or_active == "initialization"]
        row.update(
            n_nodet_views=len(dn),
            nodet_entropy_drop_mean=float(np.mean(dn)) if dn else "",
            det_entropy_drop_mean=float(np.mean(dd)) if dd else "",
            entropy_after_init=float(init_rows[-1].posterior_entropy) if init_rows else "",
        )
        row.update(
            status="ok", termination_reason=res.termination_reason, declared_stop=res.declared_stop,
            declared_stop_step=res.declared_stop_step if res.declared_stop_step is not None else "",
            actual_success=res.actual_success, wrong_instance_stop=res.wrong_instance_stop,
            fixed_initial_view_count=res.fixed_initial_view_count, robot_initial_view_count=res.robot_initial_view_count,
            active_robot_view_count=res.active_robot_view_count, total_view_count=res.total_view_count,
            total_path_length_m=res.total_path_length_m,
            first_true_detection_observation_step=res.first_true_detection_observation_step if res.first_true_detection_observation_step is not None else "",
            first_confidence_step=res.first_confidence_step if res.first_confidence_step is not None else "",
            first_correct_localization_step=res.first_correct_localization_step if res.first_correct_localization_step is not None else "",
            first_detection_source=res.first_detection_source or "", initial_visibility_class=res.initial_visibility_class,
            target_type=res.target_type, placement_mode=res.placement_mode, runtime_s=res.runtime_s,
            planning_latency_p50_s=res.planning_latency_p50_s, planning_latency_p95_s=res.planning_latency_p95_s,
            n_model_inconsistency=res.n_model_inconsistency, n_negative_ig_steps=res.n_negative_ig_steps,
            gt_bin_offset_m=res.extra.get("gt_bin_offset_m", ""), n_candidates_C=res.extra.get("n_candidates_C", ""),
        )
        for k, v in res.final_gt.items():
            row[f"final_{k}"] = v
    except Exception:
        row.update(status="error", termination_reason="error", runtime_s=time.time() - t0, error=traceback.format_exc()[-800:].replace("\n", " | "))
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", required=True, help="comma-separated method ids")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/isaac.yaml")
    ap.add_argument("--experiment-id", default=None)
    ap.add_argument("--gpus", default="3,4,5,7")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="run only the first N episode specs (smoke tests)")
    ap.add_argument("--split", choices=["test", "calib", "dev", "dev2"], default="test",
                    help="calib: for choosing settings; dev: the former test split, for development only")
    ap.add_argument("--stride", type=int, default=1, help="every stride-th spec, before --limit (calib: 5)")
    ap.add_argument("--override", action="append", default=[], help="dotted.key=value config override")
    ap.add_argument("--steps", action="store_true", help="also write <out>/steps/<episode_id>.jsonl per episode")
    ap.add_argument("--backend", choices=["analytic", "isaac"], default="analytic")
    ap.add_argument("--isaac-ports", default="", help="comma-separated, one Isaac server per worker (--backend isaac)")
    ap.add_argument("--detector-ports", default="", help="comma-separated open-set detector servers, shared by workers")
    ap.add_argument("--episode-ids", default="", help="comma-separated: run only these episodes (re-running failed ones)")
    args = ap.parse_args()

    from bamos.config import load_config
    from bamos.experiment.benchmark import make_episode_specs

    overrides: dict = {}
    for ov in args.override:
        key, _, val = ov.partition("=")
        node = overrides
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        try:
            node[parts[-1]] = eval(val, {"__builtins__": {}}, {})  # literal values only
        except Exception:
            node[parts[-1]] = val
    cfg = load_config(args.config, [overrides])
    specs = make_episode_specs(cfg, args.split)[:: args.stride]
    if args.limit:
        specs = specs[: args.limit]
    if args.episode_ids:
        want = set(args.episode_ids.split(","))
        specs = [s for s in specs if s.episode_id in want]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    experiment_id = args.experiment_id or f"exp_{int(time.time())}"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    steps_dir = str(out / "steps") if args.steps else None
    if steps_dir:
        Path(steps_dir).mkdir(parents=True, exist_ok=True)
    tasks = [(s, m, experiment_id, steps_dir) for m in methods for s in specs]
    gpus = [int(g) for g in args.gpus.split(",")]
    isaac_ports = [int(p) for p in args.isaac_ports.split(",") if p.strip()]
    detector_ports = [int(p) for p in args.detector_ports.split(",") if p.strip()]
    if args.backend == "isaac" and len(isaac_ports) < args.workers:
        ap.error(f"--backend isaac needs one Isaac port per worker ({args.workers}), got {len(isaac_ports)}")
    if cfg["sensor_detector"]["mode"] == "open_set" and (args.backend != "isaac" or not detector_ports):
        ap.error("sensor_detector.mode=open_set needs --backend isaac and --detector-ports")

    ctxmp = mp.get_context("spawn")
    counter = ctxmp.Value("i", 0)
    t0 = time.time()
    done = 0
    with open(out / "episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        w.writeheader()
        with ctxmp.Pool(args.workers, initializer=_init, initargs=(args.config, overrides, gpus, counter, args.backend, isaac_ports, detector_ports)) as pool:
            for row in pool.imap_unordered(_run_one, tasks, chunksize=1):
                w.writerow(row)
                f.flush()
                done += 1
                if row["status"] != "ok":
                    print(f"[error] {row['method_id']} {row['episode_id']}: {row['error'][:200]}", flush=True)
                if done % 10 == 0 or done == len(tasks):
                    el = time.time() - t0
                    print(f"{done}/{len(tasks)} episodes  {el/60:.1f} min  eta {(el/done)*(len(tasks)-done)/60:.1f} min", flush=True)
    print(f"wrote {out/'episodes.csv'} ({done} rows, {(time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
