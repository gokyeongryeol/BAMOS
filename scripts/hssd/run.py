"""Run BAMOS on HSSD ObjectNav (val) episodes with teleport poses against one teleport server.

    # the server (habitat env), one per worker
    CUDA_VISIBLE_DEVICES=1 $HAB_PY hssd_server/server.py --port 5800 --data-root results/cache/hssd_v024/root
    # the episodes: all of val, or a subset
    uv run python scripts/hssd/run.py --port 5800 --out results/hssd/my_run --gpu 0 [--scenes 102816600 --limit 3] [--steps]

One JSON row per episode in <out>/episodes.jsonl (appended; episodes already there are skipped, so a run can be resumed);
with --steps, the step records in <out>/steps/<scene>_<episode>.jsonl. Workers split the episode list with
--shard i/n (episode k goes to shard k mod n).
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def parse_value(v: str):
    """An override value: JSON, else a Python literal (False, 'x'), else the raw string (coverage)."""
    try:
        return json.loads(v)
    except ValueError:
        pass
    try:
        return ast.literal_eval(v)
    except (ValueError, SyntaxError):
        return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/hssd.yaml")
    ap.add_argument("--port", type=int, default=5800)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--scenes", default="", help="comma-separated scene ids (default: all)")
    ap.add_argument("--episodes", default="", help="comma-separated scene:episode_id (default: all)")
    ap.add_argument("--limit", type=int, default=0, help="at most this many episodes per scene (0: all)")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--steps", action="store_true")
    ap.add_argument("--split", default="val", help="the split the teleport server must hold")
    ap.add_argument("--timeout", type=float, default=7200.0, help="the teleport client's socket timeout (s)")
    ap.add_argument("--detector-port", type=int, default=0, help="the open-set detector server (sensor_detector.mode=open_set)")
    ap.add_argument("--override", action="append", default=[], help="key.path=json_value")
    ap.add_argument("--fit", default="", help="a calibration channel directory (overrides.json, fitted.yaml) to run with")
    a = ap.parse_args()

    import torch

    from bamos.config import config_hash, load_config
    from bamos.environment.hssd_env import HabitatEnvironment, TeleportClient
    from bamos.evaluation.records import JsonlWriter, _jsonable
    from bamos.experiment.benchmark import code_commit
    from bamos.planning.hssd_runner import HssdEpisodeRunner
    from bamos.planning.runner import EpisodeContext, camera_from_cfg

    overrides, extra = [], []
    if a.fit:
        # a detection channel's train calibration (scripts/hssd/calibrate.py collect, scripts/hssd/fit_constants.py): its
        # sensor overrides and fitted constants; --override entries come after and take precedence
        import yaml

        fit_dir = Path(a.fit)
        extra = json.loads((fit_dir / "overrides.json").read_text()) if (fit_dir / "overrides.json").exists() else []
        overrides.append(yaml.safe_load((fit_dir / "fitted.yaml").read_text()))
    for o in extra + a.override:
        k, v = o.split("=", 1)
        d: dict = {}
        cur = d
        parts = k.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = parse_value(v)
        overrides.append(d)
    cfg = load_config(a.config, overrides)
    torch.cuda.set_device(a.gpu)
    cfg["experiment"]["device"] = f"cuda:{a.gpu}"
    # a scene's first known-shape query (server-side surface sampling) can take many minutes on a busy machine
    connect = lambda: TeleportClient(cfg["hssd"]["host"], a.port, timeout=float(a.timeout))  # noqa: E731
    client = connect()
    detector = None
    if cfg["sensor_detector"]["mode"] == "open_set":
        from bamos.environment.open_set import DetectorClient

        if not a.detector_port:
            raise SystemExit("sensor_detector.mode=open_set needs --detector-port")
        detector = DetectorClient(cfg["hssd"]["host"], a.detector_port, cache_dir=cfg["sensor_detector"]["open_set_cache_dir"])
    if client.info.get("split") != a.split:  # a train server would silently run train episodes into a val output
        raise SystemExit(f"server on port {a.port} holds the {client.info.get('split')!r} split, not {a.split!r}")
    eps = sorted(client.episodes(), key=lambda e: (e["scene"], int(e["episode_id"])))
    if a.scenes:
        keep = set(a.scenes.split(","))
        eps = [e for e in eps if e["scene"] in keep]
    if a.episodes:
        keep = {tuple(x.split(":")) for x in a.episodes.split(",")}
        eps = [e for e in eps if (e["scene"], e["episode_id"]) in keep]
    if a.limit:
        per: dict = {}
        eps = [e for e in eps if per.setdefault(e["scene"], []).append(e) or len(per[e["scene"]]) <= a.limit]
    i, n = (int(x) for x in a.shard.split("/"))
    eps = eps[i::n]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "episodes.jsonl"
    done = set()
    if rows_path.exists():
        for line in open(rows_path):
            r = json.loads(line)
            done.add((r["scene"], r["episode_id"]))
    (out / "config.json").write_text(json.dumps(_jsonable(cfg), indent=1))
    camera = camera_from_cfg(cfg["cameras"]["robot"])
    chash, commit = config_hash(cfg), code_commit()
    print(f"[hssd_run] {len(eps)} episodes ({len(done)} already done), config {chash}", flush=True)
    for e in eps:
        key = (e["scene"], e["episode_id"])
        if key in done:
            continue
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        row = {"scene": e["scene"], "episode_id": e["episode_id"], "category": e["category"], "config_hash": chash,
               "code_commit": commit}
        try:
            env = HabitatEnvironment(client, cfg, e["scene"], e["episode_id"], camera)
            env.open_set_client = detector
            ctx = EpisodeContext(out.name, f"{e['scene']}_{e['episode_id']}", e["scene"], 0, int(e["episode_id"]), "bamos",
                                 0, 0, chash, commit)
            writer = JsonlWriter(out / "steps" / f"{e['scene']}_{e['episode_id']}.jsonl") if a.steps else None
            runner = HssdEpisodeRunner(env, cfg, ctx, step_writer=writer)
            res = runner.run()
            if writer is not None:
                writer.close()
            row.update({
                "status": "ok", "termination_reason": res.termination_reason, "declared_stop": res.declared_stop,
                "declared_stop_step": res.declared_stop_step, "success": res.actual_success,
                "active_views": res.active_robot_view_count, "total_views": res.total_view_count,
                "path_length_m": res.total_path_length_m, "first_true_detection_step": res.first_true_detection_observation_step,
                "first_correct_localization_step": res.first_correct_localization_step, "runtime_s": res.runtime_s,
                "planning_latency_p50_s": res.planning_latency_p50_s, "planning_latency_p95_s": res.planning_latency_p95_s,
                "n_model_inconsistency": res.n_model_inconsistency, "n_candidates_C": res.extra.get("n_candidates_C"),
                **{f"final_{k}": v for k, v in res.final_gt.items()}, **res.extra,
            })
        except Exception:
            row.update({"status": "error", "error": traceback.format_exc()[-3000:]})
            # a timed-out or failed call leaves the reply stream out of step: a fresh connection for the next episode
            try:
                client.sock.close()
            except OSError:
                pass
            client = connect()
        row["wall_s"] = round(time.time() - t0, 1)
        row["gpu_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        with open(rows_path, "a") as f:
            f.write(json.dumps(_jsonable(row)) + "\n")
        print(f"[hssd_run] {key}: {row.get('status')} {row.get('termination_reason', '')} success {row.get('success')} "
              f"viewpoint {row.get('viewpoint_success')} views {row.get('active_views')} {row['wall_s']} s", flush=True)
        if row["status"] == "error":
            print(row["error"], flush=True)


if __name__ == "__main__":
    main()
