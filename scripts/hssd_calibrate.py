"""HSSD stage 1 and 2a: calibration frames on train scenes, the voxel belief's model, and every frame's model quantities.

    uv run python scripts/hssd_calibrate.py select --out results/hssd/calib --n-scenes 40 --seed 0   # writes scenes.txt
    # a teleport server on the train split, those scenes only:
    CUDA_VISIBLE_DEVICES=1 <habitat python> habitat_server/teleport_server.py --port 5810 --split train \\
        --scenes $(cat results/hssd/calib/scenes.txt)
    uv run python scripts/hssd_calibrate.py collect --out results/hssd/calib --port 5810 [--shard i/n]
    uv run python scripts/hssd_calibrate.py occupancy --out results/hssd/calib --port 5810
    uv run python scripts/hssd_calibrate.py replay --out results/hssd/calib --port 5810
    uv run python scripts/hssd_fit_constants.py --root results/hssd/calib/oracle

The constants are fitted on train and evaluated on val. As the Isaac stage 1 and 2a
(scripts/calibrate_channel.py, scripts/calib_voxel_belief.py), on the HSSD world of planning/hssd_runner.py:

- select: n_scenes train scenes (seeded); in each, one episode per category (seeded).
- collect: per episode, the start view and `poses` robot poses of the teleport pose pool, stratified over the goals'
  apparent area (calibrate_channel.stratified: a sixth out of view, the rest spread over log-area; the area of a pose is
  the largest s_g^2 fx fy / (W H z^2) over the goal instances whose centre is in view within the depth range, occlusion
  not counted), in a random order after the start view. Each frame through the oracle detector exactly as the planner
  receives it. oracle/frames_<shard>.csv: the status, the reported (largest) goal instance, the world centroid of its
  valid-depth pixels, the camera position; oracle/arrays/<episode>.npz: the depth, the reported mask (the union of the
  goal instances detected, as the planner's voxel belief update leaves out) and the near-clipped pixels, every
  occupancy_pixel_stride-th pixel; oracle/episodes.json: each episode's goals and workspace.
- occupancy: the Bernoulli voxel belief's p0, q_occ, q_free by maximum likelihood (calib_voxel_belief.fit_binomial_mixture)
  over the voxels not in the known shape (those are opaque and fixed in the planner), from each episode's frames in
  order. Writes occupancy_model.json.
- replay: the frames in order with that model on the known-shape voxel belief; before each frame, for every goal instance
  the candidate cells represent (its box within evaluation.max_center_error_m of its nearest cell v*_g), at (v*_g, s*_g)
  (s*_g the category's size hypothesis nearest the instance's box size scale): the apparent area a and the front-face
  transmittance T at each of the cell's points. oracle/model_area.csv. Which instance is a frame's true hypothesis is
  decided in the fit (rule a: the reported instance of a DET, the most detectable one of a NO_DET).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# the train episodes, under the data root scripts/make_hssd_root.sh makes (HSSD_DATA_ROOT, as scripts/launch_hssd.sh)
TRAIN_CONTENT = Path(os.environ.get("HSSD_DATA_ROOT", "results/cache/hssd_v024/root")) / "data/datasets/objectnav/hssd-hab/train/content"
FIELDS = ["episode_id", "scene", "hssd_episode", "category", "pose_id", "frame", "status", "det_goal", "n_det_instances",
          "goal_px", "centroid", "cam_pos", "oracle_det", "instances", "centroids"]


def size_scale(lo, hi) -> float:
    """sqrt(surface area / 4) of a box: the size scale of formulation §3 (as the size hypotheses were made)."""
    a, b, c = np.asarray(hi, dtype=float) - np.asarray(lo, dtype=float)
    return float(np.sqrt((a * b + b * c + c * a) / 2.0))


def setup(cfg_path: str, port: int, gpu: int, overrides: list | None = None):
    import torch

    from bamos.config import load_config
    from bamos.environment.habitat_env import TeleportClient
    from bamos.planning.runner import camera_from_cfg

    torch.cuda.set_device(gpu)
    cfg = load_config(cfg_path, overrides or [])
    cfg["experiment"]["device"] = f"cuda:{gpu}"
    return cfg, TeleportClient(cfg["hssd"]["host"], port), camera_from_cfg(cfg["cameras"]["robot"])


def runner_for(env, cfg):
    from bamos.planning.hssd_runner import HssdEpisodeRunner
    from bamos.planning.runner import EpisodeContext

    return HssdEpisodeRunner(env, cfg, EpisodeContext("calib", "calib", env.scene_name, 0, 0, "bamos", 0, 0, "calib"))


def cmd_select(a) -> int:
    rng = np.random.default_rng(a.seed)
    scenes = sorted(p.name.split(".")[0] for p in TRAIN_CONTENT.glob("*.json.gz"))
    pick = sorted(rng.choice(scenes, size=min(a.n_scenes, len(scenes)), replace=False).tolist())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "scenes.txt").write_text(",".join(pick) + "\n")
    print(f"{len(pick)} of {len(scenes)} train scenes -> {out / 'scenes.txt'}")
    return 0


def episode_list(client, seed: int) -> list[dict]:
    """One episode per (scene, category) of the loaded scenes, chosen with the seed."""
    by: dict = {}
    for e in client.episodes():
        by.setdefault((e["scene"], e["category"]), []).append(e)
    rng = np.random.default_rng(seed)
    return [sorted(v, key=lambda e: int(e["episode_id"]))[int(rng.integers(len(v)))] for _, v in sorted(by.items())]


def cmd_collect(a) -> int:
    from bamos.environment.habitat_env import HabitatEnvironment
    from bamos.mapping.camera import inverse_transform
    from calibrate_channel import stratified

    from run_hssd import parse_value

    ovs = []
    for o in a.override:
        k, v = o.split("=", 1)
        d: dict = {}
        cur = d
        parts = k.split(".")
        for q in parts[:-1]:
            cur = cur.setdefault(q, {})
        cur[parts[-1]] = parse_value(v)
        ovs.append(d)
    cfg, client, cam = setup(a.config, a.port, a.gpu, ovs)
    detector = None
    if cfg["sensor_detector"]["mode"] == "open_set":
        from bamos.environment.open_set import DetectorClient

        detector = DetectorClient(cfg["hssd"]["host"], a.detector_port, cache_dir=cfg["sensor_detector"]["open_set_cache_dir"])
    st = int(cfg["mapping"]["occupancy_pixel_stride"])
    k, n = (int(x) for x in a.shard.split("/"))
    eps = episode_list(client, a.seed)[k::n]
    out = Path(a.out) / a.channel  # the channel's frames; the poses are the oracle's (same seed), so is its replay
    (out / "arrays").mkdir(parents=True, exist_ok=True)
    (out / "overrides.json").write_text(json.dumps(a.override))
    meta_path = out / f"episodes_{k}.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    rows = list(csv.DictReader(open(out / f"frames_{k}.csv"))) if (out / f"frames_{k}.csv").exists() else []
    print(f"[collect] {len(eps)} episodes (shard {k}/{n}), {len(meta)} already done", flush=True)
    for e in eps:
        ep_id = f"{e['scene']}_{e['episode_id']}"
        if ep_id in meta:
            continue
        t0 = time.time()
        try:
            env = HabitatEnvironment(client, cfg, e["scene"], e["episode_id"], cam)
            env.open_set_client = detector
            env.open_set_record_instances = detector is not None
        except RuntimeError as err:  # e.g. no goal object in the scene: the episode is skipped, logged
            print(f"[collect] {ep_id}: skipped ({str(err)[:160]})", flush=True)
            continue
        goals = env.ground_truth().goals
        rng = np.random.default_rng([a.seed, int(e["episode_id"]), len(ep_id)])
        # the apparent-area proxy of every pose: the largest s_g^2 fx fy / (W H z^2) of the goal centres in view
        c = np.stack([g["center"] for g in goals])
        s2 = np.array([size_scale(g["lo"], g["hi"]) ** 2 for g in goals])
        Ti = np.stack([inverse_transform(t) for t in env.pool.world_from_camera])
        pc = np.einsum("pij,nj->pni", Ti[:, :3, :3], c) + Ti[:, None, :3, 3]
        z = pc[..., 2]
        zz = np.maximum(z, 1e-6)
        u, v = cam.fx * pc[..., 0] / zz + cam.cx, cam.fy * pc[..., 1] / zz + cam.cy
        inview = (z >= cam.near) & (z <= cam.far) & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height)
        area = np.where(inview, s2[None] * cam.fx * cam.fy / (cam.n_pixels * zz ** 2), 0.0).max(axis=1)
        pick = stratified(area, a.poses, rng)
        pick = pick[pick != env.robot_start_pose_id]
        order = [env.robot_start_pose_id] + rng.permutation(pick).tolist()
        D, M, IG = [], [], []
        for i, pid in enumerate(order):
            o = env.observe_robot(int(pid), None, i)
            det = o.detection.value == "DET"
            q = env.gt_quality_of_observation(o)
            sel = None
            m = (o.instance_masks[0] if o.instance_masks else o.target_mask) if det else None
            if det:  # the reported instance: the goal holding most of a reported mask (the oracle's largest goal; with
                # several masks, multi-mask, the first in the report's order that lies on a goal); none ("") when no
                # mask has half its pixels on goals: a false positive
                sel = ""
                for mk in (o.instance_masks if o.instance_masks else [m]):
                    under = env.last_ids[mk]
                    gs = [(int(g), int((under == g).sum())) for g in env.goal_ids]
                    g_best, n_best = max(gs, key=lambda t: t[1]) if gs else (0, 0)
                    if n_best >= 0.5 * max(int(mk.sum()), 1):
                        sel = str(g_best)
                        break
            def centroid_of(mk):  # the world centroid of a mask's valid-depth pixels, as the planner's kappa (>= 3 pixels)
                vs_, us_ = np.nonzero(mk)
                d = o.depth[vs_, us_]
                ok = np.isfinite(d) & (d > 0) & (d <= cam.far)
                if ok.sum() < 3:
                    return None
                p = cam.backproject(np.stack([us_[ok], vs_[ok]], 1).astype(np.float64), d[ok])
                return (p @ o.world_from_camera[:3, :3].T + o.world_from_camera[:3, 3]).mean(axis=0)

            cen = centroid_of(m) if det else None
            # multi-mask: the centroid of every mask the planner's DET update averages kappa over (instance_masks)
            cens = [c_ for c_ in (centroid_of(mk) for mk in o.instance_masks) if c_ is not None] if det and o.instance_masks else []
            rows.append({"episode_id": ep_id, "scene": e["scene"], "hssd_episode": e["episode_id"], "category": e["category"],
                         "pose_id": int(pid), "frame": i, "status": o.detection.value, "det_goal": "" if sel is None else sel,
                         "n_det_instances": o.meta["n_goal_instances_detected"], "goal_px": json.dumps(q["goal_px"]),
                         "centroid": "" if cen is None else json.dumps([round(float(x), 4) for x in cen]),
                         "cam_pos": json.dumps([round(float(x), 4) for x in o.world_from_camera[:3, 3]]),
                         "oracle_det": bool(q["gt_detected"]),
                         "instances": json.dumps(o.meta["noise_applied"].get("instances")) if detector is not None else "",
                         "centroids": json.dumps([[round(float(x), 4) for x in c_] for c_ in cens]) if cens else ""})
            D.append(np.nan_to_num(o.depth[::st, ::st], nan=np.inf).astype(np.float32))
            M.append((o.target_mask[::st, ::st] if det else np.zeros_like(o.depth[::st, ::st], dtype=bool)))
            IG.append(np.asarray(o.meta["depth_ignore"])[::st, ::st])
        np.savez_compressed(out / "arrays" / f"{ep_id}.npz", depth=np.stack(D), mask=np.stack(M), ignore=np.stack(IG),
                            pose_id=np.asarray(order), stride=st)
        meta[ep_id] = {"scene": e["scene"], "hssd_episode": e["episode_id"], "category": e["category"],
                       "workspace_lo": list(env.scene.workspace_lo), "workspace_hi": list(env.scene.workspace_hi),
                       "goals": [{"semantic_id": g["semantic_id"], "lo": g["lo"].tolist(), "hi": g["hi"].tolist()} for g in goals]}
        with open(out / f"frames_{k}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        meta_path.write_text(json.dumps(meta))
        nd = sum(r["status"] == "DET" for r in rows if r["episode_id"] == ep_id)
        print(f"[collect] {ep_id} {e['category']}: {len(order)} frames, {nd} DET ({time.time() - t0:.1f} s)", flush=True)
    return 0


def frames(out: Path, ep_id: str, env):
    z = np.load(out / "arrays" / f"{ep_id}.npz")
    for i in range(len(z["pose_id"])):
        pid = int(z["pose_id"][i])
        yield i, pid, env.pool.world_from_camera[pid], z["depth"][i], z["mask"][i], z["ignore"][i]


def episodes_meta(out: Path) -> dict:
    meta = {}
    for p in sorted(out.glob("episodes_*.json")):
        meta.update(json.loads(p.read_text()))
    return meta


def cmd_occupancy(a) -> int:
    import torch

    from bamos.belief.voxel_belief import VoxelBelief
    from bamos.environment.habitat_env import HabitatEnvironment
    from calib_voxel_belief import fit_binomial_mixture

    cfg, client, cam = setup(a.config, a.port, a.gpu)
    out = Path(a.out) / "oracle"
    meta = episodes_meta(out)
    pairs: dict = {}
    for ep_id, m in meta.items():
        env = HabitatEnvironment(client, cfg, m["scene"], m["hssd_episode"], cam)
        r = runner_for(env, cfg)
        og, known = r.known()
        vb = VoxelBelief(og, 0.5, 0.5, 0.01, cfg["experiment"]["device"], int(cfg["mapping"]["occupancy_pixel_stride"]))
        vb.near_hits = bool(cfg["mapping"]["near_hits"])  # the planner's rule (counts only: the constants do not enter)
        for i, pid, T, d, msk, ig in frames(out, ep_id, env):
            vb.update_subsampled(d, msk if msk.any() else None, cam, T, ig)
        h, n_ = vb.n_hit.cpu().numpy(), (vb.n_hit + vb.n_pass).cpu().numpy()
        sel = (n_ > 0) & ~known
        for key, c in zip(*np.unique(np.stack([h[sel], n_[sel]], 1), axis=0, return_counts=True)):
            pairs[tuple(key)] = pairs.get(tuple(key), 0) + int(c)
        print(f"[occupancy] {ep_id}: {int(sel.sum())} observed voxels outside the known shape", flush=True)
        del vb
        torch.cuda.empty_cache()
    keys = np.array(list(pairs))
    w = np.array(list(pairs.values()))
    p0, q_occ, q_free, ll = fit_binomial_mixture(keys[:, 0], keys[:, 1], w)
    model = {"p0": p0, "q_occ": q_occ, "q_free": q_free, "loglik": ll, "n_voxel_episodes": int(w.sum()),
             "pooled_hit_rate": float((keys[:, 0] * w).sum() / (keys[:, 1] * w).sum()), "n_episodes": len(meta),
             "stride": int(cfg["mapping"]["occupancy_pixel_stride"]), "voxel_m": float(cfg["mapping"]["occupancy_voxel_m"]),
             "unobserved_pass_probability": 1 - p0 * q_occ - (1 - p0) * q_free, "excludes": "the known-shape voxels"}
    (Path(a.out) / "occupancy_model.json").write_text(json.dumps(model, indent=1))
    print(f"voxel belief: p0 {p0:.4g}, q_occ {q_occ:.4g}, q_free {q_free:.4g} (pooled hit rate {model['pooled_hit_rate']:.4g}), "
          f"from {int(w.sum())} voxel-episodes of {len(meta)} episodes", flush=True)
    return 0


def cmd_replay(a) -> int:
    import torch

    from bamos.environment.habitat_env import HabitatEnvironment
    from bamos.evaluation.objectnav_evaluator import box_distance

    cfg, client, cam = setup(a.config, a.port, a.gpu)
    model = json.loads((Path(a.out) / "occupancy_model.json").read_text())
    cfg["mapping"]["occupancy_model"] = [model["p0"], model["q_occ"], model["q_free"]]
    out = Path(a.out) / "oracle"
    meta = episodes_meta(out)
    r_max = float(cfg["evaluation"]["max_center_error_m"])
    rows = []
    for ep_id, m in meta.items():
        t0 = time.time()
        env = HabitatEnvironment(client, cfg, m["scene"], m["hssd_episode"], cam)
        run = runner_for(env, cfg)
        eng, sc, vb, sizes = run.engine, run.scorer, run.voxel_belief, run.sizes
        cc = run.state.candidate_centers()
        goals = env.ground_truth().goals
        lo, hi = np.stack([g["lo"] for g in goals]), np.stack([g["hi"] for g in goals])
        vstar = np.array([int(np.argmin(np.linalg.norm(cc - g["center"][None], axis=1))) for g in goals])
        rep = box_distance(cc[vstar], lo, hi)[np.arange(len(goals)), np.arange(len(goals))] <= r_max
        gi = np.nonzero(rep)[0]
        si = np.array([int(np.argmin(np.abs(sizes - size_scale(g["lo"], g["hi"])))) for g in goals])
        idx = torch.as_tensor(vstar[gi], device=eng.device)
        for i, pid, T, d, msk, ig in frames(out, ep_id, env):
            per = []
            if len(gi):
                A_, T_ = [], []
                for q in range(len(eng.offsets)):
                    phi = eng.candidate_coverage(T, sizes, cam, q, idx)[0]  # [n_g, n_S]
                    ah = sc.projected_area(T, cam, q, idx)[0]
                    tr = eng.transmittance(T[:3, 3][None], sizes, vb, q, idx)[0]
                    A_.append((phi * ah)[torch.arange(len(gi)), torch.as_tensor(si[gi])].cpu().numpy())
                    T_.append(tr[torch.arange(len(gi)), torch.as_tensor(si[gi])].cpu().numpy())
                A_, T_ = np.stack(A_, 1), np.stack(T_, 1)  # [n_g, Q]
                for j, g in enumerate(gi):
                    per.append({"goal": int(goals[g]["semantic_id"]), "s_star": float(sizes[si[g]]),
                                "x_star": [round(float(x), 4) for x in cc[vstar[g]]],
                                "a_cell": [float(f"{x:.6e}") for x in A_[j]], "T_cell": [round(float(x), 6) for x in T_[j]]})
            rows.append({"episode_id": ep_id, "frame": i, "pose_id": pid, "goals": json.dumps(per),
                         "workspace_volume": float(np.prod(np.asarray(m["workspace_hi"]) - np.asarray(m["workspace_lo"]))),
                         "n_goals": len(goals), "n_represented": int(len(gi))})
            vb.update_subsampled(d, msk if msk.any() else None, cam, T, ig)
        print(f"[replay] {ep_id}: {len(gi)}/{len(goals)} goal instances represented ({time.time() - t0:.1f} s)", flush=True)
        del run, vb
        torch.cuda.empty_cache()
    with open(out / "model_area.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out / 'model_area.csv'} ({len(rows)} frames)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["select", "collect", "occupancy", "replay"])
    ap.add_argument("--out", default="results/hssd/calib")
    ap.add_argument("--config", default="configs/hssd.yaml")
    ap.add_argument("--port", type=int, default=5810)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n-scenes", type=int, default=40)
    ap.add_argument("--poses", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--channel", default="oracle", help="collect: the output subdirectory of this detection channel")
    ap.add_argument("--override", action="append", default=[], help="collect: key.path=value of the channel")
    ap.add_argument("--detector-port", type=int, default=0, help="collect: the open-set detector server")
    a = ap.parse_args()
    return {"select": cmd_select, "collect": cmd_collect, "occupancy": cmd_occupancy, "replay": cmd_replay}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
