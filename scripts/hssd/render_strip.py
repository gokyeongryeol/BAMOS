"""Every k-th observation of one HSSD episode as a strip: the frame and the target belief after it (top and side).

    uv run python scripts/hssd/render_strip.py 102815835 3 --port 6100 --every 5 --label oracle --fit results/hssd/calib/oracle
        # writes results/figures/hssd_strip_<scene>_<episode>_<label>.png

The panels of scripts/render_hssd_episode_steps.py without its curve and its change panel, one observation per row:
obs 0, every --every-th observation, and the last one (the declaration).

Re-runs the episode with HssdEpisodeRunner on a teleport server (configs/hssd.yaml, optional --fit / --override as in
scripts/hssd/run.py); with --check, the replay is compared against that run's step log (the same poses). Top: pi(v_hat) (the MAP
cell), p_succ (formulation §10) and pi(success cells) (the belief on the cells whose declaration would succeed: centre
within evaluation.max_center_error_m of a goal box) per observation. Then per observation, four panels of pi(v) summed
over sizes:
  frame   the RGB of the pose, the goal instances' visible pixels outlined in yellow (the reported mask in cyan when it
          differs), green border = DET;
  top     log pi summed over heights, on the candidate cells' x-y range (+ the goal boxes); grey = the scene's
          surfaces at the cells' heights, red = goal boxes, orange = the camera, white = the path so far;
  side    log pi summed over the other horizontal axis, height on the vertical axis; grey = the scene's surfaces, the
          camera and its vertical field of view;
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

LOG_RANGE = 1e-6  # the top / side panels show pi from max * LOG_RANGE to max


def find(d, k):
    if isinstance(d, dict):
        if k in d:
            return d[k]
        for v in d.values():
            r = find(v, k)
            if r is not None:
                return r
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("episode")
    ap.add_argument("--config", default="configs/hssd.yaml")
    ap.add_argument("--port", type=int, default=5840)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--fit", default="")
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--detector-port", type=int, default=0)
    ap.add_argument("--check", default="", help="a scripts/hssd/run.py output whose step log the replay must match")
    ap.add_argument("--out", default="")
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    import torch
    import yaml

    from bamos.config import load_config
    from bamos.environment.hssd_env import HssdEnvironment, HssdClient
    from bamos.observation.types import DetectionStatus
    from bamos.planning.hssd_runner import HssdEpisodeRunner
    from bamos.planning.runner import EpisodeContext, camera_from_cfg
    from run import parse_value

    overrides, extra = [], []
    if a.fit:
        fit_dir = Path(a.fit)
        extra = json.loads((fit_dir / "overrides.json").read_text()) if (fit_dir / "overrides.json").exists() else []
        overrides.append(yaml.safe_load((fit_dir / "fitted.yaml").read_text()))
    for o in extra + a.override:
        k, v = o.split("=", 1)
        d: dict = {}
        cur = d
        for p in k.split(".")[:-1]:
            cur = cur.setdefault(p, {})
        cur[k.split(".")[-1]] = parse_value(v)
        overrides.append(d)
    cfg = load_config(a.config, overrides)
    torch.cuda.set_device(a.gpu)
    cfg["experiment"]["device"] = f"cuda:{a.gpu}"
    client = HssdClient(cfg["hssd"]["host"], a.port, timeout=7200.0)
    cam = camera_from_cfg(cfg["cameras"]["robot"])
    env = HssdEnvironment(client, cfg, a.scene, a.episode, cam)
    if cfg["sensor_detector"]["mode"] == "open_set":
        from bamos.environment.open_set import DetectorClient
        env.open_set_client = DetectorClient(cfg["hssd"]["host"], a.detector_port,
                                             cache_dir=cfg["sensor_detector"].get("open_set_cache_dir"))
    goals = env.ground_truth().goals
    gids = np.array([int(g["semantic_id"]) for g in goals])
    glo, ghi = np.stack([g["lo"] for g in goals]), np.stack([g["hi"] for g in goals])

    obs_log, ids_log = [], []
    orb = env.observe_robot

    def observe(pid, rng, k):
        o = orb(pid, rng, k)
        obs_log.append(o)
        ids_log.append(None if env.last_ids is None else np.asarray(env.last_ids).copy())
        return o

    env.observe_robot = observe
    arrays: dict = {}
    steps: list = []

    from dataclasses import asdict

    from bamos.evaluation.records import _jsonable

    class Writer:
        def write(self, rec):
            steps.append(json.loads(json.dumps(_jsonable(asdict(rec) if hasattr(rec, "__dataclass_fields__") else rec))))

        def close(self):
            pass

    ctx = EpisodeContext("render", f"{a.scene}_{a.episode}", a.scene, 0, int(a.episode), "bamos", 0, 0, "x", "x")
    runner = HssdEpisodeRunner(env, cfg, ctx, step_writer=Writer(), array_store=arrays)
    res = runner.run()
    C = runner.state.candidate_centers()
    nS = len(runner.sizes)
    marg = [np.asarray(p, dtype=np.float64).reshape(len(C), nS).sum(axis=1) for p in arrays["pi"]]
    n = min(len(marg), len(obs_log), len(steps))
    print(f"{n} observations, {len(C)} candidate cells, termination {res.termination_reason}")
    if a.check:
        logged = [json.loads(l) for l in open(Path(a.check) / "steps" / f"{a.scene}_{a.episode}.jsonl")]
        same = all(find(x, "selected_pose_id") == find(y, "selected_pose_id") for x, y in zip(steps, logged))
        print(f"replay vs {a.check}: same poses {same} ({len(steps)} vs {len(logged)} steps)")
    pmap = np.array([m.max() for m in marg[:n]])
    psucc = np.array([find(s, "p_succ") or np.nan for s in steps[:n]])
    near = runner.evaluator.near_goal
    pwin = np.array([m[near].sum() for m in marg[:n]])
    fire = next((i for i in range(n) if find(steps[i], "declared_stop")), None)

    # the panels' frame: the candidate cells' range plus the goal boxes, 0.5 m padding
    step_m = float(cfg["mapping"]["voxel_size_m"]) * int(cfg["target"]["center_stride_voxels"])
    lo = np.minimum(C.min(axis=0), glo.min(axis=0)) - 0.5
    hi = np.maximum(C.max(axis=0), ghi.max(axis=0)) + 0.5
    lo[2] = min(lo[2], float(np.nanmin(env.floor_z)) - 0.3)
    side_ax = 0 if (hi[0] - lo[0]) >= (hi[1] - lo[1]) else 1  # the side view runs along the longer horizontal axis
    other = 1 - side_ax
    ix = np.floor((C - lo) / step_m).astype(int)
    shape = np.ceil((hi - lo) / step_m).astype(int) + 1

    def top(p):
        H = np.zeros((shape[0], shape[1]))
        np.add.at(H, (ix[:, 0], ix[:, 1]), p)
        return H

    def side(p):
        H = np.zeros((shape[side_ax], shape[2]))
        np.add.at(H, (ix[:, side_ax], ix[:, 2]), p)
        return H

    K = env.known_points()
    kin = np.all((K >= lo) & (K <= hi), axis=1)
    K = K[kin]
    zc = (C[:, 2].min() - 0.1, C[:, 2].max() + 0.1)
    Ktop = K[(K[:, 2] > zc[0]) & (K[:, 2] < zc[1])][::4]
    Kside = K[::4]
    ext_top = (lo[0], lo[0] + shape[0] * step_m, lo[1], lo[1] + shape[1] * step_m)
    ext_side = (lo[side_ax], lo[side_ax] + shape[side_ax] * step_m, lo[2], lo[2] + shape[2] * step_m)
    half_vfov = np.arctan(cam.height / 2 / cam.fy)

    show = sorted(set(range(0, n, a.every)) | {n - 1})
    rows = len(show)
    fig = plt.figure(figsize=(9.6, 0.55 + 2.75 * rows))
    gs1 = fig.add_gridspec(rows, 3, hspace=0.38, wspace=0.12, top=1 - 0.5 / (0.55 + 2.75 * rows), bottom=0.01)
    outcome = "success" if res.actual_success else ("wrong declaration" if fire is not None else res.termination_reason)
    fig.suptitle(f"{a.label or cfg['sensor_detector']['mode']}: HSSD {a.scene} #{a.episode}, {env.category}; "
                 f"{outcome} after {n - 1} views", fontsize=12, y=1 - 0.12 / (0.55 + 2.75 * rows))
    path = []
    for i in range(n):
        o = obs_log[i]
        T = o.world_from_camera
        p, f = T[:3, 3], T[:3, 2]
        path.append(p)
        if i not in show:
            continue
        r = show.index(i)
        det = o.detection == DetectionStatus.DET
        # frame
        axf = fig.add_subplot(gs1[r, 0])
        axf.imshow(o.rgb[..., :3] if o.rgb is not None else np.zeros((cam.height, cam.width, 3)))
        if ids_log[i] is not None:
            gm = np.isin(ids_log[i], gids)
            if gm.any():
                axf.contour(gm.astype(float), levels=[0.5], colors="yellow", linewidths=0.9)
            if o.target_mask is not None and np.asarray(o.target_mask).any() and \
                    (np.asarray(o.target_mask) != gm).mean() > 0.001:
                axf.contour(np.asarray(o.target_mask).astype(float), levels=[0.5], colors="cyan", linewidths=1.1)
        axf.set_xticks([]); axf.set_yticks([])
        for sp in axf.spines.values():
            sp.set_edgecolor("#2ca02c" if det else "0.4"); sp.set_linewidth(2.5 if det else 0.8)
        info = steps[i]
        tag = "  DECLARE" if i == fire else ("  present view" if find(info, "present_view") else
                                             ("  S_hat = 1" if find(info, "s_hat") else ""))
        pitch = np.degrees(np.arcsin(np.clip(f[2], -1, 1)))
        col = ("#2ca02c" if res.actual_success else "#d62728") if i == fire else "k"
        axf.set_title(f"obs {i} {o.detection.value}{tag}  pitch {pitch:+.0f}°\npi(v_hat) {pmap[i]:.3f}  p_succ {psucc[i]:.3f}  "
                      f"pi(success) {pwin[i]:.2g}", fontsize=8, color=col, weight="bold" if i == fire else "normal")
        # top
        Ht = top(marg[i])
        hm = max(Ht.max(), 1e-300)
        axt = fig.add_subplot(gs1[r, 1])
        axt.scatter(Ktop[:, 0], Ktop[:, 1], s=0.2, c="0.55", zorder=0)
        axt.imshow(np.ma.masked_less(Ht.T, hm * LOG_RANGE), origin="lower", norm=LogNorm(hm * LOG_RANGE, hm), cmap="viridis",
                   extent=ext_top, alpha=0.9, zorder=1, interpolation="nearest")
        for gl, gh in zip(glo, ghi):
            axt.add_patch(Rectangle((gl[0], gl[1]), max(gh[0] - gl[0], 0.05), max(gh[1] - gl[1], 0.05), fill=False, ec="#d62728", lw=1.4, zorder=4))
        P = np.array(path)
        axt.plot(P[:, 0], P[:, 1], "-", color="w", lw=0.6, alpha=0.8, zorder=3)
        axt.arrow(p[0], p[1], f[0] * 0.6, f[1] * 0.6, head_width=0.15, color="#ff7f0e", zorder=5)
        axt.set_xlim(ext_top[0], ext_top[1]); axt.set_ylim(ext_top[2], ext_top[3]); axt.set_aspect("equal")
        axt.set_xticks([]); axt.set_yticks([])
        axt.set_title(f"top: log pi (max {hm:.2g})", fontsize=8)
        # side
        Hs = side(marg[i])
        axs = fig.add_subplot(gs1[r, 2])
        axs.scatter(Kside[:, side_ax], Kside[:, 2], s=0.2, c="0.55", zorder=0)
        hs = max(Hs.max(), 1e-300)
        axs.imshow(np.ma.masked_less(Hs.T, hs * LOG_RANGE), origin="lower", norm=LogNorm(hs * LOG_RANGE, hs), cmap="viridis",
                   extent=ext_side, alpha=0.9, zorder=1, interpolation="nearest", aspect="auto")
        for gl, gh in zip(glo, ghi):
            axs.add_patch(Rectangle((gl[side_ax], gl[2]), max(gh[side_ax] - gl[side_ax], 0.05), gh[2] - gl[2], fill=False, ec="#d62728", lw=1.4, zorder=4))
        fh = np.hypot(f[0], f[1])
        dir_h = f[side_ax] / max(fh, 1e-9)
        for dv in (-half_vfov, 0.0, half_vfov):
            ang = np.arcsin(np.clip(f[2], -1, 1)) + dv
            L = 3.0
            axs.plot([p[side_ax], p[side_ax] + L * np.cos(ang) * dir_h], [p[2], p[2] + L * np.sin(ang)],
                     color="#ff7f0e", lw=0.9 if dv == 0 else 0.6, ls="-" if dv == 0 else ":", zorder=5)
        axs.plot(p[side_ax], p[2], "o", color="#ff7f0e", ms=4, zorder=6)
        axs.set_xlim(ext_side[0], ext_side[1]); axs.set_ylim(ext_side[2], ext_side[3])
        axs.set_xticks([]); axs.tick_params(axis="y", labelsize=7)
        axs.set_title(f"side ({'xy'[side_ax]}-z): log pi over {'xy'[other]}", fontsize=8)
    out = Path(a.out or f"results/figures/hssd_strip_{a.scene}_{a.episode}_{a.label or 'run'}.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
