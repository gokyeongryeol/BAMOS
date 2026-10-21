"""One Isaac Sim test episode searched with the oracle, SAM 3 and Falcon Perception, side by side: for each detector, a
few observations as rows (the first, evenly spaced ones and the last), each with the frame, the target belief from
above and from the side. The counterpart of the HSSD session's scripts/render_hssd_episode_strip.py.

    python scripts/render_isaac_episode_strip.py EPISODE --isaac-port 5570 --sam3-port 5610 --falcon-port 5620
        # writes results/figures/isaac_strip_<episode>.png (and .pdf); --split: one figure per detector

Each detector re-runs the episode exactly as its test arm (configs/conditions; Isaac backend, the experiment's render
and detection caches), and the replay is checked against that arm's step log. The frames shown are re-rendered with
FIG_RENDER_STEPS render steps (denoised) for display only. Panels:
  frame  RGB of the view; the target's visible silhouette in yellow, the reported mask in cyan; green border = DET;
  top    log pi summed over sizes and heights, over the room (grey: furniture footprints), red star = target,
         orange = camera, white = path so far;
  side   log pi summed over x, height on the vertical axis, grey = furniture, orange = camera.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bamos.config import config_hash, load_config  # noqa: E402
from bamos.environment.analytic_renderer import set_cast_backend  # noqa: E402
from bamos.environment.isaac_client import IsaacClient  # noqa: E402
from bamos.environment.open_set import DetectorClient  # noqa: E402
from bamos.experiment.benchmark import build_env_for_spec, code_commit, make_episode_specs  # noqa: E402
from bamos.observation.types import DetectionStatus  # noqa: E402
from bamos.planning.runner import EpisodeContext, EpisodeRunner  # noqa: E402
from plot_belief_evolution import overrides_of  # noqa: E402

C = Path("configs/conditions")
LOG_RANGE = 1e-5  # belief panels show pi from max * LOG_RANGE to max
DETECTORS = [("oracle", "oracle", "b_oracle_isaac"), ("SAM 3", None, "bp_bamos"), ("Falcon", "falcon", "fa_bamos")]


def config_for(cond: str) -> dict:
    cfg = load_config("configs/normal.yaml", [{"stop": {"mode": "joint"}, "environment": {"backend": "isaac"}},
                                              overrides_of([C / "common.args", C / f"{cond}.channel", C / f"{cond}.constants"]),
                                              {"environment": {"isaac": {"render_cache_dir": "results/cache/isaac"}}}])
    return cfg


def replay(ep: str, cond: str, arm: str, isaac: IsaacClient, detector: DetectorClient | None) -> dict:
    cfg = config_for(cond)
    spec = next(s for s in make_episode_specs(cfg, "test") if s.episode_id == ep)
    env, manifest = build_env_for_spec(spec, cfg, None, isaac, detector)
    obs_log = []
    orb = env.observe_robot
    env.observe_robot = lambda pid, rng, k: obs_log.append(orb(pid, rng, k)) or obs_log[-1]
    ctx = EpisodeContext("strip", ep, spec.layout_id, spec.layout_seed, spec.episode_seed, "bamos", 0, 0, config_hash(cfg),
                         code_commit())
    arrays: dict = {}
    runner = EpisodeRunner(env, cfg, ctx, None, arrays)
    res = runner.run()
    cc = runner.state.candidate_centers()
    pis = [np.asarray(p, dtype=np.float64).reshape(len(cc), -1).sum(axis=1) for p in arrays["pi"]]
    log = Path("results/raw/test") / arm / "steps" / f"{ep}.jsonl"
    if log.exists():
        logged = [json.loads(line) for line in open(log) if line.strip()]
        m = min(len(pis), len(logged))
        dev = max(abs(pis[i].max() - logged[i]["posterior_max"]) for i in range(m))
        print(f"{ep} {arm}: replay vs test log, max |pi(v_hat) difference| {dev:.1e} over {m} observations"
              f" ({len(pis)} replayed, {len(logged)} logged)")
    return {"env": env, "manifest": manifest, "obs": obs_log, "pi": pis, "cc": cc, "result": res, "cfg": cfg}


def grid2(cc: np.ndarray, p: np.ndarray, a: int, b: int, step: float):
    ia = np.round(cc[:, a] / step).astype(int)
    ib = np.round(cc[:, b] / step).astype(int)
    ia -= ia.min(); ib -= ib.min()
    H = np.zeros((ia.max() + 1, ib.max() + 1))
    np.add.at(H, (ia, ib), p)
    lo_a, lo_b = cc[:, a].min(), cc[:, b].min()
    ext = (lo_a - step / 2, lo_a + (H.shape[0] - 0.5) * step, lo_b - step / 2, lo_b + (H.shape[1] - 0.5) * step)
    return H, ext


def boxes(scene, a: int, b: int):
    for o in scene.objects:
        if o.role in ("floor", "wall") or o.instance_id == scene.gt_target_instance_id:
            continue
        lo, hi = o.aabb()
        yield lo[a], lo[b], hi[a] - lo[a], hi[b] - lo[b]


def draw(ep: str, runs: list[tuple[str, dict]], out: Path, framer_port: int, n_rows: int) -> None:
    from bamos.visualization.isaac_frames import IsaacFramer

    scene = runs[0][1]["env"].scene
    gt = np.asarray(scene.gt_target_center)
    W, D, Hc = (float(x) for x in scene.workspace_hi)
    framer = IsaacFramer(runs[0][1]["env"], port=framer_port)
    cols = 3 * len(runs)
    fig = plt.figure(figsize=(2.05 * cols, 0.7 + 1.75 * n_rows))
    gs = fig.add_gridspec(n_rows, cols, width_ratios=[1.25, 1.0, 0.75] * len(runs), wspace=0.12, hspace=0.42,
                          left=0.01, right=0.99, top=1 - 0.55 / (0.7 + 1.75 * n_rows), bottom=0.02)
    for j, (label, R) in enumerate(runs):
        obs, pis, cc = R["obs"], R["pi"], R["cc"]
        step = float(R["cfg"]["mapping"]["voxel_size_m"]) * int(R["cfg"]["target"]["center_stride_voxels"])
        res = R["result"]
        n = len(obs)
        idx = sorted(set([0] + [round(i * (n - 1) / (n_rows - 1)) for i in range(1, n_rows)]))
        while len(idx) < n_rows and len(idx) < n:
            idx = sorted(set(idx + [max(set(range(n)) - set(idx))]))
        verdict = ("success" if res.actual_success else ("wrong declaration" if res.declared_stop else "no declaration"))
        fig.text((j + 0.5) / len(runs), 1 - 0.18 / (0.7 + 1.75 * n_rows), f"{label}: {verdict} after {n - 1} views",
                 ha="center", va="top", fontsize=10, fontweight="bold",
                 color="#2ca02c" if res.actual_success else "#d62728")
        path = np.array([o.world_from_camera[:3, 3] for o in obs])
        for r, i in enumerate(idx):
            o = obs[i]
            p = pis[min(i, len(pis) - 1)]  # the belief after observation i
            det = o.detection == DetectionStatus.DET
            # frame
            ax = fig.add_subplot(gs[r, 3 * j])
            img = framer.rgb(o.camera, o.world_from_camera)
            ax.imshow(img, interpolation="nearest")
            sil = None
            renderer = getattr(R["env"], "renderer", None) or getattr(getattr(R["env"], "inner", None), "renderer", None)
            if renderer is not None:
                try:
                    sil = renderer.target_silhouette(o.camera, o.world_from_camera)
                except Exception:  # noqa: BLE001
                    sil = None
            if sil is not None and np.asarray(sil).any():
                ax.contour(np.asarray(sil).astype(float), levels=[0.5], colors="yellow", linewidths=0.8)
            if det and o.target_mask is not None and np.asarray(o.target_mask).any():
                ax.contour(np.asarray(o.target_mask).astype(float), levels=[0.5], colors="cyan", linewidths=0.9)
            for sp in ax.spines.values():
                sp.set_edgecolor("#2ca02c" if det else "0.4"); sp.set_linewidth(2.2 if det else 0.8)
            ax.set_xticks([]); ax.set_yticks([])
            last = i == n - 1
            tag = (" DECLARE" if (last and res.declared_stop) else "")
            ax.set_title(f"obs {i} {o.detection.value}{tag}   max π {p.max():.2f}", fontsize=7,
                         color=("#2ca02c" if (last and res.actual_success) else ("#d62728" if (last and res.declared_stop) else "k")))
            # top
            ax = fig.add_subplot(gs[r, 3 * j + 1])
            Hm, ext = grid2(cc, p, 0, 1, step)
            hm = Hm.max()
            ax.imshow(np.ma.masked_less(Hm.T, hm * LOG_RANGE), origin="lower", extent=ext, cmap="viridis",
                      norm=LogNorm(hm * LOG_RANGE, hm), zorder=2, alpha=0.9)
            for x0, y0, w, h in boxes(scene, 0, 1):
                ax.add_patch(Rectangle((x0, y0), w, h, fc="0.8", ec="none", zorder=1))
            ax.plot(path[: i + 1, 0], path[: i + 1, 1], "-", color="w", lw=0.8, zorder=3)
            ax.plot(path[i, 0], path[i, 1], "o", color="#ff7f0e", ms=3.5, zorder=4)
            ax.plot(gt[0], gt[1], "*", color="#d62728", ms=6, mec="k", mew=0.4, zorder=5)
            ax.set_xlim(0, W); ax.set_ylim(0, D); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title("top: log π", fontsize=7)
            # side
            ax = fig.add_subplot(gs[r, 3 * j + 2])
            Hs, ext = grid2(cc, p, 1, 2, step)
            hs = Hs.max()
            ax.imshow(np.ma.masked_less(Hs.T, hs * LOG_RANGE), origin="lower", extent=ext, cmap="viridis",
                      norm=LogNorm(hs * LOG_RANGE, hs), zorder=2, alpha=0.9, aspect="auto")
            for y0, z0, w, h in boxes(scene, 1, 2):
                ax.add_patch(Rectangle((y0, z0), w, h, fc="0.85", ec="none", zorder=1))
            ax.plot(path[i, 1], path[i, 2], "o", color="#ff7f0e", ms=3.5, zorder=4)
            ax.plot(gt[1], gt[2], "*", color="#d62728", ms=6, mec="k", mew=0.4, zorder=5)
            ax.set_xlim(0, D); ax.set_ylim(0, Hc)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title("side: log π", fontsize=7)
    t = scene.gt_target_type
    fig.suptitle(f"Isaac Sim {ep} ({t})", fontsize=9, y=0.999, x=0.01, ha="left")
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200)
    framer.close()
    print(f"wrote {out}.(png|pdf)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("episode")
    ap.add_argument("--isaac-port", type=int, default=5570)
    ap.add_argument("--sam3-port", type=int, default=5610)
    ap.add_argument("--falcon-port", type=int, default=5620)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--out", default="")
    ap.add_argument("--split", action="store_true", help="one figure per detector: isaac_strip_<episode>_<detector>.png")
    ap.add_argument("--combined", action="store_true", help="with --split, also the side-by-side figure")
    a = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    set_cast_backend("cuda:0")
    sam3 = json.loads((C / "frozen.json").read_text())["sam3_condition"]
    runs = []
    host = config_for("oracle")["environment"]["isaac"]["host"]
    isaac = IsaacClient(host, a.isaac_port)  # the server takes one connection at a time: one client for every replay
    for label, cond, arm in DETECTORS:
        cond = cond or sam3
        cfg = config_for(cond)
        det = None
        if cfg["sensor_detector"]["mode"] == "open_set":
            port = a.sam3_port if label == "SAM 3" else a.falcon_port
            det = DetectorClient(host, port, cache_dir=cfg["sensor_detector"].get("open_set_cache_dir"))
        runs.append((label, replay(a.episode, cond, arm, isaac, det)))
    isaac.close()  # the framer opens its own connection
    base = a.out or f"results/figures/isaac_strip_{a.episode}"
    if a.split:
        for label, R in runs:
            tag = {"oracle": "oracle", "SAM 3": "sam3", "Falcon": "falcon"}[label]
            draw(a.episode, [(label, R)], Path(f"{base}_{tag}"), a.isaac_port, a.rows)
    if not a.split or a.combined:
        draw(a.episode, runs, Path(base), a.isaac_port, a.rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
