"""The belief pi_t(v) of one episode as the active views accumulate: a row of top-down panels.

    uv run python scripts/plot_belief_evolution.py EPISODE_ID [--condition oracle] [--split test] [--k 0,1,2,5,10,20,30]
    # an open-set condition needs its servers (scripts/launch_open_set.sh)
    uv run python scripts/plot_belief_evolution.py EPISODE_ID --condition falcon --isaac-port 5555 --detector-port 5600

The episode is re-run with the frozen settings of the condition (configs/conditions/<condition>.channel and
.constants, common.args), BAMOS, stop.mode=none, recording pi after every observation. Each panel shows the marginal
pi(v) summed over the size and height axes, after k active views (k = 0: the start view): the
layout, the GT target (star), the camera path so far (dots: DET filled, NO_DET open; the last one with its heading),
the MAP candidate (x) and the success radius tau around it, once pi has concentrated (max > 10 / n_C). The title gives k, the last observation and the mass
within tau of the MAP (the stop rule's quantity, spec 10). Writes <out>/<episode>__<condition>.png.
"""
from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def overrides_of(files: list[Path]) -> dict:
    """The --override key=value flags of the frozen condition files, as a nested dict."""
    out: dict = {}
    for f in files:
        toks = shlex.split(f.read_text())
        for i in range(0, len(toks) - 1):
            if toks[i] != "--override":
                continue
            key, _, val = toks[i + 1].partition("=")
            node = out
            for p in key.split(".")[:-1]:
                node = node.setdefault(p, {})
            try:
                node[key.split(".")[-1]] = eval(val, {"__builtins__": {}}, {})
            except Exception:
                node[key.split(".")[-1]] = val
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("episode_id")
    ap.add_argument("--condition", default="oracle", help="a configs/conditions/<condition>.channel")
    ap.add_argument("--split", default="test", choices=["test", "calib", "dev", "dev2"])
    ap.add_argument("--k", default="0,1,2,5,10,20,30", help="active view counts to show")
    ap.add_argument("--conditions-dir", default="configs/conditions")
    ap.add_argument("--isaac-port", type=int, default=5555)
    ap.add_argument("--detector-port", type=int, default=5600)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default="results/figures/belief")
    a = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(a.gpu))

    from bamos.config import config_hash, load_config
    from bamos.environment.analytic_renderer import set_cast_backend
    from bamos.experiment.benchmark import build_env_for_spec, code_commit, make_episode_specs
    from bamos.planning.runner import EpisodeContext, EpisodeRunner
    from bamos.visualization.topdown import scene_topdown

    C = Path(a.conditions_dir)
    ov = overrides_of([C / f"{a.condition}.channel", C / f"{a.condition}.constants", C / "common.args"])
    open_set = ov.get("sensor_detector", {}).get("mode") == "open_set"
    base = {"environment": {"backend": "isaac" if open_set else "analytic",
                            "isaac": {"render_cache_dir": "results/cache/isaac"}},
            "stop": {"mode": "none"}}
    cfg = load_config("configs/normal.yaml", [base, ov])
    spec = next((s for s in make_episode_specs(cfg, a.split) if s.episode_id == a.episode_id), None)
    if spec is None:
        raise SystemExit(f"{a.episode_id} is not a {a.split} episode")
    set_cast_backend("cuda:0")
    isaac = det = None
    if open_set:
        from bamos.environment.isaac_client import IsaacClient
        from bamos.environment.open_set import DetectorClient
        isaac = IsaacClient("127.0.0.1", a.isaac_port)
        det = DetectorClient("127.0.0.1", a.detector_port, cache_dir=cfg["sensor_detector"]["open_set_cache_dir"])
    env, manifest = build_env_for_spec(spec, cfg, None, isaac, det)

    obs_log = []
    orig_robot = env.observe_robot

    def cap_robot(pid, rng, k):
        o = orig_robot(pid, rng, k)
        obs_log.append(o)
        return o

    env.observe_robot = cap_robot
    ctx = EpisodeContext("belief", spec.episode_id, spec.layout_id, spec.layout_seed, spec.episode_seed, "bamos", 0, 0,
                         config_hash(cfg), code_commit())
    arrays: dict = {}
    runner = EpisodeRunner(env, cfg, ctx, None, arrays)
    runner.run()
    pis = arrays["pi"]  # one per observation, in order
    n_init = 1  # the start view
    cc = runner.state.candidate_centers()
    gt = np.asarray(env.ground_truth().target_center)
    tau = float(cfg["stop"]["success_radius_m"])
    W, D = env.scene.workspace_hi[:2]
    step = float(cfg["mapping"]["voxel_size_m"]) * int(cfg["target"]["center_stride_voxels"])  # the candidate lattice
    ks = [k for k in (int(x) for x in a.k.split(",")) if n_init - 1 + k < len(pis)]

    fig, axes = plt.subplots(1, len(ks), figsize=(3.2 * len(ks), 3.2 * D / W + 0.6), squeeze=False)
    for ax, k in zip(axes[0], ks):
        i = n_init - 1 + k  # index of the observation after which pi is shown
        pi = np.asarray(pis[i], dtype=np.float64).reshape(len(cc), -1).sum(axis=1)
        ix = np.round(cc[:, 0] / step).astype(int)
        iy = np.round(cc[:, 1] / step).astype(int)
        H = np.zeros((ix.max() + 1, iy.max() + 1))
        np.add.at(H, (ix, iy), pi)  # sum over heights: the xy marginal
        scene_topdown(ax, manifest)
        ax.imshow(H.T, origin="lower", extent=(-step / 2, (H.shape[0] - 0.5) * step, -step / 2, (H.shape[1] - 0.5) * step),
                  cmap="viridis", alpha=0.75, vmin=0, vmax=max(H.max(), 1e-9), zorder=1)
        robot = [o for o in obs_log[: i + 1] if o.camera_type == "robot"]
        xs = [o.world_from_camera[0, 3] for o in robot]
        ys = [o.world_from_camera[1, 3] for o in robot]
        ax.plot(xs, ys, "-", color="w", lw=1.0, alpha=0.8, zorder=3)
        for o in robot:
            filled = o.detection.value == "DET"
            ax.plot(o.world_from_camera[0, 3], o.world_from_camera[1, 3], "o", ms=4, zorder=4,
                    mfc="#ff7f0e" if filled else "none", mec="#ff7f0e")
        last = obs_log[i]
        p, f = last.world_from_camera[:3, 3], last.world_from_camera[:3, 2]
        ax.arrow(p[0], p[1], f[0] * 0.6, f[1] * 0.6, head_width=0.15, color="#ff7f0e", zorder=5)
        m = int(np.argmax(pi))
        if pi[m] > 10.0 / len(pi):  # a MAP is only drawn once pi has concentrated (a near-uniform argmax is arbitrary)
            ax.plot(cc[m, 0], cc[m, 1], "x", color="w", ms=9, mew=2, zorder=6)
            ax.add_patch(plt.Circle(cc[m, :2], tau, fill=False, color="w", lw=1, zorder=6))
        ax.plot(gt[0], gt[1], "*", color="#d62728", ms=12, mec="k", zorder=7)
        mass = float(pi[np.linalg.norm(cc - cc[m], axis=1) <= tau + 1e-9].sum())  # spec 10, in 3-D
        ax.set_title(f"k = {k}  {last.detection.value}\nmass within tau {mass:.2f}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{spec.episode_id}__{a.condition}.png"
    fig.suptitle(f"{spec.episode_id} ({spec.target_type}, {spec.placement_mode}), {a.condition}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
