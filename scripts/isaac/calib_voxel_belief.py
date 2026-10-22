"""Stage 2 input: the voxel belief's model, and every calib frame's model quantities at the true hypothesis.

    uv run python scripts/isaac/calib_voxel_belief.py [--calib results/raw/calib] [--gpu 0]

1. The Bernoulli voxel belief's constants (formulation §4) by maximum likelihood: the prior occupancy p0 and the hit
   probabilities q_occ, q_free of a ray entering an occupied / a free voxel. Each episode's frames of the oracle
   channel (depth and GT mask as reported, in the order observed) give every voxel n observations and h hits, and
   independently over the voxels with n > 0, h ~ p0 Bin(n, q_occ) + (1 - p0) Bin(n, q_free). One model serves every
   channel: it describes the scenes and the depth sensor, not the detector. Written to <calib>/occupancy_model.json.
2. For every channel, the frames are replayed in their recorded order with that model: before each frame, at the true
   hypothesis (v*, s*) (the candidate nearest the GT centre, the size hypothesis of the target's type), the image
   share phi, the apparent area a_hat and the expected front-face transmittance T of the belief built from the
   earlier frames (formulation §6, §7.2), at each of the cell's points x_{v*} + o_q the planner averages over
   (formulation §7.3; columns a_cell, T_cell, their means a_model, T_model); then the frame's depth updates the belief,
   its reported mask left out.
   Written to <channel>/model_area.csv, the fit's input (scripts/isaac/fit_constants.py).

The masks are the channel's as collected. A channel with recorded instances (SAM 3) is replayed once per score
threshold, since a raised threshold turns some DET frames into NO_DET, whose mask is then not left out: a frame counts
as DET at threshold t if its top-scoring instance (at least min_mask_px) scores at least t, and its mask is that
instance's, the same at every threshold it passes. Those channels get model_area_t<100 t>.csv, the others
model_area.csv. The episodes are rebuilt with the analytic backend (placement only); the GT centre of
every frame must agree with the rebuilt episode.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

SCORE_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)  # SAM 3's, as scripts/isaac/stage2_fit.sh fits them


def fit_binomial_mixture(h: np.ndarray, n: np.ndarray, w: np.ndarray) -> tuple[float, float, float, float]:
    """ML (p0, q_occ, q_free) of h ~ p0 Bin(n, q_occ) + (1 - p0) Bin(n, q_free), multiplicities w; q_occ > q_free."""
    import torch
    H, N, Wt = (torch.as_tensor(x, dtype=torch.float64) for x in (h, n, w))
    comb = torch.lgamma(N + 1) - torch.lgamma(H + 1) - torch.lgamma(N - H + 1)
    lg = lambda p: float(np.log(p / (1 - p)))  # noqa: E731
    th = torch.tensor([lg(0.05), lg(0.5), lg(0.01)], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([th], lr=1.0, max_iter=500, tolerance_grad=1e-12, tolerance_change=1e-14,
                            line_search_fn="strong_wolfe")

    def ll():
        p0, qo, qf = torch.sigmoid(th[0]), torch.sigmoid(th[1]), torch.sigmoid(th[2])
        lo = torch.log(p0) + H * torch.log(qo) + (N - H) * torch.log1p(-qo)
        lf = torch.log1p(-p0) + H * torch.log(qf) + (N - H) * torch.log1p(-qf)
        return (Wt * (comb + torch.logaddexp(lo, lf))).sum()

    def closure():
        opt.zero_grad()
        loss = -ll()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        p0, qo, qf = (float(torch.sigmoid(x)) for x in th)
        if qo < qf:  # the labels of the two components are arbitrary: "occupied" stops more rays
            p0, qo, qf = 1 - p0, qf, qo
        return p0, qo, qf, float(ll())


def fit_occupancy(calib: Path, episode, frames_of, stride: int, dev: str, voxel_m: float) -> tuple[float, float, float]:
    """1. The voxel belief's model, from the oracle channel; written to <calib>/occupancy_model.json."""
    from bamos.belief.voxel_belief import VoxelBelief
    ora = calib / "oracle"
    eps = sorted(p.stem for p in (ora / "arrays").glob("*.npz"))
    pairs: dict = {}
    for ep in eps:
        vgrid = episode(ep)[6]
        vb = VoxelBelief(vgrid, 0.5, 0.5, 0.01, dev, stride)  # counts only: the constants do not enter them
        for cid, pid, T, cam, d, m in frames_of(ora, ep):
            vb.update_subsampled(d, m if m.any() else None, cam, T)
        h, n = vb.n_hit.cpu().numpy(), (vb.n_hit + vb.n_pass).cpu().numpy()
        sel = n > 0
        for key, c in zip(*np.unique(np.stack([h[sel], n[sel]], 1), axis=0, return_counts=True)):
            pairs[tuple(key)] = pairs.get(tuple(key), 0) + int(c)
    keys = np.array(list(pairs))
    p0, q_occ, q_free, ll = fit_binomial_mixture(keys[:, 0], keys[:, 1], np.array(list(pairs.values())))
    n_vox = int(sum(pairs.values()))
    hit_rate = float((keys[:, 0] * np.array(list(pairs.values()))).sum() / (keys[:, 1] * np.array(list(pairs.values()))).sum())
    model = {"p0": p0, "q_occ": q_occ, "q_free": q_free, "loglik": ll, "n_voxel_episodes": n_vox,
             "pooled_hit_rate": hit_rate, "n_episodes": len(eps), "stride": stride, "voxel_m": voxel_m,
             "unobserved_pass_probability": 1 - p0 * q_occ - (1 - p0) * q_free}
    (calib / "occupancy_model.json").write_text(json.dumps(model, indent=1))
    print(f"voxel belief: p0 {p0:.4g}, q_occ {q_occ:.4g}, q_free {q_free:.4g} (pooled hit rate {hit_rate:.4g}; an unobserved "
          f"voxel passes a ray with {model['unobserved_pass_probability']:.4g}), from {n_vox} voxel-episodes of {len(eps)} episodes",
          flush=True)
    return p0, q_occ, q_free


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="results/raw/calib")
    ap.add_argument("--config", default="configs/isaac.yaml")
    ap.add_argument("--gpu", type=int, default=0)
    # 2026-10-07 added channels (Falcon Perception, false-positive sweep): replay only these,
    # with the occupancy model already fitted (it is the oracle channel's and does not change), at these score thresholds
    ap.add_argument("--channels", nargs="*", default=None, help="replay only these channel directories")
    ap.add_argument("--thresholds", type=float, nargs="*", default=list(SCORE_THRESHOLDS),
                    help="score thresholds for channels whose instances carry scores")
    ap.add_argument("--reuse-occupancy", action="store_true", help="read <calib>/occupancy_model.json instead of refitting")
    a = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(a.gpu))
    import torch

    from bamos.belief.voxel_belief import VoxelBelief
    from bamos.config import load_config
    from bamos.environment.analytic_renderer import set_cast_backend
    from bamos.environment.layouts import TARGET_SPECS
    from bamos.experiment.benchmark import build_env_for_spec, make_episode_specs
    from bamos.mapping.grid import VoxelGrid
    from bamos.planning.runner import make_candidates, make_grid
    from bamos.planning.scorer import Scorer
    from bamos.visibility.engine import EngineConfig, VisibilityEngine, cell_offsets

    cfg = load_config(a.config, [{"environment": {"backend": "analytic"}, "sensor_detector": {"mode": "rendered_threshold"}}])
    mp = cfg["mapping"]
    sizes = np.asarray(cfg["planner"]["size_grid_m"], dtype=np.float64)
    offsets = cell_offsets(float(mp["voxel_size_m"]) * int(cfg["target"]["center_stride_voxels"]))
    set_cast_backend("cuda:0")
    dev = "cuda:0"
    calib = Path(a.calib)
    specs = {s.episode_id: s for s in make_episode_specs(cfg, "calib")}
    envs: dict = {}

    def episode(ep: str):
        if ep not in envs:
            env, _ = build_env_for_spec(specs[ep], cfg, None, None, None)
            grid = make_grid(cfg, env.scene)
            cand = make_candidates(cfg, grid, env.scene)
            eng = VisibilityEngine(grid, env.robot_camera, cand, EngineConfig(device=dev, ray_margin_m=float(mp["occupancy_ray_margin_m"])))
            eng.offsets = offsets  # the cell's points the planner averages over (formulation §7.3)
            sc = Scorer(eng, None)
            sc.sizes = sizes
            cc = grid.centers(grid.unravel(cand))
            gt = np.asarray(env.ground_truth().target_center)
            vstar = int(np.argmin(np.linalg.norm(cc - gt, axis=1)))
            si = int(np.argmin(np.abs(sizes - TARGET_SPECS[specs[ep].target_type]["s"])))
            hi = grid.lower + np.asarray(grid.shape) * grid.voxel_size
            vgrid = VoxelGrid.from_bounds(grid.lower, hi, float(mp["occupancy_voxel_m"]))
            envs[ep] = (env, eng, sc, vstar, si, gt, vgrid)
        return envs[ep]

    def frames_of(ch_dir: Path, ep: str):
        z = np.load(ch_dir / "arrays" / f"{ep}.npz")
        env = episode(ep)[0]
        for k in range(len(z["camera_id"])):
            cid, pid = str(z["camera_id"][k]), int(z["pose_id"][k])
            T, cam = env.pool.world_from_camera[pid], env.robot_camera
            yield cid, pid, T, cam, z["depth"][k], z["mask"][k]

    stride = int(mp["occupancy_pixel_stride"])
    if a.reuse_occupancy:
        model = json.loads((calib / "occupancy_model.json").read_text())
        p0, q_occ, q_free = model["p0"], model["q_occ"], model["q_free"]
        print(f"voxel belief (reused): p0 {p0:.4g}, q_occ {q_occ:.4g}, q_free {q_free:.4g}", flush=True)
    else:
        p0, q_occ, q_free = fit_occupancy(calib, episode, frames_of, stride, dev, float(mp["occupancy_voxel_m"]))

    # 2. every channel's frames with that model (open-set channels once per score threshold)
    min_px = int(cfg["sensor_detector"]["open_set_min_mask_px"])

    def replay(ch_dir: Path, name: str, th, top: dict, gts: dict):
        out = []
        for ep in sorted(p.stem for p in (ch_dir / "arrays").glob("*.npz")):
            env, eng, sc, vstar, si, gt, vgrid = episode(ep)
            if ep in gts and np.max(np.abs(np.asarray(gts[ep]) - gt)) > 1e-3:
                raise SystemExit(f"{ep}: the rebuilt GT centre {gt} differs from the frames' {gts[ep]}")
            vb = VoxelBelief(vgrid, p0, q_occ, q_free, dev, stride)
            for cid, pid, T, cam, d, m in frames_of(ch_dir, ep):
                a_cell, T_cell = [], []
                for q in range(len(offsets)):
                    phi_q = float(eng.candidate_coverage(T, sizes, cam, q)[0, vstar, si])
                    a_cell.append(phi_q * float(sc.projected_area(T, cam, q)[0, vstar, si]))
                    T_cell.append(float(eng.transmittance(T[:3, 3][None], sizes, vb, q)[0, vstar, si]))
                out.append({"episode_id": ep, "camera_id": cid, "pose_id": pid, "phi": round(float(np.mean([a > 0 for a in a_cell])), 6),
                            "a_hat": "", "a_model": f"{np.mean(a_cell):.6e}", "T_model": f"{np.mean(T_cell):.6f}",
                            "a_cell": json.dumps([float(f"{a:.6e}") for a in a_cell]),
                            "T_cell": json.dumps([round(t, 6) for t in T_cell])})
                det = m.any() and (th is None or top.get((ep, cid, pid), -np.inf) >= th)
                vb.update_subsampled(d, m if det else None, cam, T)
        with open(ch_dir / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0]))
            w.writeheader()
            w.writerows(out)
        print(f"wrote {ch_dir / name} ({len(out)} frames)", flush=True)

    for ch_dir in sorted(p for p in calib.iterdir() if (p / "arrays").is_dir()):
        if a.channels is not None and ch_dir.name not in a.channels:
            continue
        gts, top = {}, {}
        for f in sorted(ch_dir.glob("frames_*.csv")):
            for r in csv.DictReader(open(f)):
                gts[r["episode_id"]] = json.loads(r["gt_center"])
                if r.get("instances"):
                    sc_ = [x[1] for x in json.loads(r["instances"]) if x[0] >= min_px]
                    top[(r["episode_id"], r["camera_id"], int(r["pose_id"]))] = max(sc_) if sc_ else -np.inf
        # a model without scores (Falcon Perception) records -1 for every instance: one replay, every mask a DET
        scored = any(v > -1 for v in top.values() if np.isfinite(v))
        runs = [(f"model_area_t{round(100 * th):03d}.csv", th) for th in a.thresholds] if scored else [("model_area.csv", None)]
        for name, th in runs:
            replay(ch_dir, name, th, top, gts)
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
