"""HSSD stage 2b: the detector constants of formulation §7 and §8.2 fitted on the train frames (scripts/hssd/calibrate.py).

    uv run python scripts/hssd/fit_constants.py --root results/hssd/calib/oracle [--a0 0.000488] [--boot 200]

The model and the likelihood are the Isaac fit's (scripts/isaac/fit_constants.py: joint_nll, fit_joint): all six
constants (eta_f, eta_t, k, a0, beta, lambda) by maximum likelihood of the model's own observation likelihood at each
frame's true hypothesis (v*, s*), averaged over the cell's points. a0 is held at the oracle's DET threshold
(hssd.oracle_min_px over the image area), as for the Isaac oracle.

What is HSSD's: a scene holds several goal instances, so which one is a frame's true hypothesis is chosen per frame
(rule a, agreed 2026-10-08):
- DET: the reported instance (the goal with the most valid-depth pixels); its centroid is the frame's c. A DET whose
  mask lies on no goal instance (a false positive of an open-set or injected channel; det_goal empty) takes its
  hypothesis as a NO_DET does, and its centroid is explained by the model's false-positive branch.
- NO_DET: the instance the model finds most detectable in that frame, the largest r_bar(v*_g, s*_g) under the current
  constants: the one that should have been seen.
Instances the candidate cells do not represent (scripts/hssd/calibrate.py replay) are never chosen; a DET on one, or a frame
with none represented, is left out (counted). Since the NO_DET choice depends on the constants, the fit alternates:
choose with the spec constants, fit, choose again with the fitted ones, fit again, until the choice no longer changes
(at most 5 rounds). u = 1 / (the episode's workspace volume), as the runner's. Intervals: a bootstrap over episodes
with the choice held at the final one.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import isaac_scripts  # noqa: E402

fcc = isaac_scripts.load("fit_constants")  # scripts/isaac/fit_constants.py
A0_HI, A0_LO, B_HI, B_LO, K_HI, K_LO, L_HI, L_LO, HI, LO = (fcc.A0_HI, fcc.A0_LO, fcc.B_HI, fcc.B_LO, fcc.K_HI, fcc.K_LO,
                                                         fcc.L_HI, fcc.L_LO, fcc.HI, fcc.LO)
fit, fit_joint, r_bar, r_of = fcc.fit, fcc.fit_joint, fcc.r_bar, fcc.r_of

SPEC = {"false_positive_rate": 0.01, "detection_gain": 0.95, "h_slope": 3.0, "kappa_beta": 0.3333, "kappa_lambda": 0.2887}
KEYS = ["false_positive_rate", "detection_gain", "h_slope", "h_area0", "kappa_beta", "kappa_lambda"]


def load(root: Path, model_area: Path) -> list[dict]:
    rows = []
    for p in sorted(root.glob("frames_*.csv")):
        rows += list(csv.DictReader(open(p)))
    ma = {(m["episode_id"], int(m["frame"])): m for m in csv.DictReader(open(model_area))}
    for r in rows:
        m = ma.get((r["episode_id"], int(r["frame"])))
        if m is None:
            raise SystemExit(f"model_area.csv has no frame {r['episode_id']} {r['frame']}: run scripts/hssd/calibrate.py replay")
        r["goals"] = json.loads(m["goals"])
        r["log_u"] = -float(np.log(float(m["workspace_volume"])))
    return rows


def choose(rows: list[dict], par: dict, a0: float) -> tuple[list[int], dict]:
    """The true hypothesis of every frame (index into its goals, -1: left out) under rule (a)."""
    pick, why = [], {"det_unrepresented": 0, "none_represented": 0, "det_false": 0}
    for r in rows:
        gs = r["goals"]
        if not gs:
            pick.append(-1)
            why["none_represented"] += 1
            continue
        if r["status"] == "DET" and str(r["det_goal"]) != "":
            j = next((i for i, g in enumerate(gs) if str(g["goal"]) == str(r["det_goal"])), -1)
            why["det_unrepresented"] += j < 0
            pick.append(j)
            continue
        why["det_false"] += r["status"] == "DET"
        a = torch.as_tensor([g["a_cell"] for g in gs], dtype=torch.float64)
        T = torch.as_tensor([g["T_cell"] for g in gs], dtype=torch.float64)
        rb = r_bar(a, par["false_positive_rate"], par["detection_gain"], par["h_area0"], par["h_slope"], T)
        pick.append(int(torch.argmax(rb)))
    return pick, why


def joint_data(rows: list[dict], pick: list[int], offsets: np.ndarray) -> tuple[dict, list[int]]:
    """scripts/isaac/fit_constants.py: joint_data's arrays at the chosen hypotheses; returns (J, the kept row indices)."""
    keep = [i for i, j in enumerate(pick) if j >= 0]
    Q = len(offsets)
    n = len(keep)
    a, T = np.zeros((n, Q)), np.zeros((n, Q))
    y = np.zeros(n)
    has_c = np.zeros(n, dtype=bool)
    # multi-mask: every reported mask's centroid (r["centroids"], else the one centroid); M the most in a frame
    cents = [[json.loads(r["centroid"])] if r.get("centroid") else [] for r in (rows[i] for i in keep)]
    for k, i in enumerate(keep):
        if rows[i].get("centroids"):
            cents[k] = json.loads(rows[i]["centroids"])
    M = max([1] + [len(c) for c in cents])
    D, Nv, cm = np.zeros((n, M, Q, 3)), np.zeros((n, Q, 3)), np.zeros((n, M), dtype=bool)
    S_, logu = np.ones(n), np.zeros(n)
    for k, i in enumerate(keep):
        r, g = rows[i], rows[i]["goals"][pick[i]]
        a[k], T[k] = g["a_cell"], g["T_cell"]
        y[k] = r["status"] == "DET"
        S_[k], logu[k] = g["s_star"], r["log_u"]
        if r["status"] != "DET" or not cents[k]:
            continue
        p = np.asarray(json.loads(r["cam_pos"]))
        x = np.asarray(g["x_star"])[None, :] + offsets
        nv = p[None, :] - x
        if np.min(np.linalg.norm(nv, axis=1)) < 1e-9:
            continue
        has_c[k] = True
        Nv[k] = nv / np.linalg.norm(nv, axis=1, keepdims=True)
        for j, c in enumerate(cents[k]):
            D[k, j], cm[k, j] = np.asarray(c)[None, :] - x, True
    return {"a": a, "T": T, "y": y, "has_c": has_c, "D": D, "Nv": Nv, "cm": cm, "S": S_, "logu": logu}, keep


def joint_nll(J: dict, idx, eta_f, eta_t, a0, k, beta, lam) -> torch.Tensor:
    """scripts/isaac/fit_constants.py: joint_nll with several masks per frame (multi-mask, as the runner's DET update): at most
    one mask is the target, each with probability 1/M, so the true branch's Gaussian is the mean over the M centroids
    (per cell point q). With one centroid per frame it is the single-mask likelihood exactly."""
    t = lambda x: torch.as_tensor(x[idx], dtype=torch.float64)  # noqa: E731
    a, y, hc, D, Nv, S_, logu = t(J["a"]), t(J["y"]), torch.as_tensor(J["has_c"][idx]), t(J["D"]), t(J["Nv"]), t(J["S"]), t(J["logu"])
    cm = torch.as_tensor(J["cm"][idx]).clone()
    cm[~cm.any(dim=1), 0] = True  # a frame with no centroid never reads log_c; a finite dummy keeps the gradient clean
    rq = r_of(a, eta_f, eta_t, a0, k, t(J["T"])).clamp(1e-12, 1 - 1e-12)  # [n, Q]
    r = rq.mean(dim=1)
    var = (lam * S_) ** 2  # [n]
    res = D - beta * S_[:, None, None, None] * Nv[:, None]  # [n, M, Q, 3]
    log_n = -1.5 * torch.log(2 * np.pi * var)[:, None, None] - 0.5 * (res * res).sum(dim=3) / var[:, None, None]  # [n, M, Q]
    log_n = torch.where(cm[:, :, None], log_n, torch.full_like(log_n, -torch.inf))
    m = cm.sum(dim=1).clamp(min=1).to(torch.float64)
    log_mean = torch.logsumexp(log_n, dim=1) - torch.log(m)[:, None]  # [n, Q] the mean over the masks
    log_true = torch.logsumexp(torch.log(torch.clamp(rq - eta_f, min=1e-300)) + log_mean, dim=1) - np.log(rq.shape[1])
    log_c = torch.logaddexp(torch.log(torch.as_tensor(eta_f, dtype=torch.float64)) + logu, log_true)
    ll = torch.where(y > 0, torch.where(hc, log_c, torch.log(r)), torch.log(1 - r))
    return -ll.sum()


fcc.joint_nll = joint_nll  # fit_joint looks it up in its own module


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/hssd/calib/oracle")
    ap.add_argument("--config", default="configs/hssd.yaml")
    ap.add_argument("--a0", default=None, help="a value, 'fit' (open-set channels, as Isaac), default: hssd.oracle_min_px / image area")
    ap.add_argument("--model-area", default="", help="the replay's model_area.csv (default: <root>/model_area.csv, else "
                    "<root>/../oracle/model_area.csv: the channels share the oracle's poses)")
    ap.add_argument("--boot", type=int, default=200)
    a = ap.parse_args()
    from bamos.config import load_config
    from bamos.visibility.engine import cell_offsets

    cfg = load_config(a.config)
    rc = cfg["cameras"]["robot"]
    a0_oracle = cfg["hssd"]["oracle_min_px"] / float(rc["width"] * rc["height"])
    a0 = None if a.a0 == "fit" else (float(a.a0) if a.a0 is not None else a0_oracle)
    offsets = cell_offsets(float(cfg["mapping"]["voxel_size_m"]) * int(cfg["target"]["center_stride_voxels"]))
    root = Path(a.root)
    ma = Path(a.model_area) if a.model_area else (root / "model_area.csv" if (root / "model_area.csv").exists()
                                                    else root.parent / "oracle" / "model_area.csv")
    rows = load(root, ma)
    eps = sorted({r["episode_id"] for r in rows})
    print(f"{root}: {len(rows)} frames, {sum(r['status'] == 'DET' for r in rows)} DET, {len(eps)} train episodes; " + ("a0 fitted" if a0 is None else f"a0 held at {a0:.4g}") + f"; model areas from {ma}")

    par = {**SPEC, "h_area0": a0 if a0 is not None else a0_oracle}
    pick, why = choose(rows, par, a0)
    for it in range(5):
        J, keep = joint_data(rows, pick, offsets)
        par = fit_joint(J, np.arange(len(keep)), a0, par)
        new, why = choose(rows, par, a0)
        changed = sum(p != q for p, q in zip(pick, new))
        print(f"round {it + 1}: {len(keep)} frames used, {int(J['has_c'].sum())} with a centroid; "
              f"NO_DET choices changed by the fit: {changed}; " + ", ".join(f"{k} {par[k]:.4g}" for k in KEYS))
        pick = new
        if changed == 0:
            break
    J, keep = joint_data(rows, pick, offsets)
    best = fit_joint(J, np.arange(len(keep)), a0, par)
    print(f"left out: {why['det_unrepresented']} DET frames on an instance no cell represents, "
          f"{why['none_represented']} frames with no represented instance; {why['det_false']} DET frames on no goal (kept)")

    ep_of = [rows[i]["episode_id"] for i in keep]
    by_ep: dict = {}
    for k, e in enumerate(ep_of):
        by_ep.setdefault(e, []).append(k)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(a.boot):
        idx = np.concatenate([by_ep[e] for e in rng.choice(list(by_ep), size=len(by_ep), replace=True)])
        boots.append(fit_joint(J, idx, a0, best))
    bounds = {"h_slope": (K_LO, K_HI), "h_area0": (A0_LO, A0_HI), "kappa_beta": (B_LO, B_HI), "kappa_lambda": (L_LO, L_HI)}
    ci = {}
    print(f"\njoint fit (used): {len(keep)} frames, {int(J['y'].sum())} DET, {int(J['has_c'].sum())} with a centroid")
    print(f"{'constant':<22}{'spec':>9}{'fitted':>10}{'95% CI':>26}")
    for k_ in KEYS:
        if k_ == "h_area0" and a0 is not None:
            print(f"{k_:<22}{'':>9}{a0:>10.4g}   (held)")
            continue
        v = np.array([b[k_] for b in boots])
        v = v[np.isfinite(v)]
        ci[k_] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [float("nan")] * 2
        lo_, hi_ = bounds.get(k_, (LO, HI))
        edge = "  at the bound" if best[k_] <= lo_ + 1e-4 * abs(lo_) or best[k_] >= hi_ - 1e-4 * abs(hi_) else ""
        print(f"{k_:<22}{SPEC.get(k_, a0_oracle):>9.4g}{best[k_]:>10.4g}   [{ci[k_][0]:.4g}, {ci[k_][1]:.4g}]{edge}")
    all_idx = np.arange(len(keep))
    nll_of = lambda q: float(joint_nll(J, all_idx, q["false_positive_rate"], q["detection_gain"], q.get("h_area0", a0_oracle), q["h_slope"],  # noqa: E731
                                        q["kappa_beta"], q["kappa_lambda"]))
    print(f"joint log-likelihood per frame: spec {-nll_of(SPEC) / len(keep):.4f}   fitted {-best['nll'] / len(keep):.4f}")
    # the event-only fit, a diagnostic of the detection curve alone
    ev = fit(J["a"], J["y"], best["h_area0"], J["T"])
    print(f"event-only fit (diagnostic): eta_f {ev['false_positive_rate']:.4g}, eta_t {ev['detection_gain']:.4g}, "
          f"k {ev['h_slope']:.4g}; direct out-of-view DET rate {ev['eta_f_direct']:.4g}")
    Am = J["a"].mean(axis=1)
    print(f"\n{'a bin (chosen hypothesis)':<26}{'n':>6}{'DET':>7}{'r fitted':>10}")
    bins = [(-1, 0, "a = 0 (out of view)")] + [(10**lo, 10**(lo + 0.5), f"({10**lo:.1e}, {10**(lo + 0.5):.1e}]")
                                              for lo in np.arange(-5, 0, 0.5)]
    for lo, hi, lab in bins:
        m = (Am <= 0) if hi == 0 else (Am > lo) & (Am <= hi)
        if m.sum():
            rf = float(r_bar(torch.as_tensor(J["a"][m]), best["false_positive_rate"], best["detection_gain"], best["h_area0"],
                             best["h_slope"], torch.as_tensor(J["T"][m])).mean())
            print(f"{lab:<26}{int(m.sum()):>6}{J['y'][m].mean():>7.3f}{rf:>10.3f}")
    occ = json.loads((root.parent / "occupancy_model.json").read_text()) if (root.parent / "occupancy_model.json").exists() else None
    out = {"a0": best["h_area0"], "a0_fitted": a0 is None, "n_frames": len(keep), "n_episodes": len(by_ep), "fitted": {k_: best[k_] for k_ in KEYS}, "ci95": ci,
           "spec": SPEC, "left_out": why, "joint_loglik_per_frame": {"spec": -nll_of(SPEC) / len(keep), "fitted": -best["nll"] / len(keep)},
           "event_only": ev, "occupancy_model": occ}
    (root / "fit.json").write_text(json.dumps(out, indent=1))
    yaml_lines = ["detection_likelihood: {" + ", ".join(f"{k_}: {best[k_]:.4g}" for k_ in KEYS[:4]) + "}",
                  f"planner: {{kappa_beta: {best['kappa_beta']:.4g}, kappa_lambda: {best['kappa_lambda']:.4g}}}"]
    if occ:
        yaml_lines.append(f"mapping: {{occupancy_model: [{occ['p0']:.4g}, {occ['q_occ']:.4g}, {occ['q_free']:.4g}]}}")
    (root / "fitted.yaml").write_text("\n".join(yaml_lines) + "\n")
    print(f"\nwrote {root / 'fit.json'} and {root / 'fitted.yaml'}:\n" + "\n".join(yaml_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
