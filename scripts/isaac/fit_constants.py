"""Re-run stage 2: fit the detector constants of spec 7 and 8.2 to one detection channel.

    uv run python scripts/isaac/fit_constants.py --root results/raw/calib/oracle --a0 0.0001
    uv run python scripts/isaac/fit_constants.py --root results/raw/calib/sam3 --min-score 0.7 --out-name t070

Only constants change; the model's form stays that of the formulation:

    r = eta_f + (1 - eta_f) eta_t T h(a),   h(a) = sigma(k ln(a / a0))      (spec 7)

All six constants (eta_f, eta_t, k, a0, beta, lambda) are fitted jointly, by maximum likelihood of the model's own
observation likelihood (spec 8.1-8.2) at the model's true hypothesis (v*, s*) of every frame, v* the candidate nearest
the GT target centre on the lattice the planner uses (planning/runner.make_candidates):

    NO_DET:                  1 - r_bar,   r_bar = mean_q r(a_q, T_q)
    DET with a centroid c:   eta_f u + mean_q (r(a_q, T_q) - eta_f) N(c; x* + o_q + beta s* n_q, (lambda s*)^2 I)
    DET without a centroid:  r_bar           (fewer than n_min valid depth pixels)

averaged over the points x* + o_q inside the true cell (formulation §7.3), as the planner does;
a_q, T_q and n_q are taken at each point.

with a = a(v*, s*) = phi a_hat the model's own apparent area of the true hypothesis from the frame's camera pose
(formulation §6), T = T(v*, s*) the voxel belief's expected front-face transmittance before the frame (formulation §4,
§7.2), both written per channel by scripts/isaac/calib_voxel_belief.py to <channel>/model_area.csv; x* = x_{v*} the centre of that candidate and s* the size
hypothesis of the target's type, n the unit vector from x* to the camera and u = 1 / (workspace volume) the density
of a false positive's centroid. Both the area and the centroid are taken at v*, not at the GT centre, because the
model only ever evaluates lattice candidates: fitted from the exact centre, lambda s would leave out the lattice's own
offset (median 0.10 m) and come out far narrower than the spread the model meets, so that true detections read as
false positives; and r fitted on the target's actual area would come out as the detector's own sharp threshold, while
the model applies it to a(v*), which misses the actual area by the lattice offset (at the image edge, by most of it). Whether a detection was a false positive is left to the mixture; ground truth only places the
true hypothesis. a0 is fitted (--a0 fit, the open-set channels) or held at a given value (--a0 <value>: an oracle
channel's own DET area threshold, sensor_detector.min_normalized_area, which is what its a0 is by construction).
The event-only fit of (eta_f, eta_t, k, a0), the direct out-of-view estimate of eta_f (spec 7.2's definition) and the
closed-form Gaussian of the true detections (IoU >= eval-iou) are reported beside it as diagnostics. Every
probability is kept in [0.001, 0.999], k in [0.1, 100], beta in [-3, 3] and lambda in [0.01, 3]: an oracle reports
no false positive, and an unbounded estimate of exactly 0 would make one stray event impossible; its detection is a
hard threshold, and an unbounded k would run off to infinity. Intervals are a bootstrap over episodes, since the
frames of one episode share a scene.

Open-set frames are always rebuilt from the instances the calibration recorded, under --min-mask-px and
--min-score (a raised score threshold): an instance is kept if it covers at least open_set_min_mask_px pixels and scores at least the threshold,
the frame is DET if any is kept, and the kept one --mask-select picks (SAM 3: the top score) is the frame's mask, as
open_set_observe does.

The closed-form diagnostic on the true detections (d = c - x_{v*}): beta = mean(d . n / s*),
lambda^2 = mean(|d - beta s* n|^2 / s*^2) / 3; it also starts the joint fit.

Writes <root>/fit[_<out-name>].json and the matching override flags to <root>/overrides[_<out-name>].txt.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

LO, HI = 1e-3, 0.999  # bounds of eta_f, eta_t
K_LO, K_HI = 0.1, 100.0  # bounds of k: an oracle's hard threshold would otherwise send it to infinity
A0_LO, A0_HI = 1e-5, 0.1  # bounds of a fitted a0 (normalized image area)
SPEC = {"false_positive_rate": 0.01, "detection_gain": 0.95, "h_slope": 3.0, "h_area0": 0.0001,
        "kappa_beta": 0.3333, "kappa_lambda": 0.2887}
# the size hypothesis of each target type (planner.size_grid_m: size scales sqrt(surface area / 4))
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from bamos.environment.layouts import TARGET_SPECS  # noqa: E402
TYPE_SIZE = {k: v["s"] for k, v in TARGET_SPECS.items()}
B_LO, B_HI = -3.0, 3.0  # bounds of beta
L_LO, L_HI = 0.01, 3.0  # bounds of lambda


_CAND: dict = {}


def candidate_centres(archetype: str) -> np.ndarray:
    """[n_C, 3] the target-location candidates of an archetype's workspace, as the planner builds them."""
    if archetype not in _CAND:
        import sys
        from types import SimpleNamespace
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
        from bamos.config import load_config
        from bamos.environment.layouts import ARCHETYPES
        from bamos.planning.runner import make_candidates, make_grid
        cfg = load_config("configs/isaac.yaml")
        W, D = ARCHETYPES[archetype]["size"]
        scene = SimpleNamespace(workspace_lo=(0.0, 0.0, 0.0), workspace_hi=(W, D, ARCHETYPES[archetype]["ceiling"]), meta={})
        grid = make_grid(cfg, scene)
        _CAND[archetype] = grid.centers(grid.unravel(make_candidates(cfg, grid, scene)))
    return _CAND[archetype]


def true_hypothesis(archetype: str, gt: np.ndarray) -> np.ndarray:
    """x_{v*}: the centre of the candidate nearest the GT target centre."""
    cc = candidate_centres(archetype)
    return cc[int(np.argmin(np.linalg.norm(cc - gt[None, :], axis=1)))]


def workspace_volume(archetype: str) -> float:
    """The room the episode's workspace spans, (0, 0, 0) to (W, D, ceiling) (environment/episodes.py)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from bamos.environment.layouts import ARCHETYPES
    W, D = ARCHETYPES[archetype]["size"]
    return float(W * D * ARCHETYPES[archetype]["ceiling"])


def cell_offsets_cfg() -> np.ndarray:
    """[Q, 3] the points o_q the planner averages a cell's likelihood over (formulation §7.3)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from bamos.config import load_config
    from bamos.visibility.engine import cell_offsets
    cfg = load_config("configs/isaac.yaml")
    step = float(cfg["mapping"]["voxel_size_m"]) * int(cfg["target"]["center_stride_voxels"])
    return cell_offsets(step)


def joint_data(rows: list[dict]) -> dict:
    """Per frame: the areas a_q and gates T_q at the cell's points [n, Q], DET y, and for a DET with a centroid the
    offsets d_q = c - (x_{v*} + o_q) and units n_q [n, Q, 3], s* and log u."""
    off = cell_offsets_cfg()
    Q = len(off)
    a = np.array([json.loads(r["a_cell"]) for r in rows], dtype=float).reshape(len(rows), Q)
    T = np.array([json.loads(r["T_cell"]) for r in rows], dtype=float).reshape(len(rows), Q)
    y = np.array([r["status"] == "DET" for r in rows], dtype=float)
    has_c = np.zeros(len(rows), dtype=bool)
    D = np.zeros((len(rows), Q, 3)); Nv = np.zeros((len(rows), Q, 3)); S_ = np.ones(len(rows)); logu = np.zeros(len(rows))
    for i, r in enumerate(rows):
        logu[i] = -np.log(workspace_volume(r["archetype"]))
        S_[i] = TYPE_SIZE[r["target_type"]]
        if r["status"] != "DET" or not r.get("centroid"):
            continue
        c, g, p = (np.asarray(json.loads(r[k]), dtype=float) for k in ("centroid", "gt_center", "cam_pos"))
        x = true_hypothesis(r["archetype"], g)[None, :] + off  # [Q, 3]
        n = p[None, :] - x
        if np.min(np.linalg.norm(n, axis=1)) < 1e-9:
            continue
        has_c[i] = True; D[i] = c[None, :] - x; Nv[i] = n / np.linalg.norm(n, axis=1, keepdims=True)
    return {"a": a, "T": T, "y": y, "has_c": has_c, "D": D, "Nv": Nv, "S": S_, "logu": logu}


def joint_nll(J: dict, idx, eta_f, eta_t, a0, k, beta, lam) -> torch.Tensor:
    """- sum of the log observation likelihood above over the frames idx."""
    t = lambda x: torch.as_tensor(x[idx], dtype=torch.float64)  # noqa: E731
    a, y, hc, D, Nv, S_, logu = t(J["a"]), t(J["y"]), torch.as_tensor(J["has_c"][idx]), t(J["D"]), t(J["Nv"]), t(J["S"]), t(J["logu"])
    rq = r_of(a, eta_f, eta_t, a0, k, t(J["T"])).clamp(1e-12, 1 - 1e-12)  # [n, Q]
    r = rq.mean(dim=1)
    var = (lam * S_) ** 2  # [n]
    res = D - beta * S_[:, None, None] * Nv  # [n, Q, 3]
    log_n = -1.5 * torch.log(2 * np.pi * var)[:, None] - 0.5 * (res * res).sum(dim=2) / var[:, None]  # [n, Q]
    log_true = torch.logsumexp(torch.log(torch.clamp(rq - eta_f, min=1e-300)) + log_n, dim=1) - np.log(rq.shape[1])
    log_c = torch.logaddexp(torch.log(torch.as_tensor(eta_f, dtype=torch.float64)) + logu, log_true)
    ll = torch.where(y > 0, torch.where(hc, log_c, torch.log(r)), torch.log(1 - r))
    return -ll.sum()


def fit_joint(J: dict, idx, a0: float | None, start: dict) -> dict:
    """All six constants by maximum likelihood (a0 held when given), started at the event fit and closed-form kappa."""
    inv = lambda v, lo, hi: float(np.log((v - lo) / (hi - v)))  # noqa: E731
    la_lo, la_hi = np.log(A0_LO), np.log(A0_HI)
    cl = lambda v, lo, hi: float(np.clip(v, lo + 1e-6 * (hi - lo), hi - 1e-6 * (hi - lo)))  # noqa: E731
    th0 = [inv(cl(start["false_positive_rate"], LO, HI), LO, HI), inv(cl(start["detection_gain"], LO, HI), LO, HI),
           inv(cl(start["h_slope"], K_LO, K_HI), K_LO, K_HI), inv(cl(start["kappa_beta"], B_LO, B_HI), B_LO, B_HI),
           inv(cl(start["kappa_lambda"], L_LO, L_HI), L_LO, L_HI)]
    th0 += [inv(cl(np.log(start["h_area0"]), la_lo, la_hi), la_lo, la_hi)] if a0 is None else []
    th = torch.tensor(th0, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([th], lr=1.0, max_iter=1000, tolerance_grad=1e-10, tolerance_change=1e-12,
                            line_search_fn="strong_wolfe")
    a0_of = (lambda: torch.exp(bounded(th[5], la_lo, la_hi))) if a0 is None else (lambda: a0)
    par = lambda: (bounded(th[0]), bounded(th[1]), a0_of(), bounded(th[2], K_LO, K_HI),  # noqa: E731
                   bounded(th[3], B_LO, B_HI), bounded(th[4], L_LO, L_HI))

    def closure():
        opt.zero_grad()
        loss = joint_nll(J, idx, *par())
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        ef, et, a0v, kv, bv, lv = (float(x) for x in par())
        out = {"false_positive_rate": ef, "detection_gain": et, "h_slope": kv, "h_area0": a0v, "kappa_beta": bv,
               "kappa_lambda": lv}
        out["nll"] = float(joint_nll(J, idx, ef, et, a0v, kv, bv, lv))
    return out


def kappa_samples(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """(d [N,3], n [N,3], s* [N], episode ids) over the frames whose reported mask is a true detection."""
    D, Nv, S_, ep = [], [], [], []
    for r in rows:
        if r["status"] != "DET" or r["fp"] == "True" or not r.get("centroid"):
            continue
        c, g, p = (np.asarray(json.loads(r[k]), dtype=float) for k in ("centroid", "gt_center", "cam_pos"))
        x = true_hypothesis(r["archetype"], g)
        n = p - x
        if np.linalg.norm(n) < 1e-9:
            continue
        D.append(c - x); Nv.append(n / np.linalg.norm(n)); S_.append(TYPE_SIZE[r["target_type"]]); ep.append(r["episode_id"])
    return np.asarray(D).reshape(-1, 3), np.asarray(Nv).reshape(-1, 3), np.asarray(S_), ep


def fit_kappa(D: np.ndarray, Nv: np.ndarray, S_: np.ndarray) -> dict:
    if not len(S_):
        return {"kappa_beta": float("nan"), "kappa_lambda": float("nan")}
    beta = float(np.mean(np.sum(D * Nv, axis=1) / S_))
    res = D - beta * S_[:, None] * Nv
    lam = float(np.sqrt(np.mean(np.sum(res * res, axis=1) / S_**2) / 3.0))
    return {"kappa_beta": beta, "kappa_lambda": lam}


def kappa_loglik(D, Nv, S_, beta, lam) -> float:
    """mean log N(c; x* + beta s* n, (lambda s*)^2 I) over the samples, per frame."""
    var = (lam * S_) ** 2
    res = D - beta * S_[:, None] * Nv
    return float(np.mean(-1.5 * np.log(2 * np.pi * var) - 0.5 * np.sum(res * res, axis=1) / var))


def select(inst: list, rule: str) -> list:
    """The frame's one mask among the kept instances, by the channel's rule (open_set_observe)."""
    if rule == "score":
        return max(inst, key=lambda x: x[1])
    return max(inst, key=lambda x: x[0])


def load(root: Path, min_score: float, min_px: int, eval_iou: float, rule: str = "largest", ma_name: str = "") -> list[dict]:
    rows, mismatch = [], 0
    for p in sorted(root.glob("frames_*.csv")):
        for r in csv.DictReader(open(p)):
            if r["status"] not in ("DET", "NO_DET"):
                continue  # MISSING updates nothing (spec 2)
            if min_score > 0.0 and not r["instances"]:
                raise SystemExit(f"--min-score needs recorded instances ({p} has none)")
            if r["instances"]:
                # an open-set frame: always rebuilt from its instances under the current min_px and threshold, so
                # the fit follows the channel's settings rather than the ones in force when the frame was collected
                # the score threshold applies only when set, as in open_set_observe: a model that scores nothing
                # (Falcon) records every instance's score as -1
                inst = [x for x in json.loads(r["instances"]) if x[0] >= min_px and (min_score <= 0.0 or x[1] >= min_score)]
                r = dict(r)
                rebuilt = "DET" if inst else "NO_DET"
                # with no score threshold the rebuilt event must be the one the channel reported when collected
                mismatch += min_score <= 0.0 and rebuilt != r["status"]
                r["status"] = rebuilt
                sel = select(inst, rule) if inst else None
                r["fp"] = str(sel[2] < eval_iou) if inst else ""
                r["centroid"] = json.dumps(sel[3]) if inst and len(sel) > 3 and sel[3] is not None else ""
            rows.append(r)
    if mismatch:
        raise SystemExit(f"{mismatch} open-set frames rebuild to a different DET/NO_DET than the channel reported")
    # the per-frame model quantities; an open-set channel has one table per score threshold (scripts/isaac/calib_voxel_belief.py)
    ma_path = root / (f"model_area_{ma_name}.csv" if ma_name and (root / f"model_area_{ma_name}.csv").exists() else "model_area.csv")
    if not ma_path.exists():
        raise SystemExit(f"{ma_path} is missing: run scripts/isaac/calib_voxel_belief.py first")
    ma = {(m["episode_id"], m["camera_id"], int(m["pose_id"])): m for m in csv.DictReader(open(ma_path))}
    for r in rows:
        key = (r["episode_id"], r["camera_id"], int(r["pose_id"]))
        if key not in ma:
            raise SystemExit(f"{ma_path} has no frame {key}: rebuild it")
        r["a_model"], r["T_model"] = ma[key]["a_model"], ma[key]["T_model"]
        if "a_cell" not in ma[key]:
            raise SystemExit(f"{ma_path} has no per-point a_cell / T_cell: rebuild it with scripts/isaac/calib_voxel_belief.py")
        r["a_cell"], r["T_cell"] = ma[key]["a_cell"], ma[key]["T_cell"]
    return rows


def bounded(th: torch.Tensor, lo: float = LO, hi: float = HI) -> torch.Tensor:
    return lo + (hi - lo) * torch.sigmoid(th)


def r_of(a: torch.Tensor, eta_f, eta_t, a0, k, T=None) -> torch.Tensor:
    """spec 7: r = eta_f + (1 - eta_f) eta_t T h(a), h(a) = sigma(k ln(a / a0)) (T: the voxel belief's gate, 1 if
    None); this form cannot turn into 0 / 0 for a large k, and h(0) = 0 exactly."""
    a = torch.as_tensor(a, dtype=torch.float64)
    a0 = torch.as_tensor(a0, dtype=torch.float64)
    h = torch.where(a > 0, torch.sigmoid(k * (torch.log(torch.clamp(a, min=1e-300)) - torch.log(a0))), torch.zeros_like(a))
    if T is not None:
        h = h * torch.as_tensor(T, dtype=torch.float64)
    return eta_f + (1.0 - eta_f) * eta_t * h


def r_bar(a, eta_f, eta_t, a0, k, T=None) -> torch.Tensor:
    """[n] the cell average of r over the points q (a, T: [n, Q]), the probability of a DET (formulation §7.3)."""
    return r_of(a, eta_f, eta_t, a0, k, T).mean(dim=-1)


def nll(a, y, eta_f, eta_t, a0, k, T=None) -> torch.Tensor:
    r = r_bar(a, eta_f, eta_t, a0, k, T).clamp(1e-12, 1 - 1e-12)
    return -(y * torch.log(r) + (1 - y) * torch.log(1 - r)).sum()


def fit(a: np.ndarray, y: np.ndarray, a0: float | None, T: np.ndarray | None = None) -> dict:
    """eta_f, eta_t, k (and a0 if None) by maximum likelihood over every frame; a0 otherwise held at the value.

    eta_f enters every frame's r, as the floor under the in-view curve too, so the full likelihood estimates it from
    all frames. Its direct estimate from the out-of-view frames alone (spec 7.2's definition) is reported beside it:
    the two part where the curve misfits the in-view frames.
    """
    out = np.all(a <= 0, axis=-1)  # no point of the cell in view
    direct = float(np.clip(y[out].mean(), LO, HI)) if out.any() else LO
    A = torch.as_tensor(a, dtype=torch.float64)
    Y = torch.as_tensor(y, dtype=torch.float64)
    inv = lambda v, lo=LO, hi=HI: float(np.log((v - lo) / (hi - v)))  # noqa: E731
    la_lo, la_hi = np.log(A0_LO), np.log(A0_HI)
    start = [inv(float(np.clip(direct, 2 * LO, HI - LO))), inv(0.95), inv(3.0, K_LO, K_HI)]
    start += [inv(np.log(0.0025), la_lo, la_hi)] if a0 is None else []
    th = torch.tensor(start, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([th], lr=1.0, max_iter=500, tolerance_grad=1e-10, tolerance_change=1e-12,
                            line_search_fn="strong_wolfe")
    a0_of = (lambda: torch.exp(bounded(th[3], la_lo, la_hi))) if a0 is None else (lambda: a0)

    def closure():
        opt.zero_grad()
        loss = nll(A, Y, bounded(th[0]), bounded(th[1]), a0_of(), bounded(th[2], K_LO, K_HI), T)
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        p = {"false_positive_rate": float(bounded(th[0])), "detection_gain": float(bounded(th[1])),
             "h_slope": float(bounded(th[2], K_LO, K_HI)), "h_area0": float(a0_of()), "eta_f_direct": direct}
        p["nll"] = float(nll(A, Y, p["false_positive_rate"], p["detection_gain"], p["h_area0"], p["h_slope"], T))
    return p


def fp_share_in_view(rows: list[dict]) -> float:
    """Diagnostic only: the share of reported masks that are false positives with the target in view."""
    dets = [r for r in rows if r["status"] == "DET" and r["in_view"] == "True"]
    return float(np.mean([r["fp"] == "True" for r in dets])) if dets else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--a0", required=True,
                    help="'fit', or a value to hold a0 at (an oracle channel: its sensor_detector.min_normalized_area)")
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--min-mask-px", type=int, default=3, help="sensor_detector.open_set_min_mask_px")
    ap.add_argument("--eval-iou", type=float, default=0.25, help="sensor_detector.open_set_eval_iou")
    ap.add_argument("--mask-select", choices=["largest", "score"], default="largest",
                    help="sensor_detector.open_set_mask_select of the channel (SAM 3: score)")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--out-name", default="")
    a = ap.parse_args()
    root = Path(a.root)
    a0 = None if a.a0 == "fit" else float(a.a0)
    rows = load(root, a.min_score, a.min_mask_px, a.eval_iou, a.mask_select, a.out_name)
    Q = len(cell_offsets_cfg())
    A = np.array([json.loads(r["a_cell"]) for r in rows], dtype=float).reshape(len(rows), Q)  # [n, Q]
    TT = np.array([json.loads(r["T_cell"]) for r in rows], dtype=float).reshape(len(rows), Q)
    Am = A.mean(axis=1)  # the cell-mean area, for the reported bins
    Y = np.array([r["status"] == "DET" for r in rows], dtype=float)
    fp = np.array([r["fp"] == "True" for r in rows])
    eps = sorted({r["episode_id"] for r in rows})
    print(f"{root}: {len(rows)} frames from {len(eps)} calib episodes (min score {a.min_score}); out of view "
          f"{int((Am <= 0).sum())}, in view {int((Am > 0).sum())} (of the cell's {Q} points)")

    best = fit(A, Y, a0, TT)
    spec_nll = float(nll(torch.as_tensor(A), torch.as_tensor(Y), SPEC["false_positive_rate"], SPEC["detection_gain"],
                         SPEC["h_area0"], SPEC["h_slope"], TT))
    by_ep = {e: [i for i, r in enumerate(rows) if r["episode_id"] == e] for e in eps}
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(a.boot):
        idx = np.concatenate([by_ep[e] for e in rng.choice(eps, size=len(eps), replace=True)])
        p = fit(A[idx], Y[idx], a0, TT[idx])
        boots.append(p)
    keys = ["false_positive_rate", "detection_gain", "h_slope", "h_area0"]
    names = {"false_positive_rate": "eta_f", "detection_gain": "eta_t", "h_slope": "k", "h_area0": "a0"}
    ci = {}
    print("\nevent-only fit (diagnostic; DET / NO_DET alone)")
    print(f"{'constant':<8}{'spec':>9}{'fitted':>10}{'95% CI':>24}   (a0 {'fitted' if a0 is None else 'held at ' + str(a0)})")
    for k_ in keys:
        v = np.array([b[k_] for b in boots])
        v = v[np.isfinite(v)]
        ci[k_] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [float("nan")] * 2
        lo_, hi_ = {"h_slope": (K_LO, K_HI), "h_area0": (A0_LO, A0_HI)}.get(k_, (LO, HI))
        if k_ == "h_area0" and a0 is not None:
            print(f"{names[k_]:<8}{SPEC[k_]:>9.4g}{best[k_]:>10.4g}   (held)")
            continue
        edge = "  at the bound" if best[k_] <= lo_ + 1e-4 * lo_ or best[k_] >= hi_ - 1e-4 * hi_ else ""
        print(f"{names[k_]:<8}{SPEC[k_]:>9.4g}{best[k_]:>10.4g}   [{ci[k_][0]:.4g}, {ci[k_][1]:.4g}]{edge}")
    print(f"eta_f direct estimate (out-of-view DET rate, diagnostic): {best['eta_f_direct']:.4g}"
          f" vs fitted {best['false_positive_rate']:.4g}")
    print(f"bootstrap fits that failed: {sum(not np.isfinite(b['false_positive_rate']) for b in boots)}/{a.boot}")
    print(f"log-likelihood per frame: spec {-spec_nll / len(Y):.4f}   fitted {-best['nll'] / len(Y):.4f}")

    print(f"\n{'a bin':<22}{'n':>6}{'DET':>7}{'  of which FP':>14}{'r spec':>8}{'r fitted':>10}")
    bins = [(-1, 0, "a = 0 (out of view)")] + [(10**lo, 10**(lo + 0.5), f"({10**lo:.1e}, {10**(lo + 0.5):.1e}]")
                                              for lo in np.arange(-5, 0, 0.5)]
    for lo, hi, lab in bins:
        m = (Am <= 0) if hi == 0 else (Am > lo) & (Am <= hi)
        if m.sum() == 0:
            continue
        am = torch.as_tensor(A[m])
        rs = float(r_bar(am, SPEC["false_positive_rate"], SPEC["detection_gain"], SPEC["h_area0"], SPEC["h_slope"], TT[m]).mean())
        rf = float(r_bar(am, best["false_positive_rate"], best["detection_gain"], best["h_area0"], best["h_slope"], TT[m]).mean())
        print(f"{lab:<22}{int(m.sum()):>6}{Y[m].mean():>7.3f}{(fp & (Y > 0))[m].mean():>14.3f}{rs:>8.3f}{rf:>10.3f}")

    # kappa: the closed-form Gaussian of the true detections, a diagnostic and the start of the joint fit
    import bamos.config as _cfg
    grid = _cfg.load_config("configs/isaac.yaml")["planner"]["size_grid_m"]
    assert all(any(abs(v - g) < 1e-3 for g in grid) for v in TYPE_SIZE.values()), f"TYPE_SIZE {TYPE_SIZE} vs size_grid_m {grid}"
    D, Nv, S_, kep = kappa_samples(rows)
    kap = fit_kappa(D, Nv, S_)
    by_kep = {}
    for i, e in enumerate(kep):
        by_kep.setdefault(e, []).append(i)
    kb = []
    keps = sorted(by_kep)
    for _ in range(a.boot if keps else 0):
        idx = np.concatenate([by_kep[e] for e in rng.choice(keps, size=len(keps), replace=True)])
        kb.append(fit_kappa(D[idx], Nv[idx], S_[idx]))
    print(f"\nkappa closed form (diagnostic), from {len(S_)} true detections in {len(keps)} episodes (d = observed centroid - GT centre)")
    for k_ in ("kappa_beta", "kappa_lambda"):
        v = np.array([b[k_] for b in kb])
        ci[k_] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [float("nan")] * 2
        best[k_] = kap[k_]
        print(f"{k_:<13}{SPEC[k_]:>9.4g}{kap[k_]:>10.4g}   [{ci[k_][0]:.4g}, {ci[k_][1]:.4g}]")
    if len(S_):
        print(f"log kappa per true detection at the truth: spec {kappa_loglik(D, Nv, S_, SPEC['kappa_beta'], SPEC['kappa_lambda']):.3f}"
              f"   fitted {kappa_loglik(D, Nv, S_, kap['kappa_beta'], kap['kappa_lambda']):.3f}")
        for t in sorted(TYPE_SIZE):  # per type, as a diagnostic of the one shared (beta, lambda)
            m = np.array([TYPE_SIZE[t] == x for x in S_])
            if m.any():
                ft = fit_kappa(D[m], Nv[m], S_[m])
                print(f"  {t:<7} n={int(m.sum()):>4}  beta {ft['kappa_beta']:.3f}  lambda {ft['kappa_lambda']:.3f}")
    else:
        print("no true detection with a centroid: the joint fit starts kappa at the spec beta and lambda")
        best["kappa_beta"], best["kappa_lambda"] = SPEC["kappa_beta"], SPEC["kappa_lambda"]

    # the joint fit of all six constants: what the model uses
    J = joint_data(rows)
    all_idx = np.arange(len(rows))
    sep = {k_: best[k_] for k_ in keys + ["kappa_beta", "kappa_lambda"]}
    jb = fit_joint(J, all_idx, a0, sep)
    jboots = []
    for _ in range(a.boot):
        idx = np.concatenate([by_ep[e] for e in rng.choice(eps, size=len(eps), replace=True)])
        jboots.append(fit_joint(J, idx, a0, jb))
    jkeys = keys + ["kappa_beta", "kappa_lambda"]
    jnames = {**names, "kappa_beta": "beta", "kappa_lambda": "lambda"}
    bounds = {"h_slope": (K_LO, K_HI), "h_area0": (A0_LO, A0_HI), "kappa_beta": (B_LO, B_HI), "kappa_lambda": (L_LO, L_HI)}
    n_c = int(J["has_c"].sum())
    print(f"\njoint fit (used): {len(rows)} frames, {int(J['y'].sum())} DET, {n_c} with a centroid")
    print(f"{'constant':<8}{'spec':>9}{'separate':>10}{'joint':>10}{'95% CI':>24}")
    ci = {}
    for k_ in jkeys:
        v = np.array([b[k_] for b in jboots])
        v = v[np.isfinite(v)]
        ci[k_] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [float("nan")] * 2
        if k_ == "h_area0" and a0 is not None:
            print(f"{jnames[k_]:<8}{SPEC[k_]:>9.4g}{sep[k_]:>10.4g}{jb[k_]:>10.4g}   (held)")
            continue
        lo_, hi_ = bounds.get(k_, (LO, HI))
        edge = "  at the bound" if jb[k_] <= lo_ + 1e-4 * abs(lo_) or jb[k_] >= hi_ - 1e-4 * abs(hi_) else ""
        print(f"{jnames[k_]:<8}{SPEC[k_]:>9.4g}{sep[k_]:>10.4g}{jb[k_]:>10.4g}   [{ci[k_][0]:.4g}, {ci[k_][1]:.4g}]{edge}")
    nll_of = lambda q: float(joint_nll(J, all_idx, q["false_positive_rate"], q["detection_gain"], q["h_area0"],  # noqa: E731
                                        q["h_slope"], q["kappa_beta"], q["kappa_lambda"]))
    print(f"joint log-likelihood per frame: spec {-nll_of(SPEC) / len(rows):.4f}   separate {-nll_of(sep) / len(rows):.4f}"
          f"   joint {-jb['nll'] / len(rows):.4f}")
    with torch.no_grad():  # how the fitted mixture reads the detections with a centroid
        hc = J["has_c"]
        rq = r_of(torch.as_tensor(J["a"][hc]), jb["false_positive_rate"], jb["detection_gain"], jb["h_area0"], jb["h_slope"],
                  J["T"][hc])  # [n, Q]
        S_c = torch.as_tensor(J["S"][hc])
        var = (jb["kappa_lambda"] * S_c) ** 2
        res = torch.as_tensor(J["D"][hc]) - jb["kappa_beta"] * S_c[:, None, None] * torch.as_tensor(J["Nv"][hc])
        log_n = -1.5 * torch.log(2 * np.pi * var)[:, None] - 0.5 * (res * res).sum(dim=2) / var[:, None]
        log_true = torch.logsumexp(torch.log(torch.clamp(rq - jb["false_positive_rate"], min=1e-300)) + log_n, dim=1) - np.log(rq.shape[1])
        lfp = np.log(jb["false_positive_rate"]) + torch.as_tensor(J["logu"][hc])
        p_fp = torch.exp(lfp - torch.logaddexp(lfp, log_true)).numpy()
    lab_fp = np.array([rows[i]["fp"] == "True" for i in np.nonzero(hc)[0]])
    if n_c:
        print(f"posterior P(false positive) of the detections with a centroid: mean {p_fp.mean():.3f}; "
              f"labelled FP (IoU < eval-iou) {p_fp[lab_fp].mean() if lab_fp.any() else float('nan'):.3f}, "
              f"labelled true {p_fp[~lab_fp].mean() if (~lab_fp).any() else float('nan'):.3f}")
    best.update({k_: jb[k_] for k_ in jkeys})
    best["nll"] = jb["nll"]

    sfx = f"_{a.out_name}" if a.out_name else ""
    ov = " ".join(f"--override detection_likelihood.{k_}={best[k_]:.4g}" for k_ in keys)
    ov += f" --override planner.kappa_beta={best['kappa_beta']:.4g} --override planner.kappa_lambda={best['kappa_lambda']:.4g}"
    print(f"false-positive share of the reported masks with the target in view (diagnostic): {fp_share_in_view(rows):.3f}")
    if a.min_score > 0.0:
        ov += f" --override sensor_detector.open_set_min_score={a.min_score:g}"
    if a.mask_select != "largest":  # the channel the constants were fitted under, for the stage 3 runs
        ov += f" --override sensor_detector.open_set_mask_select={a.mask_select}"
    (root / f"overrides{sfx}.txt").write_text(ov + "\n")
    (root / f"fit{sfx}.json").write_text(json.dumps({
        "a0": "fitted" if a0 is None else a0, "min_score": a.min_score, "n_frames": len(rows), "n_episodes": len(eps),
        "fitted": {k_: best[k_] for k_ in keys + ["kappa_beta", "kappa_lambda"]}, "ci95": ci, "spec": SPEC,
        "n_true_detections": int(len(S_)), "eta_f_direct": best["eta_f_direct"],
        "joint_loglik_per_frame": {"spec": -nll_of(SPEC) / len(rows), "separate": -nll_of(sep) / len(rows),
                                   "joint": -jb["nll"] / len(rows)},
        "separate_fit": sep, "overrides": ov}, indent=1))
    print(f"\nwrote {root / f'fit{sfx}.json'}\n{ov}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
