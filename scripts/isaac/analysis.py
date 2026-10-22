"""The success definition, the anytime curve and the paired bootstrap the stage 6 analyses share (scripts/isaac/analyze_stage5.py,
scripts/isaac/analyze_stage8.py, scripts/isaac/analyze_stage9_10.py, scripts/isaac/analyze_stage13.py, scripts/isaac/select_threshold.py, scripts/isaac/analyze_failures.py, the figure
scripts); a library, run by none of them directly.

Every arm runs with the stop rule acting (stop.mode=joint): an episode ends at its declaration, the present view
that confirms S_hat = 1 (planning/runner.py, spec 10), logged as declared_stop. success@B: the episode declared
within B active views and the MAP was within tau of the target at the declaration (and the mask not on another
object). How well the declared view shows the target is reported beside it (shown@declare: the target not cut by
the image edge and at least half visible), not part of the test. success@k is the anytime curve: declared correctly
within k active views; its mean over k = 0..B is the AUC.

All comparisons are paired on the episodes every listed arm completed.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

def _budget() -> int:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from bamos.config import load_config
    return int(load_config("configs/isaac.yaml")["experiment"]["additional_robot_view_budget"])


B = _budget()  # the active view budget of the frozen config (success@k runs over k = 0..B)


def load_eval(cfg_path="configs/isaac.yaml"):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from bamos.config import load_config
    ev = load_config(cfg_path)["evaluation"]
    return ev["max_center_error_m"]


TAU = load_eval()


def passes(s: dict) -> bool:
    return s["gt_center_error_m"] <= TAU and "wrong_instance_id" not in (s["extra"].get("noise_applied") or {})


def fire_step(seq: list[dict], n: int, radius: float = 0.2) -> int | None:
    """Index into seq of the declaration under StopTracker (planning/stopping.py) with n confirmations: the
    n-th S_hat = 1 step whose MAP lies within radius of the MAP at the first S_hat = 1 step of that location."""
    anchors: list[list] = []
    for i, s in enumerate(seq):
        if not s["declared_stop"]:
            continue
        x = np.asarray(s["map_center"], dtype=float)
        for a in anchors:
            if np.linalg.norm(x - a[0]) <= radius + 1e-9:
                a[1] += 1
                if a[1] >= n:
                    return i
                break
        else:
            anchors.append([x, 1])
            if n <= 1:
                return i
    return None


CONFIRM = 1  # N of the stop rule; the analyses set it from configs/isaac_conditions/frozen.json


def episode(steps: list[dict], k_max: int) -> dict:
    init = [s for s in steps if s["initialization_or_active"] == "initialization"]
    act = [s for s in steps if s["initialization_or_active"] == "active"]
    seq = ([init[-1]] if init else []) + act[:k_max]  # index 0: state after the shared initialisation
    fire = fire_step(seq, CONFIRM, TAU)
    ok = fire is not None and passes(seq[fire])
    anytime = np.zeros(k_max + 1)  # success@k: declared correctly within k active views
    if ok:
        anytime[fire:] = 1.0
    noise = [s["extra"].get("noise_applied") or {} for s in steps]
    return {
        # what any other confirmation count needs, so a sweep does not reread the logs
        "seq_decl": [s for s in ({"declared_stop": q["declared_stop"], "map_center": q["map_center"]} for q in seq)],
        "seq_pass": [passes(q) for q in seq],
        "anytime": anytime,
        "declared": fire is not None,
        "success": ok,
        "false_decl": fire is not None and not passes(seq[fire]),
        "fire_step": fire,
        "shown": fire is not None and not seq[fire]["gt_truncated"] and seq[fire]["gt_visible_fraction"] >= 0.5,
        "poses": [s["selected_pose_id"] for s in act],
        "n_frames": len(steps),
        "n_multi": sum(int(n.get("n_kept", 0)) > 1 for n in noise),
        "n_fp": sum(bool(n.get("fp")) for n in noise),
        "n_fn": sum(bool(n.get("fn")) for n in noise),
        "n_missing": sum(s["detection_status"] == "MISSING" for s in steps),
        "n_open_set_det": sum("fp" in n for n in noise),
    }


def outcome(x: dict, n: int) -> tuple[bool, bool, bool]:
    """(success, false declaration, declared) of one episode under n confirmations."""
    i = fire_step(x["seq_decl"], n, TAU)
    if i is None:
        return False, False, False
    return x["seq_pass"][i], not x["seq_pass"][i], True


def load_arm(root: Path, tag: str, k_max: int = B) -> dict[str, dict] | None:
    d = root / tag
    if not (d / "episodes.csv").exists():
        return None
    ok = {r["episode_id"]: r for r in csv.DictReader(open(d / "episodes.csv")) if r["status"] == "ok"}
    out = {}
    for ep, row in ok.items():
        p = d / "steps" / f"{ep}.jsonl"
        if p.exists():
            out[ep] = {**episode([json.loads(l) for l in open(p) if l.strip()], k_max), "row": row}
    return out


def boot(x: np.ndarray, n: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_diff(a: np.ndarray, b: np.ndarray, n: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    d = b[idx].mean(axis=1) - a[idx].mean(axis=1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def mcnemar(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    b01, b10 = int(((a == 0) & (b == 1)).sum()), int(((a == 1) & (b == 0)).sum())
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    p = 2.0 * sum(math.comb(n, i) for i in range(min(b01, b10) + 1)) / 2.0**n
    return b01, b10, min(1.0, p)


def table(title: str, arms: list[tuple[str, str]], data: dict, eps: list[str]) -> None:
    ref_tag = arms[0][0]
    print(f"\n## {title}  (n = {len(eps)} paired episodes)")
    print(f"{'arm':<26}{'success@B [95% CI]':>24}{'false decl':>11}{'no decl':>9}{'shown':>7}{'AUC':>7}{'views':>6}"
          f"{'  vs ' + ref_tag + ': delta [95% CI], McNemar p':>44}")
    if ref_tag not in data:
        print(f"  (reference {ref_tag} not available: no paired comparison)")
    ref = np.array([data[ref_tag][e]["success"] for e in eps], dtype=float) if ref_tag in data else None
    for tag, name in arms:
        if tag not in data:
            print(f"{name:<26}  (not run)")
            continue
        d = [data[tag][e] for e in eps]
        s = np.array([x["success"] for x in d], dtype=float)
        lo, hi = boot(s)
        fd = np.mean([x["false_decl"] for x in d])
        nd = np.mean([not x["declared"] for x in d])
        anyt = np.array([x["anytime"] for x in d])
        shown = np.mean([x["shown"] for x in d if x["declared"]]) if any(x["declared"] for x in d) else float("nan")
        views = np.median([x["fire_step"] for x in d if x["declared"]]) if any(x["declared"] for x in d) else float("nan")
        cmp = ""
        if tag != ref_tag and ref is not None:
            b01, b10, p = mcnemar(ref, s)
            dlo, dhi = boot_diff(ref, s)
            cmp = f"{s.mean() - ref.mean():+.3f} [{dlo:+.3f},{dhi:+.3f}]  p={p:.2g} ({b10}/{b01})"
        print(f"{name:<26}{s.mean():>8.3f} [{lo:.3f},{hi:.3f}]{fd:>11.3f}{nd:>9.3f}{shown:>7.3f}"
              f"{anyt.mean():>7.3f}{views:>6.0f}{cmp:>44}")

