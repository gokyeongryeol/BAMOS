"""Failure analysis (paper outline §5.3): assign every failed episode of a run to one hypothesised cause.

    uv run python scripts/isaac/analyze_failures.py --run results/raw/test/a_bamos --budget-run results/raw/test/a_s6_budget2x
    uv run python scripts/isaac/analyze_failures.py --run results/raw/test/b_bamos --oracle-run results/raw/test/b_oracle_isaac

The run must have been made with --steps. A failure is an episode that fails the main metric, success@B read at the
N-th confirmation of the stop rule (scripts/isaac/analysis.py; N from configs/isaac_conditions/frozen.json unless
--confirm is given); --metric final or ever picks the MAP-within-tau test at the last view or at any view instead. Every criterion reads only the logs, and each is a hypothesis to
verify: a category that no episode meets is reported as empty, not dropped. An episode can meet several
criteria; the table reports each criterion's count (non-exclusive) and one exclusive assignment by priority.

The thresholds are tied to constants the model or the evaluator already has and are fixed before any result
is read (outline §5.3). --sensitivity repeats the exclusive assignment with every free threshold halved and
doubled, for the appendix.

Oracle-side (priority O3 > O4 > O5 > O1)
  O3 occlusion       on robot NO_DET views with the target in the frustum but occluded (unoccluded
                     silhouette > 0, visible fraction < --occ-vis, the evaluator's min_gt_visible_fraction),
                     the GT bin lost at least a factor --occ-drop (2: one bit against the truth) of its mass.
  O4 ambiguity       the MAP is wrong and the belief knows it is not sure (pi of the MAP cell
                     < tau_pi), while the truth is still live (mass >= --amb-mass within tau of the GT centre).
  O5 confident error the MAP is wrong and the belief is sure of it (pi of the MAP cell >= tau_pi).
  O1 budget          the same episode succeeds in --budget-run (a longer budget); unknown without it.
Viewpoint discretisation is not a category: the benchmark builds only episodes with at least one pose meeting
the task quality thresholds, so it is excluded by construction. The report instead compares the number of such
poses (n_reachable_quality_poses) between failed and successful episodes.

Perception-specific (priority P2 > P1 > P3 > P4), for open-set runs and for runs of the synthetic corruption
channel (sensor_detector.fn_rate / fp_rate). A failure whose paired --oracle-run episode also
failed is not a perception failure: it gets the oracle-side category instead, tagged "shared".
  P2 false positive  a reported mask with IoU < open_set_eval_iou is followed by a confident wrong belief
                     (MAP off by > tau with mass >= tau_pi within tau of it), or the stop rule's N-th
                     confirmation falls on a view that fails the success test (a false declaration).
  P1 repeated FN     at least --fn-repeat frames where the oracle detects and the model reports NO_DET.
  P3 centroid        a true detection whose mask centroid lies more than tau from the GT mask centroid.
  P4 amplification   the paired oracle episode succeeded and some perception error occurred, but none of
                     P1-P3 fired: an error too small to name by itself that the sequential update carried.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import analysis as W  # noqa: E402
from bamos.evaluation.records import read_jsonl  # noqa: E402

CONFIRM = 1  # the stop rule's confirmation count N, set in main()

ORACLE_ORDER = ["O3", "O4", "O5", "O1"]
PERCEPTION_ORDER = ["P2", "P1", "P3", "P4"]
# the thresholds --sensitivity scales; tau and tau_pi are the model's own and stay fixed
FREE = ("occ_vis", "occ_drop", "amb_mass", "fn_repeat")


def truthy(v) -> bool:
    return str(v).strip().lower() in ("true", "1")


def load_rows(run: Path) -> dict[str, dict]:
    with open(run / "episodes.csv") as f:
        return {r["episode_id"]: r for r in csv.DictReader(f) if r["status"] == "ok"}


def read_off(steps: list[dict], k_max: int = W.B) -> tuple[bool, bool, bool]:
    """(success, false declaration, declared) at the N-th confirmation within k_max active views, as
    scripts/isaac/analysis.py reads success@B."""
    return W.outcome(W.episode(steps, k_max), CONFIRM)


def success(row: dict, steps: list[dict] | None, metric: str, k_max: int = W.B) -> bool:
    if metric == "success":
        if steps is None:
            raise SystemExit("--metric success needs the step logs (--steps)")
        return read_off(steps, k_max)[0]
    if metric == "final":
        return truthy(row["final_gt_map_center_correct"])
    if steps is None:
        raise SystemExit("--metric ever needs the step logs (--steps)")
    return any(s["gt_map_center_correct"] for s in steps if s["initialization_or_active"] == "active")


def oracle_flags(row: dict, steps: list[dict], th: dict) -> dict[str, bool | None]:
    drop, prev = 0.0, None
    for s in steps:
        m = float(s["gt_bin_posterior_mass"])
        occluded_in_view = (s["camera_type"] == "robot" and s["detection_status"] == "NO_DET"
                            and int(s["extra"].get("gt_iso_area_px", 0)) > 0
                            and float(s["gt_visible_fraction"]) < th["occ_vis"])
        if occluded_in_view and prev is not None and prev > 0 and m > 0:
            drop += math.log(m / prev)
        prev = m
    sure = float(row.get("final_map_prob") or 0.0) >= th["tau_pi"]  # the stop rule's own criterion
    return {
        "O3": drop <= -math.log(th["occ_drop"]),
        "O4": not sure and float(row.get("final_gt_mass_within_tau") or 0.0) >= th["amb_mass"],
        "O5": sure,
        "O1": None,  # filled in by the caller when a budget run is given
    }


def perception_flags(row: dict, steps: list[dict], th: dict) -> tuple[dict[str, bool], int]:
    noise = [s["extra"].get("noise_applied") or {} for s in steps]
    n_fn = sum(bool(n.get("fn")) for n in noise)
    n_fp = sum(bool(n.get("fp")) for n in noise)
    n_centroid = sum(("fp" in n and not n["fp"]) and float(n.get("centroid_err_m", 0.0)) > th["tau"] for n in noise)
    p2 = read_off(steps)[1]  # declared on a view that fails the success test
    seen_fp = False
    for s, n in zip(steps, noise):
        seen_fp |= bool(n.get("fp"))
        conf = s["rho_current_map"]
        if seen_fp and conf is not None and not (isinstance(conf, float) and math.isnan(conf)):
            if float(s["gt_center_error_m"]) > th["tau"] and float(conf) >= th["tau_pi"]:
                p2 = True
                break
    flags = {"P2": p2, "P1": n_fn >= th["fn_repeat"], "P3": n_centroid > 0, "P4": False}
    return flags, n_fn + n_fp + n_centroid


def classify(failed: list[tuple[str, dict, list[dict]]], th: dict, perception_run: bool,
             oracle_ok: dict[str, bool | None], budget_ok: dict[str, bool | None]):
    """Per-episode records, non-exclusive counts and exclusive counts under one set of thresholds."""
    out, nonexcl, excl = [], Counter(), Counter()
    for ep, row, st in failed:
        rec = {"episode_id": ep, "target_type": row["target_type"], "archetype": row["archetype"],
               "n_reachable_quality_poses": row.get("n_reachable_quality_poses", "")}
        flags = oracle_flags(row, st, th)
        flags["O1"] = budget_ok.get(ep)
        order, shared = ORACLE_ORDER, False
        if perception_run:
            pflags, n_err = perception_flags(row, st, th)
            ok = oracle_ok.get(ep)
            pflags["P4"] = bool(ok) and n_err > 0 and not (pflags["P1"] or pflags["P2"] or pflags["P3"])
            rec["n_perception_errors"] = n_err
            rec["oracle_success"] = "" if ok is None else ok
            if ok is False:
                shared = True  # the model fails here even with a perfect detector
            else:
                flags, order = {**pflags, **flags}, PERCEPTION_ORDER
        for k, v in flags.items():
            rec[k] = "" if v is None else bool(v)
            if v:
                nonexcl[k] += 1
        rec["category"] = next((k for k in order if flags.get(k)), "unassigned") + ("_shared" if shared else "")
        excl[rec["category"]] += 1
        out.append(rec)
    return out, nonexcl, excl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--oracle-run", default=None, help="paired oracle run, for an open-set --run")
    ap.add_argument("--budget-run", default=None, help="same arm with a longer view budget, for O1")
    ap.add_argument("--metric", choices=["success", "final", "ever"], default="success")
    ap.add_argument("--frozen", default="configs/isaac_conditions/frozen.json")
    ap.add_argument("--confirm", type=int, default=None, help="stop confirmations N (default: the frozen N)")
    ap.add_argument("--tau", type=float, default=0.2, help="success radius (spec 10)")
    ap.add_argument("--tau-pi", type=float, default=None, help="stop threshold (spec 10); default stop.tau_pi")
    ap.add_argument("--occ-vis", type=float, default=None, help="O3: default evaluation.min_gt_visible_fraction")
    ap.add_argument("--occ-drop", type=float, default=2.0, help="O3: factor the GT bin must lose on occluded views")
    ap.add_argument("--amb-mass", type=float, default=0.1, help="O4: mass within tau of the GT that keeps it live")
    ap.add_argument("--fn-repeat", type=int, default=2, help="P1: false negatives that count as repeated")
    ap.add_argument("--sensitivity", action="store_true", help="also report every free threshold at x0.5 and x2")
    a = ap.parse_args()
    global CONFIRM
    frozen = json.loads(Path(a.frozen).read_text()) if Path(a.frozen).exists() else {}
    CONFIRM = a.confirm if a.confirm is not None else int(frozen.get("confirmations", 1))
    if a.tau_pi is None:  # the stop rule's own threshold, from the config
        from bamos.config import load_config
        a.tau_pi = float(load_config("configs/isaac.yaml")["stop"]["tau_pi"])
    if a.occ_vis is None:  # tied to the evaluator's own threshold (outline 5.3)
        from bamos.config import load_config
        a.occ_vis = float(load_config("configs/isaac.yaml")["evaluation"]["min_gt_visible_fraction"])
    th = {"tau": a.tau, "tau_pi": a.tau_pi, "occ_vis": a.occ_vis, "occ_drop": a.occ_drop,
          "amb_mass": a.amb_mass, "fn_repeat": a.fn_repeat}

    run = Path(a.run)
    rows = load_rows(run)

    def steps_of(r: Path, ep: str) -> list[dict]:
        p = r / "steps" / f"{ep}.jsonl"
        if not p.exists():
            raise SystemExit(f"missing {p}: the run must be made with --steps")
        return read_jsonl(p)

    def paired(other: str | None, k_max: int = W.B) -> dict[str, bool | None]:
        if other is None:
            return {}
        orows = load_rows(Path(other))
        return {ep: success(o, steps_of(Path(other), ep) if a.metric != "final" else None, a.metric, k_max)
                for ep, o in orows.items() if ep in rows}

    failed, n_quality = [], {True: [], False: []}
    perception_run = False
    for ep, row in sorted(rows.items()):
        st = steps_of(run, ep)
        # the oracle channel logs an empty noise record on every frame
        perception_run |= any(bool(s["extra"].get("noise_applied")) for s in st)
        ok = success(row, st, a.metric)
        if row.get("n_reachable_quality_poses"):
            n_quality[ok].append(int(row["n_reachable_quality_poses"]))
        if not ok:
            failed.append((ep, row, st))
    # the budget run is read within its own, longer budget (O1: fails within B, succeeds within 2B)
    oracle_ok, budget_ok = paired(a.oracle_run), paired(a.budget_run, 2 * W.B)
    out, nonexcl, excl = classify(failed, th, perception_run, oracle_ok, budget_ok)

    kind = "perception-error" if perception_run else "oracle"
    print(f"{run}: {kind}, metric={a.metric} (N = {CONFIRM}), {len(rows)} episodes, {len(failed)} failures")
    keys = (PERCEPTION_ORDER + ORACLE_ORDER) if perception_run else ORACLE_ORDER
    print(f"{'criterion':<20}{'meets (non-excl.)':>18}{'assigned':>10}")
    for k in keys:
        print(f"{k:<20}{('n/a' if k == 'O1' and a.budget_run is None else nonexcl[k]):>18}{excl[k]:>10}")
    for k in sorted(c for c in excl if c not in keys):
        print(f"{k:<20}{'':>18}{excl[k]:>10}")
    # discretisation as a covariate instead of a category
    for ok, name in ((False, "failed"), (True, "succeeded")):
        v = n_quality[ok]
        if v:
            print(f"n_reachable_quality_poses, {name}: median {statistics.median(v):g}, min {min(v)}, n {len(v)}")

    if a.sensitivity:
        table = [("baseline", excl)]
        for name in FREE:
            if name == "fn_repeat" and not perception_run:
                continue
            for f in (0.5, 2.0):
                v = th[name] * f
                if name == "fn_repeat":
                    v = max(1, round(v))
                if name == "occ_vis":
                    v = min(v, 1.0)
                _, _, e = classify(failed, {**th, name: v}, perception_run, oracle_ok, budget_ok)
                table.append((f"{name} {v:g}", e))
        cats = sorted(set(keys).union(*(e.keys() for _, e in table)))
        print("\nsensitivity: exclusive counts with one free threshold scaled")
        print(f"{'threshold':<20}" + "".join(f"{c:>18}" for c in cats))
        for label, e in table:
            print(f"{label:<20}" + "".join(f"{e[c]:>18}" for c in cats))

    if out:
        fields = list(dict.fromkeys(f for r in out for f in r))
        with open(run / "failures.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(out)
        print(f"wrote {run / 'failures.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
