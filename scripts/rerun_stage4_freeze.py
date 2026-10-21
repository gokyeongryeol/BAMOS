"""Re-run stage 4 (README.md, "Isaac Sim pipeline"): freeze each detector condition's configuration for the test runs.

    uv run python scripts/rerun_stage4_freeze.py            # then commit configs/conditions/ before stage 5

Reads the stage 2 fits (results/raw/calib/<channel>/overrides*.txt) and the stage 3 selection
(results/raw/stage3/selection.json) and writes, under configs/conditions/:

    <condition>.channel     the detector channel's settings (S5 corruption, open-set mode, SAM 3 threshold)
    <condition>.constants   the fitted model constants (eta_f, eta_t, k, a0, kappa's beta and lambda)
    common.args             what every arm shares (the selected stop confirmation count N, the voxel belief's model)
    frozen.json             the commit the fits came from, the selection, and a summary of every fit

The channel and the constants are kept apart because the spec-constant reference runs share a condition's channel
and differ only in its constants. Commit the directory: that commit is the freeze, and stage 5 refuses to run
without it. The test predictions go into the results doc before stage 5.
"""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
from pathlib import Path

ORACLE_FAMILY = {  # condition -> channel overrides, calib directory
    "oracle": ([], "oracle"),
}
SAM3_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)


def tag(t: float) -> str:
    return f"t{round(100 * t):03d}"


def flags(items: list[str]) -> str:
    return " ".join(f"--override {x}" for x in items)


def constants_of(overrides_file: Path) -> str:
    """The model-constant overrides of a stage 2 fit, without its channel settings (the SAM 3 threshold)."""
    toks = overrides_file.read_text().split()
    pairs = [toks[i + 1] for i in range(0, len(toks) - 1, 2) if toks[i] == "--override"]
    keep = [p for p in pairs if not p.startswith("sensor_detector.")]
    return flags(keep)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="results/raw/calib")
    ap.add_argument("--stage3", default="results/raw/stage3")
    ap.add_argument("--out", default="configs/conditions")
    a = ap.parse_args()
    cal, out = Path(a.calib), Path(a.out)
    sel = json.loads((Path(a.stage3) / "selection.json").read_text())
    t_sel, n_sel = float(sel["sam3_threshold"]), int(sel["confirmations"])
    out.mkdir(parents=True, exist_ok=True)
    # a previous freeze's files (a removed condition's included) must not survive into this one
    for f in list(out.glob("*.channel")) + list(out.glob("*.constants")) + [out / "common.args", out / "frozen.json"]:
        f.unlink(missing_ok=True)

    conds: dict[str, dict] = {}
    for name, (chan, d) in ORACLE_FAMILY.items():
        conds[name] = {"channel": flags(chan), "overrides_file": cal / d / "overrides.txt", "fit": cal / d / "fit.json"}
    for t in SAM3_THRESHOLDS:
        conds[f"sam3_{tag(t)}"] = {
            "channel": flags(["sensor_detector.mode=open_set", f"sensor_detector.open_set_min_score={t:g}",
                              "sensor_detector.open_set_mask_select=score"]),
            "overrides_file": cal / "sam3" / f"overrides_{tag(t)}.txt", "fit": cal / "sam3" / f"fit_{tag(t)}.json"}

    missing = [str(c["overrides_file"]) for c in conds.values() if not c["overrides_file"].exists()]
    if missing:
        raise SystemExit(f"stage 2 fits missing: {missing}")
    summary = {}
    for name, c in conds.items():
        (out / f"{name}.channel").write_text(c["channel"] + "\n")
        (out / f"{name}.constants").write_text(constants_of(c["overrides_file"]) + "\n")
        fit = json.loads(c["fit"].read_text()) if c["fit"].exists() else {}
        summary[name] = {"channel": c["channel"], "constants": constants_of(c["overrides_file"]),
                         "fitted": fit.get("fitted"), "ci95": fit.get("ci95"),
                         "n_frames": fit.get("n_frames"), "n_true_detections": fit.get("n_true_detections")}
    # the voxel belief's fitted model (scripts/calib_voxel_belief.py) is shared by every condition and arm
    pr = json.loads((cal / "occupancy_model.json").read_text())
    (out / "common.args").write_text(flags([f"stop.confirmations={n_sel}",
                                            f"mapping.occupancy_model=[{pr['p0']:.6g},{pr['q_occ']:.6g},{pr['q_free']:.6g}]"]) + "\n")
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    (out / "frozen.json").write_text(json.dumps({
        "frozen_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "code_commit_of_fits": commit, "sam3_threshold": t_sel, "sam3_condition": f"sam3_{tag(t_sel)}",
        "confirmations": n_sel, "selection": {k: v for k, v in sel.items() if k != "success"},
        "conditions": summary}, indent=1))
    print(f"wrote {len(conds)} conditions to {out}/ (SAM 3 threshold {t_sel:g}, N = {n_sel})")
    print(f"next: git add {out} && git commit, write down the test predictions, then stage 5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
