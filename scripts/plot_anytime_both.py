"""Success within k views in Isaac Sim (top row, B = 60) and HSSD houses (bottom row, B = 200), per detector.

    python3 scripts/plot_anytime_both.py [--isaac results/raw/test] [--hssd results/hssd/val]
                                         [--out results/figures]

Writes <out>/anytime_both.{pdf,png} (Figure 2 of the paper, Appendix D).

k counts the active views (the initial view excluded) in both environments: Isaac active_robot_view_count, HSSD
active_views. Success within k: the episode succeeded and declared within k active views.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig_style as S  # noqa: E402

DETS = [("oracle", "a", "Oracle"), ("sam3", "bp", "SAM 3"), ("falcon", "fa", "Falcon-Perception")]
# style key -> (Isaac arm, HSSD arm)
ARMS = [("bamos", "bamos", "bamos"), ("scene", "scene", "scene_only"), ("d5_novoxel", "d5_novoxel", "target_only"),
        ("d2_nonodet", "d2_nonodet", "no_nodet"), ("d3_nok", "d3_nok", "kappa1")]
ENVS = [("isaac", "Isaac Sim", 60, 345), ("hssd", "HSSD houses", 200, 1248)]


def isaac(root: Path, prefix: str, arm: str) -> np.ndarray:
    rows = [r for r in csv.DictReader(open(root / f"{prefix}_{arm}" / "episodes.csv")) if r["status"] == "ok"]
    assert len(rows) == 345, (prefix, arm, len(rows))
    ok = lambda r: r["actual_success"] in ("1", "True", "true")
    return np.array([int(r["active_robot_view_count"]) if ok(r) else 10**6 for r in rows])


def hssd(root: Path, det: str, arm: str) -> np.ndarray:
    rows = [r for r in map(json.loads, open(root / f"{det}_{arm}" / "episodes.jsonl")) if r["status"] == "ok"]
    assert len(rows) == 1248, (det, arm, len(rows))
    return np.array([int(r["active_views"]) if r["success"] else 10**6 for r in rows])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--isaac", default="results/raw/test")
    ap.add_argument("--hssd", default="results/hssd/val")
    ap.add_argument("--out", default="results/figures")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S.apply(plt)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(2, 3, figsize=(S.WIDTH, 3.7), sharey=True)
    for row, (env, env_name, B, n) in enumerate(ENVS):
        ks = np.arange(B + 1)
        for col, (det, prefix, title) in enumerate(DETS):
            ax = axs[row, col]
            for i, (key, ia, ha) in enumerate(ARMS):
                v = isaac(Path(a.isaac), prefix, ia) if env == "isaac" else hssd(Path(a.hssd), det, ha)
                c = np.array([(v <= k).mean() for k in ks])
                print(env, det, key, f"@B {c[-1]:.3f}")
                name, colr, ls, mk = S.ARMS[key]
                name = name[0].upper() + name[1:]
                ax.plot(ks, c, color=colr, ls=ls, lw=2.0 if key == "bamos" else 1.3, marker=mk, ms=3.0,
                        markevery=(int(B * (0.04 * i + 0.02)), B // 4), label=name)
            ax.set_title(f"({'abcdef'[3 * row + col]}) {title}")
            ax.set_xlim(0, B)
            ax.set_ylim(0, 1)
            ax.set_xticks(np.linspace(0, B, 5).astype(int))
            ax.grid(alpha=0.3)
            if row == 1:
                ax.set_xlabel("views $k$")
        axs[row, 0].set_ylabel(f"{env_name}\nsuccess within $k$")
    fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="upper center", ncol=5, fontsize=6.0, frameon=False,
               handlelength=2.0, columnspacing=0.7, handletextpad=0.4, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=0.6)
    S.save(fig, out, "anytime_both")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
