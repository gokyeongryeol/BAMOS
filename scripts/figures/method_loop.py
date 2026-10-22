"""Figure 2 of the paper: what BAMOS represents and what one non-detection does.

    python3 scripts/figures/method_loop.py [--out results/figures] [--copy DIR]

(a) the loop, in three columns: what the robot observes, what it believes, what it decides. The observation model r
    links the beliefs to the three decisions; the detector's event Z feeds the update, which writes the target belief.
(b) a toy, drawn top-down above its numbers: five candidate cells, three in plain view, one behind an occluder, one
    outside the field of view. One non-detection multiplies each cell by 1 - r and renormalises (Eq. 4), so the belief
    moves to the two cells the camera could not see.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig_style as S  # noqa: E402

BLUE, ORANGE, GREEN, GREYBOX = "#e3ecf7", "#fde8cf", "#e2f1df", "#efefef"
EDGE = {BLUE: "#5b86b8", ORANGE: "#d4893a", GREEN: "#5aa05a", GREYBOX: "#8a8a8a"}
LINE = "#555555"


def loop_panel(ax):
    """Boxes are sized to their widest text, so no label touches a border."""
    from matplotlib.patches import FancyBboxPatch
    H, GF, PADX = 1.25, 0.30, 0.3  # box height, column gap as a share of the box width, text-to-border margin (data units)
    Y0, Y1, Y2 = 5.55, 3.15, 0.75  # row centres; rows 1.15 apart so the labels between them breathe
    ax.set_ylim(-0.75, 7.25)
    ax.axis("off")
    ax.set_title("(a) One step of BAMOS")
    boxes = [(0, Y0, "RGB-D view", "from pose $q$", GREYBOX),
             (0, Y1, "detector", "mask or nothing", GREYBOX),
             (1, Y0, "occupancy belief", "occupied or free voxels", BLUE),
             (1, Y1, "target belief", "location and size", BLUE),
             (1, Y2, "observation model", "$r$: P(detect from $q$)", ORANGE),
             (2, Y0, "where to look", "most informative", GREEN),
             (2, Y1, "when to stop", "confirm, declare", GREEN),
             (2, Y2, "update belief", "$\\times r\\kappa$ or $\\times(1-r)$", GREEN)]
    texts = []
    for _, y, title, sub, _ in boxes:
        texts.append(ax.text(0, y + 0.24, title, ha="center", va="center", fontsize=S.FS - 2.4, fontweight="bold",
                             color=S.INK, zorder=3))
        texts.append(ax.text(0, y - 0.3, sub, ha="center", va="center", fontsize=S.FS - 2.7, color="#444444", zorder=3))
    # widest text as a share of the axes width (independent of the x limits), then solve for the box width W in
    # data units: L = (3 + 2 GF) W + 0.2 and W = wf L + 2 PADX
    fig = ax.figure
    fig.canvas.draw()
    wf = max(tx.get_window_extent().width for tx in texts) / ax.get_window_extent().width
    k = 3 + 2 * GF
    assert k * wf < 0.97, f"box text too wide for the panel ({k * wf:.2f})"
    W = (0.2 * wf + 2 * PADX) / (1 - k * wf)
    G = GF * W
    ax.set_xlim(0, k * W + 0.2)
    X = [0.1 + W / 2 + k * (W + G) for k in range(3)]
    X0, X1, X2 = X
    for k, (col, y, *_rest, fc) in enumerate(boxes):
        ax.add_patch(FancyBboxPatch((X[col] - W / 2, y - H / 2), W, H, boxstyle="round,pad=0.05,rounding_size=0.25",
                                    fc=fc, ec=EDGE[fc], lw=0.8, zorder=2))
        texts[2 * k].set_x(X[col]); texts[2 * k + 1].set_x(X[col])

    def arrow(p, q, rad=0.0):
        ax.annotate("", xy=q, xytext=p, zorder=1, arrowprops=dict(arrowstyle="-|>", lw=0.8, color=LINE, shrinkA=0,
                                                                  shrinkB=0, mutation_scale=7,
                                                                  connectionstyle=f"arc3,rad={rad}"))

    def note(x, y, t):
        ax.text(x, y, t, ha="center", va="center", fontsize=S.FS - 1.8, color=LINE, zorder=3,
                bbox=dict(fc="white", ec="none", pad=0.4))

    for x, t in ((X0, "observe"), (X1, "believe and predict"), (X2, "decide")):
        ax.text(x, 6.85, t, ha="center", va="center", fontsize=S.FS - 1.2, color="#777777", style="italic")
    e0, s1, e1, s2 = X0 + W / 2, X1 - W / 2, X1 + W / 2, X2 - W / 2
    top, bot = lambda y: y + H / 2 + 0.05, lambda y: y - H / 2 - 0.05  # box edges, including the rounding pad
    arrow((e0 + 0.08, Y0 - 0.2), (s1 - 0.08, Y0 - 0.2))
    ax.text((e0 + s1) / 2, Y0 + 0.02, "depth", ha="center", va="bottom", fontsize=S.FS - 2.4, color=LINE)
    arrow((X0, bot(Y0)), (X0, top(Y1)))
    # visibility runs from the occupancy belief straight into the observation model, around the target belief
    # (through the gap between the first two columns); the target belief enters the observation model from above
    xg = 0.5 * (e0 + s1)
    xo = s1 + 0.35
    ax.plot([xo, xo, xg, xg], [bot(Y0), 0.5 * (Y0 + Y1), 0.5 * (Y0 + Y1), Y2], color=LINE, lw=0.8, zorder=1,
            solid_joinstyle="miter")
    arrow((xg, Y2), (s1 - 0.04, Y2))
    ax.text(xg - 0.12, 0.5 * (Y1 + Y2) + 0.25, "visibility", ha="center", va="center", rotation=90,
            fontsize=S.FS - 2.4, color=LINE, zorder=3, bbox=dict(fc="white", ec="none", pad=0.3))
    arrow((X1, bot(Y1)), (X1, top(Y2)))
    # the detection model feeds all three decisions through one vertical trunk with right-angle branches
    xb = 0.5 * (e1 + s2)
    ax.plot([e1, xb, xb], [Y2, Y2, Y0], color=LINE, lw=0.8, zorder=1, solid_joinstyle="miter")
    for y in (Y0, Y1, Y2):
        arrow((xb, y), (s2 - 0.04, y))
    # the detector's event Z runs under the middle column to the update
    yz = -0.45
    ax.plot([X0, X0, X2], [bot(Y1), yz, yz], color=LINE, lw=0.8, zorder=1)
    arrow((X2, yz), (X2, bot(Y2)))
    note((X0 + X1) / 2, yz, "event $Z_q$")
    # the update writes the target belief: up out of the update box, left through the gap between the rows (crossing
    # the trunk with a white gap), and up into the target belief
    import matplotlib.patheffects as pe
    xu, xt, yg = s2 + 0.6, e1 - 0.6, 0.5 * (Y1 + Y2)
    halo = [pe.withStroke(linewidth=3.2, foreground="white")]
    ax.plot([xu, xu, xt], [top(Y2), yg, yg], color=LINE, lw=0.8, zorder=1.5, solid_joinstyle="miter", path_effects=halo)
    ax.annotate("", xy=(xt, bot(Y1)), xytext=(xt, yg - 0.004), zorder=1.5,
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color=LINE, shrinkA=0, shrinkB=0, mutation_scale=7))


def toy_panel(fig, gs):
    from matplotlib.patches import Polygon, Rectangle
    sub = gs.subgridspec(2, 1, height_ratios=[1.0, 1.25], hspace=0.12)
    ax = fig.add_subplot(sub[0])
    bx = fig.add_subplot(sub[1])
    r = np.array([0.9, 0.9, 0.05, 0.9, 0.006])
    status = ["seen", "seen", "hidden", "seen", "outside"]
    ticks = ["seen", "seen", "hidden", "seen", "out"]
    prior = np.full(5, 0.2)
    post = prior * (1 - r)
    post /= post.sum()
    xs = np.arange(5)
    # top: the scene, top-down; the camera below, the cells in a row above it
    cam = (1.5, -1.55)
    ax.add_patch(Polygon([cam, (-0.6, 1.2), (3.6, 1.2)], closed=True, fc="#fff3c4", ec="#e0b400", lw=0.6, zorder=0))
    ax.plot(*cam, marker="^", ms=6, color=S.INK, zorder=4)
    ax.text(cam[0] + 0.25, cam[1], "camera", fontsize=S.FS - 2, va="center", color=S.INK)
    ax.add_patch(Rectangle((1.62, -0.02), 0.76, 0.2, fc="#6b6b6b", ec="none", zorder=3))
    ax.text(2.0, -0.3, "occluder", fontsize=S.FS - 2.2, ha="center", va="center", color="#555555")
    for x, st in zip(xs, status):
        fc = {"seen": "#d6e6f5", "hidden": "#cfcfcf", "outside": "#ffffff"}[st]
        ax.add_patch(Rectangle((x - 0.3, 0.45), 0.6, 0.6, fc=fc, ec="#555555", lw=0.6, zorder=2))
    for x, rr in zip(xs, r):
        ax.text(x, 1.42, f"{rr:.2g}", ha="center", va="center", fontsize=S.FS - 2.2, color="#444444")
    ax.text(-0.42, 1.42, "$r$", ha="right", va="center", fontsize=S.FS - 1.5, color="#444444")
    ax.text(4.05, -0.35, "out of\nview", fontsize=S.FS - 2.2, ha="center", va="center", color="#555555")
    ax.set_xlim(-0.75, 4.6)
    ax.set_ylim(-1.85, 1.65)
    ax.set_aspect("equal")
    ax.set_anchor("N")
    ax.axis("off")
    ax.set_title("(b) One non-detection")
    # bottom: belief before and after, aligned under the cells
    w = 0.36
    bx.bar(xs - w / 2, prior, w, color="#c9c9c9", label="before")
    bx.bar(xs + w / 2, post, w, color="#1f5fa8", label="after one NO_DET")
    for x, p in zip(xs, post):
        if p < 0.1:
            continue
        bx.text(x + w / 2, p + 0.015, f"{p:.2f}", ha="center", va="bottom", fontsize=S.FS - 2.2, color="#1f5fa8")
    bx.set_xlim(-0.75, 4.6)
    bx.set_xticks(xs)
    bx.set_xticklabels(ticks, fontsize=S.FS - 2.5, rotation=35, ha="right", rotation_mode="anchor")

    bx.tick_params(axis="x", length=0, pad=2)
    bx.set_ylim(0, 0.62)
    bx.set_yticks([0, 0.2, 0.4])
    bx.set_ylabel("target belief", labelpad=2)
    bx.grid(axis="y")
    bx.legend(loc="upper left", frameon=False, fontsize=S.FS - 1.8, handlelength=1.0, borderaxespad=0.0, ncol=2,
              columnspacing=0.8, bbox_to_anchor=(0.0, 1.06))
    return post, [ax, bx]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--copy", default="", help="a second directory to copy the png to")
    ap.add_argument("--loop-only", action="store_true", help="panel (a) alone, as fig2_loop (the 4-page paper's Fig. 1)")
    ap.add_argument("--toy-only", action="store_true", help="panel (b) alone, as appB_toy (an appendix figure)")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S.apply(plt)
    if a.toy_only:
        fig = plt.figure(figsize=(S.WIDTH * 0.42, 2.15))
        post, tb = toy_panel(fig, fig.add_gridspec(1, 1)[0])
        tb[0].set_title("One non-detection")
        out = Path(a.out)
        S.save(fig, out, "appB_toy")
        if a.copy:
            shutil.copy(out / "appB_toy.png", Path(a.copy) / "appB_toy.png")
        print("toy posterior", np.round(post, 3))
        return 0
    if a.loop_only:
        fig = plt.figure(figsize=(S.WIDTH * 0.72, 1.6))
        la = fig.add_subplot(111)
        loop_panel(la)
        la.set_title("")
        out = Path(a.out)
        S.save(fig, out, "fig2_loop")
        if a.copy:
            shutil.copy(out / "fig2_loop.png", Path(a.copy) / "fig2_loop.png")
        return 0
    fig = plt.figure(figsize=(S.WIDTH, 2.15))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.85, 1], wspace=0.3)
    la = fig.add_subplot(gs[0])
    loop_panel(la)
    post, tb = toy_panel(fig, gs[1])
    S.divider(fig, la, tb)
    out = Path(a.out)
    S.save(fig, out, "fig2_model")
    if a.copy:
        shutil.copy(out / "fig2_model.png", Path(a.copy) / "fig2_model.png")
    print("toy posterior", np.round(post, 3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
