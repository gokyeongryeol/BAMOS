"""The paper's method figure: the BAMOS loop, in three columns: what the robot observes, what it believes, what it
decides. The observation model r links the beliefs to the three decisions; the detector's event Z feeds the update,
which writes the target belief.

    python3 scripts/figures/method_loop.py [--out results/figures] [--copy DIR]    # writes fig2_loop.{png,pdf}
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--copy", default="", help="a second directory to copy the png to")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S.apply(plt)
    fig = plt.figure(figsize=(S.WIDTH * 0.72, 1.6))
    la = fig.add_subplot(111)
    loop_panel(la)
    la.set_title("")
    out = Path(a.out)
    S.save(fig, out, "fig2_loop")
    if a.copy:
        shutil.copy(out / "fig2_loop.png", Path(a.copy) / "fig2_loop.png")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
