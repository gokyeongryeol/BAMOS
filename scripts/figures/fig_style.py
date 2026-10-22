"""Shared look of the paper's figures: one palette per method, print-size fonts, light axes, direct labels.

Every arm keeps the same colour, line style and marker in every figure, so a reader who learns BAMOS = blue circle in
one figure can read the next without its legend.
"""
from __future__ import annotations

FS = 7.5  # body-size text at the 5.5 in print width
WIDTH = 5.5

# key -> (label, colour, line style, marker)
ARMS = {
    "bamos": ("BAMOS", "#1f5fa8", "-", "o"),
    "d5_novoxel": ("no occupancy belief", "#3a9a48", "-.", "s"),
    "scene": ("no target belief", "#e08214", ":", "^"),
    "reactive": ("coverage search", "#c0392b", (0, (5, 1.5)), "v"),
    "d2_nonodet": ("no non-detection update", "#7f7f7f", "--", "x"),
    "d3_nok": ("no mask position", "#7b4fa0", "-.", "D"),
}
GREY = "#9a9a9a"
INK = "#222222"


def apply(plt) -> None:
    plt.rcParams.update({
        "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS, "xtick.labelsize": FS - 0.5,
        "ytick.labelsize": FS - 0.5, "legend.fontsize": FS - 1, "axes.linewidth": 0.6, "lines.linewidth": 1.4,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titlelocation": "left",
        "axes.titleweight": "bold", "axes.titlepad": 4, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "grid.color": "#e3e3e3", "grid.linewidth": 0.6,
        "axes.labelpad": 5, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    })


def end_label(ax, x, y, text, color, dy=0.0, **kw):
    """A label at the right end of a line, in the line's colour (replaces a legend)."""
    ax.annotate(text, xy=(x, y), xytext=(4, dy), textcoords="offset points", va="center", ha="left",
                color=color, fontsize=FS - 1, **kw)


def divider(fig, left, right, color="#bdbdbd", lw=0.7):
    """A thin vertical rule halfway between two panels' outer boxes (labels included), so the panels read apart.

    `left` and `right` are an Axes or a list of Axes (e.g. a panel with its colorbar)."""
    from matplotlib.lines import Line2D
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()

    def box(axes):
        axes = axes if isinstance(axes, (list, tuple)) else [axes]
        bs = [a.get_tightbbox(rend).transformed(inv) for a in axes]
        return min(b.x0 for b in bs), max(b.x1 for b in bs), min(b.y0 for b in bs), max(b.y1 for b in bs)
    a, b = box(left), box(right)
    x = 0.5 * (a[1] + b[0])
    fig.add_artist(Line2D([x, x], [min(a[2], b[2]), max(a[3], b[3])], transform=fig.transFigure, color=color, lw=lw))
    return x


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
