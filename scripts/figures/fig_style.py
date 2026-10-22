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


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
