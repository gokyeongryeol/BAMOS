"""Top-down layout drawing used by the frame and ceiling-view figures."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def scene_topdown(ax, manifest: dict, mark_target: bool = True):
    sc = manifest["scene"]
    W, D = sc["workspace_hi"][0], sc["workspace_hi"][1]
    ax.set_xlim(-0.2, W + 0.2)
    ax.set_ylim(-0.2, D + 0.2)
    ax.set_aspect("equal")
    for o in sc["objects"]:
        if o["role"] in ("floor", "wall"):
            continue
        col = {"anchor": "#c9a56a", "clutter": "#999999", "target": "#d62728"}.get(o["role"], "#333")
        for p in o["prims"]:
            if p["kind"] == "box":
                cx, cy = p["center"][:2]
                sx, sy = p["size"][:2]
                yaw = p.get("yaw", 0.0)
                c, s = np.cos(yaw), np.sin(yaw)
                corners = np.array([[-sx / 2, -sy / 2], [sx / 2, -sy / 2], [sx / 2, sy / 2], [-sx / 2, sy / 2]]) @ np.array([[c, s], [-s, c]]) + [cx, cy]
                ax.fill(corners[:, 0], corners[:, 1], color=col, alpha=0.6 if o["role"] != "target" else 1.0, lw=0)
            else:
                ax.add_patch(plt.Circle(p["center"][:2], p["radius"], color=col, alpha=0.6))
    if mark_target:
        gt = sc["evaluator_only"]["gt_target_center"]
        ax.plot(gt[0], gt[1], "*", color="#d62728", ms=12, mec="k")
