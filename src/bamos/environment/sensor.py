"""GT threshold detector and the controlled perception-corruption channel.

The oracle channel is DET when the actual visible fraction and normalized area of the target exceed fixed
thresholds; the planner receives only DET/NO_DET and the mask, never these GT scalars.

`corrupt_detection` sits between that oracle channel and the planner (outline S5). It flips true
DET frames to NO_DET with probability `fn_rate`, promotes true NO_DET frames to DET with a spurious mask with
probability `fp_rate`, and can degrade the mask of a true DET. Everything the evaluator reads (GT quality,
`gt_detected`, episode solvability) is computed before this function runs and is never touched by it. With the
default all-zero rates the channel is the identity, so the oracle results reproduce bit for bit.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DetectorConfig:
    mode: str = "rendered_threshold"
    min_visible_fraction: float = 0.01
    min_normalized_area: float = 0.0001
    # --- controlled corruption (all zero => oracle channel) ---
    fn_rate: float = 0.0  # P(report NO_DET | oracle DET); the model counterpart is detection_likelihood.detection_gain
    fp_rate: float = 0.0  # P(report DET with a spurious mask | oracle NO_DET); counterpart is false_positive_rate
    fp_mask_source: str = "instance"  # "instance": a non-target object in view | "blob": a structureless disc
    # --- open-set channel (mode == "open_set"; environment/open_set.py) ---
    open_set_min_mask_px: int = 3  # returned masks smaller than this are dropped before the DET/NO_DET decision (3,
    #                                the mask-dilation step of S5 and the n_min of kappa)
    open_set_eval_iou: float = 0.25  # evaluator only: IoU with the GT mask below which a reported mask is an FP
    # which returned mask is the frame's one mask (spec 2): "largest" for a model that scores nothing (Falcon),
    # "score" (the highest-scoring) for one that does (SAM 3)
    open_set_mask_select: str = "largest"  # | "score"
    open_set_min_score: float = 0.0  # > 0 raises a scoring detector's instance threshold (SAM 3 serves >= 0.5)
    # True: all kept instances reach the planner (PlannerObservation.instance_masks, the selected one first; target_mask
    # is their union) and the DET update averages kappa over them with equal weights; False: one mask per frame
    open_set_multi_mask: bool = False

    def __post_init__(self):
        if self.mode not in ("rendered_threshold", "open_set"):
            raise ValueError(f"sensor_detector.mode={self.mode!r} must be 'rendered_threshold' or 'open_set'")
        if self.open_set_mask_select not in ("largest", "score"):
            raise ValueError(f"open_set_mask_select={self.open_set_mask_select!r} must be 'largest' or 'score'")
        if self.mode == "open_set" and self.corrupts:
            raise ValueError("the synthetic corruption channel applies to the oracle detector only, not to open_set")
        for name in ("fn_rate", "fp_rate"):
            v = float(getattr(self, name))
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{name}={v} outside [0,1]")
        if self.fp_mask_source not in ("instance", "blob"):
            raise ValueError(f"fp_mask_source={self.fp_mask_source!r} must be 'instance' or 'blob'")

    @property
    def corrupts(self) -> bool:
        return bool(self.fn_rate > 0.0 or self.fp_rate > 0.0)


def threshold_detect(quality: dict, cfg: DetectorConfig) -> bool:
    return bool(quality["visible_fraction"] >= cfg.min_visible_fraction and quality["normalized_area"] >= cfg.min_normalized_area and quality["area_px"] > 0)


# ------------------------------------------------------------------------------------------------ false positives
def _instance_mask(ids: np.ndarray, allowed_ids: np.ndarray, lo_px: int, hi_px: int, rng: np.random.Generator):
    """A whole non-target instance visible in this frame, uniformly among those of plausible size."""
    cands = []
    for i in np.asarray(allowed_ids, dtype=np.int64):
        n = int((ids == i).sum())
        if lo_px <= n <= hi_px:
            cands.append(int(i))
    if not cands:
        return None, None
    pick = int(cands[int(rng.integers(len(cands)))])
    return ids == pick, pick


def _blob_mask(shape: tuple[int, int], lo_px: int, hi_px: int, rng: np.random.Generator) -> np.ndarray:
    """A disc of plausible area at a random image position: a false positive with no scene structure behind it."""
    h, w = shape
    area = float(np.exp(rng.uniform(np.log(max(lo_px, 1)), np.log(max(hi_px, lo_px + 1)))))
    r = max(1.0, np.sqrt(area / np.pi))
    cy = float(rng.uniform(min(r, h / 2), max(h - r, h / 2)))
    cx = float(rng.uniform(min(r, w / 2), max(w - r, w / 2)))
    yy, xx = np.mgrid[0:h, 0:w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r


def corrupt_detection(
    det: bool,
    gt_mask: np.ndarray,
    ids: np.ndarray,
    cfg: DetectorConfig,
    rng: np.random.Generator | None = None,
    fp_instance_ids: np.ndarray | None = None,
) -> tuple[bool, np.ndarray, dict]:
    """(reported detection, planner-visible mask, noise record) for one frame.

    `det`/`gt_mask` are the oracle channel and are left untouched by the caller for evaluation. Both uniforms are
    drawn unconditionally so that the realization at a viewpoint does not depend on whether the frame happened to
    be a DET, which keeps the corruption identical across methods and configurations.
    """
    empty = np.zeros_like(gt_mask)
    if not cfg.corrupts or rng is None:
        return det, (gt_mask if det else empty), {}
    u_fn, u_fp = rng.random(), rng.random()
    if det:
        if u_fn < cfg.fn_rate:
            return False, empty, {"fn": True}
        return True, gt_mask, {}
    if u_fp < cfg.fp_rate:
        n_px = int(gt_mask.size)
        lo_px = max(4, int(np.ceil(cfg.min_normalized_area * n_px)))
        hi_px = max(lo_px + 1, int(0.25 * n_px))
        mask, inst = (None, None)
        if cfg.fp_mask_source == "instance" and fp_instance_ids is not None and len(fp_instance_ids):
            mask, inst = _instance_mask(ids, fp_instance_ids, lo_px, hi_px, rng)
        source = cfg.fp_mask_source
        if mask is None:  # no eligible instance in view: fall back to a structureless blob
            mask = _blob_mask(gt_mask.shape, lo_px, hi_px, rng)
            source = "blob" if cfg.fp_mask_source == "blob" else "blob_fallback"
        rec = {"fp": True, "fp_source": source, "mask_area_px": int(mask.sum())}
        if inst is not None:
            rec["wrong_instance_id"] = int(inst)
        return True, mask, rec
    return False, empty, {}
