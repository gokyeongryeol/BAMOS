"""Open-set detection channel: a text-prompted segmentation model in place of the oracle threshold detector.

The model runs in its own process (`detector_server/falcon_server.py` or `sam3_server.py`, in the detector's virtualenv) and speaks
the Isaac server's framing, so `DetectorClient` reuses that transport. The planner receives what the model
returned and nothing else: DET with one returned mask (the largest for Falcon, the highest-scoring for SAM 3;
sensor_detector.open_set_mask_select), NO_DET when no mask survives, MISSING when the
model reported an error for the frame (spec 2: MISSING updates no belief). Everything computed against the
ground truth here (IoU, centroid error, FN/FP flags) goes to `meta` for the evaluator and the failure
analysis, never to the planner.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

from bamos.environment.isaac_client import IsaacClient
from bamos.mapping.camera import CameraModel
from bamos.observation.types import DetectionStatus

# The task's text query per target type. It describes the rendered material (isaac_server/server.py PALETTE),
# which is the only appearance cue an open-vocabulary model has to go on.
FALCON_MODEL_ID = "tiiuae/Falcon-Perception"

# A frame the model errors on is MISSING. A run of them means the server is broken, not the frame, and the
# episode fails instead of quietly searching with no detector.
MAX_CONSECUTIVE_ERRORS = 3

# the text query per target type: the task specification handed to the robot (shape and colour, never the size)
OPEN_SET_QUERY = {**{f"cube{k}": "the red box" for k in (10, 20, 30)}, **{f"sphere{k}": "the red ball" for k in (10, 20, 30)}}


class DetectorClient(IsaacClient):
    """Same length-prefixed JSON + npz transport as the Isaac client; the server answers ping | detect."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5600, timeout: float = 600.0,
                 cache_dir: str | Path | None = None):
        super().__init__(host, port, timeout)
        # One cache per model, so two detectors never read each other's masks. Falcon Perception, the first
        # detector, keeps the top level so its existing entries stay valid.
        self.model = str(self.info.get("model", ""))
        sub = "" if self.model in ("", FALCON_MODEL_ID) else self.model.replace("/", "--")
        self.cache_dir = (Path(cache_dir) / sub if sub else Path(cache_dir)) if cache_dir else None
        self.consecutive_errors = 0

    def detect(self, rgb: np.ndarray, query: str, frame_key: str) -> tuple[np.ndarray, np.ndarray | None, dict]:
        """Masks [n, H, W] bool at the image resolution, their scores [n] (None if the model gives none, as
        Falcon Perception does), and the server's header.

        Detections are cached by the frame (scene, camera, pose) and the query, so every arm that visits a
        pose meets the same detector output there, as the synthetic corruption channel guarantees for its
        own errors. The key is the frame and not the image bytes: the ray-traced RGB is not bit-identical
        across renders of the same pose.
        """
        rgb = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
        key = hashlib.sha1((frame_key + "|" + query).encode()).hexdigest()[:20]
        cache = self.cache_dir / f"{key}.npz" if self.cache_dir else None
        scored = self.model not in ("", FALCON_MODEL_ID)
        if cache is not None and cache.exists():
            try:
                with np.load(cache) as z:
                    if not scored or "scores" in z:  # an entry of a scored model written without them is re-asked
                        masks = np.unpackbits(z["packed"], axis=-1, count=rgb.shape[1]).astype(bool).reshape(-1, *rgb.shape[:2])
                        return masks, (z["scores"] if "scores" in z else None), {"cached": True}
            except Exception:  # a partially written entry from a concurrent worker: ask the model again
                pass
        h, arrs = self.call({"cmd": "detect", "query": query}, {"rgb": rgb})
        masks = arrs["masks"].astype(bool) if arrs is not None and "masks" in arrs else np.zeros((0, *rgb.shape[:2]), bool)
        scores = arrs["scores"].astype(np.float32) if arrs is not None and "scores" in arrs else None
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix(f".{os.getpid()}.tmp.npz")
            extra = {"scores": scores} if scores is not None else {}
            np.savez_compressed(tmp, packed=np.packbits(masks, axis=-1), **extra)
            os.replace(tmp, cache)
        return masks, scores, h


def _world_centroid(mask: np.ndarray, depth: np.ndarray, camera: CameraModel, T: np.ndarray) -> np.ndarray | None:
    vs, us = np.nonzero(mask)
    d = depth[vs, us]
    ok = np.isfinite(d) & (d > 0) & (d <= camera.far)
    if ok.sum() < 3:
        return None
    pc = camera.backproject(np.stack([us[ok], vs[ok]], 1).astype(np.float64), d[ok])
    return (pc @ T[:3, :3].T + T[:3, 3]).mean(axis=0)


def open_set_observe(client: DetectorClient, query: str, frame_key: str, rgb: np.ndarray, depth: np.ndarray, ids: np.ndarray,
                     gt_mask: np.ndarray, oracle_det: bool, camera: CameraModel, T: np.ndarray, min_mask_px: int,
                     eval_iou: float, fp_instance_ids: np.ndarray, mask_select: str = "largest",
                     min_score: float = 0.0, record_instances: bool = False,
                     multi_mask: bool = False) -> tuple[DetectionStatus, np.ndarray | None, dict]:
    """(status, planner-visible mask, evaluator-only record) for one frame.

    min_score > 0 raises the detector's own score threshold: instances scoring below it are dropped before the
    DET / NO_DET decision. It needs a model that scores its instances (SAM 3); Falcon Perception gives none.
    """
    try:
        masks, scores, head = client.detect(rgb, query, frame_key)
        client.consecutive_errors = 0
    except RuntimeError as e:  # the server answered with an error for this frame: the detector did not run
        client.consecutive_errors += 1
        if client.consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            raise RuntimeError(f"open-set detector failed on {client.consecutive_errors} frames in a row: {e}") from e
        return DetectionStatus.MISSING, None, {"open_set": True, "missing": True, "error": str(e)[:200]}
    # ConnectionError / OSError propagate: a dead server must fail the episode, not turn every frame into MISSING
    areas = masks.reshape(len(masks), -1).sum(axis=1) if len(masks) else np.zeros(0, dtype=np.int64)
    ok = areas >= int(min_mask_px)
    n_below = 0
    if min_score > 0.0:
        if scores is None:
            raise ValueError(f"open_set_min_score={min_score} needs a detector that scores its instances ({client.model} does not)")
        n_below = int((ok & (scores < min_score)).sum())
        ok &= scores >= min_score
    keep = np.nonzero(ok)[0]
    rec: dict = {"open_set": True, "n_masks": int(len(masks)), "n_kept": int(len(keep)),
                 "cached": bool(head.get("cached", False))}
    if min_score > 0.0:
        rec["n_below_score"] = n_below
    if record_instances:
        # calibration only (scripts/calibrate_channel.py): every returned instance, so a score threshold or
        # a selection rule can be refitted offline. [area_px, score or -1, IoU with the GT mask, world centroid of
        # its valid depth pixels (None below n_min = 3, as kappa)]
        inter = masks.reshape(len(masks), -1) & gt_mask.reshape(1, -1) if len(masks) else np.zeros((0, 1), bool)
        union = masks.reshape(len(masks), -1) | gt_mask.reshape(1, -1) if len(masks) else np.zeros((0, 1), bool)
        ious = inter.sum(1) / np.maximum(union.sum(1), 1) if len(masks) else np.zeros(0)
        sc = scores if scores is not None else -np.ones(len(masks))
        cents = [_world_centroid(mk, depth, camera, T) for mk in masks]
        rec["instances"] = [[int(ar), round(float(s_), 4), round(float(i_), 4),
                             None if c_ is None else [round(float(x), 4) for x in c_]]
                            for ar, s_, i_, c_ in zip(areas, sc, ious, cents)]
    if not len(keep):
        rec["fn"] = bool(oracle_det)
        return DetectionStatus.NO_DET, np.zeros_like(gt_mask), rec
    # One mask per frame (spec 2). The rule must be GT-free and must not read the belief: choosing the mask the
    # belief likes best would let a false positive confirm itself. "score" takes the model's most confident
    # instance (SAM 3); "largest" is the rule for a model that scores nothing (Falcon). How often the rule
    # matters at all is the fraction of frames with n_kept > 1 (episodes.csv n_multi_mask_views).
    if mask_select == "score":
        if scores is None:
            raise ValueError(f"open_set_mask_select=score needs a detector that scores its instances ({client.model} does not)")
        sel = int(np.argmax(scores[keep]))
    else:
        sel = int(np.argmax(areas[keep]))
    mask = masks[keep[sel]]
    union = np.logical_or(mask, gt_mask).sum()
    iou = float(np.logical_and(mask, gt_mask).sum() / union) if union else 0.0
    rec.update(iou=round(iou, 4), mask_area_px=int(mask.sum()), fp=bool(iou < eval_iou), fn=False)
    c_rep, c_gt = _world_centroid(mask, depth, camera, T), _world_centroid(gt_mask, depth, camera, T)
    if c_rep is not None and c_gt is not None:
        rec["centroid_err_m"] = round(float(np.linalg.norm(c_rep - c_gt)), 4)
    if rec["fp"]:
        # a declaration on this frame counts as a wrong-instance stop when the mask sits on another object
        under = ids[mask]
        under = under[np.isin(under, fp_instance_ids)]
        if under.size and under.size >= 0.5 * mask.sum():
            vals, cnt = np.unique(under, return_counts=True)
            rec["wrong_instance_id"] = int(vals[int(np.argmax(cnt))])
    if multi_mask:
        # every kept instance for the planner, the selected one first; the evaluator's record above stays the selected
        # one's. rec["instance_masks"] is popped by the environment into PlannerObservation.instance_masks
        order = [keep[sel]] + [i for j, i in enumerate(keep) if j != sel]
        rec["instance_masks"] = [masks[i] for i in order]
        rec["n_planner_masks"] = len(order)
    return DetectionStatus.DET, mask, rec
