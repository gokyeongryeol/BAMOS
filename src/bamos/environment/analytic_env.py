"""Analytic (exact ray-cast) environment: GT geometry, navigation oracle, oracle threshold detector.
The Isaac client subclasses it and replaces `_render`."""
from __future__ import annotations

import json
import zlib

import numpy as np

from bamos.environment.analytic_renderer import AnalyticRenderer
from bamos.environment.base import Environment, EvaluatorGroundTruth
from bamos.environment.pathing import NavigationOracle
from bamos.environment.scene import SceneDescription
from bamos.environment.sensor import DetectorConfig, corrupt_detection, threshold_detect
from bamos.mapping.camera import CameraModel, inverse_transform
from bamos.observation.types import DetectionStatus, PlannerObservation
from bamos.planning.candidates import CandidatePool


class AnalyticEnvironment(Environment):
    def __init__(self, scene: SceneDescription, robot_camera: CameraModel, pool: CandidatePool, nav: NavigationOracle, robot_start_pose_id: int, detector: DetectorConfig, gt: EvaluatorGroundTruth | None = None, corruption_seed: int = 0):
        self.scene = scene
        self.robot_camera = robot_camera
        self.pool = pool
        self.nav = nav
        self.robot_start_pose_id = robot_start_pose_id
        self.detector = detector
        self.renderer = AnalyticRenderer(scene)
        self._gt = gt
        self._capture_counter = 0
        self._last_gt_quality: dict[str, dict] = {}
        self.corruption_seed = int(corruption_seed)
        # objects a false positive may be pinned on: real objects, never the floor, the walls or the target itself
        # open-set channel (sensor_detector.mode == "open_set"): set by build_env_for_spec
        self.open_set_client = None
        self.open_set_query = ""
        self._fp_instance_ids = np.array([o.instance_id for o in scene.objects if o.role in ("clutter", "anchor")], dtype=np.int64)

    def _corruption_rng(self, camera_id: str, pose_id: int | None) -> np.random.Generator | None:
        """One stream per (episode, viewpoint). Independent of the visit order and of the method, so every method
        and every model configuration meets the same detector errors at the same viewpoint (outline S5)."""
        if not self.detector.corrupts:
            return None
        cam_key = int(zlib.crc32(camera_id.encode())) & 0xFFFFFFFF
        return np.random.default_rng([self.corruption_seed, 0 if pose_id is None else int(pose_id) + 1, cam_key])

    def _render(self, camera: CameraModel, world_from_camera: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Returns (depth, instance id image, rgb or None) at the true pose."""
        depth, ids = self.renderer.render(camera, world_from_camera)
        return depth, ids, None

    def _make_observation(self, camera_id: str, camera_type: str, camera: CameraModel, T_true: np.ndarray, rng: np.random.Generator, pose_id: int | None, step_index: int) -> PlannerObservation:
        depth, ids, rgb = self._render(camera, T_true)
        tgt_id = self.scene.gt_target_instance_id
        gt_mask = ids == tgt_id
        # GT quality: visible target pixels (rendered) over the target-only silhouette (analytic, occluders removed)
        sil = self.renderer.target_silhouette(camera, T_true)
        a_iso = int(sil.sum())
        a_gt = int(gt_mask.sum())
        quality = {
            "visible_fraction": (a_gt / a_iso) if a_iso > 0 else 0.0,
            "area_px": a_gt,
            "iso_area_px": a_iso,
            "normalized_area": a_gt / float(camera.n_pixels),
            "truncated": bool(sil[0, :].any() or sil[-1, :].any() or sil[:, 0].any() or sil[:, -1].any()),
        }
        det = threshold_detect(quality, self.detector)  # oracle channel; the evaluator keeps this value
        if self.detector.mode == "open_set":
            from bamos.environment.open_set import open_set_observe  # the module imports the Isaac client, which imports this one

            if rgb is None or self.open_set_client is None:
                raise RuntimeError("sensor_detector.mode=open_set needs the Isaac backend (RGB) and a detector client")
            frame_key = getattr(self, "scene_hash", "") + json.dumps(camera.__dict__, sort_keys=True) + np.round(T_true, 6).tobytes().hex()
            status, mask, noise = open_set_observe(
                self.open_set_client, self.open_set_query, frame_key, rgb, depth, ids, gt_mask, det, camera, T_true,
                self.detector.open_set_min_mask_px, self.detector.open_set_eval_iou, self._fp_instance_ids,
                self.detector.open_set_mask_select, self.detector.open_set_min_score,
                getattr(self, "open_set_record_instances", False), self.detector.open_set_multi_mask)
        else:
            reported, mask, noise = corrupt_detection(det, gt_mask, ids, self.detector, self._corruption_rng(camera_id, pose_id), self._fp_instance_ids)
            status = DetectionStatus.DET if reported else DetectionStatus.NO_DET
        instance_masks = noise.pop("instance_masks", None)
        if instance_masks is not None and len(instance_masks) > 1:
            mask = np.logical_or.reduce(np.stack(instance_masks))  # the union: no reported surface counts as an occluder
        else:
            instance_masks = None
        meta = {"gt_mask_area_px": int(gt_mask.sum()), "noise_applied": noise}
        if "wrong_instance_id" in noise:  # the evaluator counts a declaration on this frame as a wrong-instance stop
            meta["wrong_instance_id"] = noise["wrong_instance_id"]
        self._capture_counter += 1
        cap = f"{camera_id}_{self._capture_counter:04d}"
        obs = PlannerObservation(camera_id, camera_type, cap, camera, T_true, depth, mask, status, True, rgb, pose_id, meta,
                                 instance_masks)
        q = dict(quality)
        q["true_world_from_camera"] = T_true
        q["gt_detected"] = det
        self._last_gt_quality[cap] = q
        return obs

    def observe_robot(self, pose_id: int, rng: np.random.Generator, step_index: int) -> PlannerObservation:
        return self._make_observation("robot", "robot", self.robot_camera, self.pool.world_from_camera[pose_id], rng, int(pose_id), step_index)

    def path_costs(self, from_pose_id: int, to_pose_ids: np.ndarray) -> np.ndarray:
        return self.nav.path_costs(self.pool.positions[from_pose_id], self.pool.positions[to_pose_ids])

    def ground_truth(self) -> EvaluatorGroundTruth:
        return self._gt

    def gt_quality_of_observation(self, obs: PlannerObservation) -> dict:
        return self._last_gt_quality[obs.capture_id]

    def gt_quality_many(self, pose_ids: np.ndarray) -> list[dict]:
        """Evaluator-only batch GT quality (episode solvability); poses whose frustum cannot contain the target are skipped."""
        out = []
        c = np.asarray(self.scene.gt_target_center)
        rad = 0.5 * np.linalg.norm(np.asarray(self.scene.gt_target_extent))
        cam = self.robot_camera
        empty = {"visible_fraction": 0.0, "area_px": 0, "iso_area_px": 0, "normalized_area": 0.0, "truncated": False}
        for pid in pose_ids:
            T = self.pool.world_from_camera[pid]
            Ti = inverse_transform(T)
            pc = Ti[:3, :3] @ c + Ti[:3, 3]
            if pc[2] + rad < cam.near or pc[2] - rad > cam.far:
                out.append(dict(empty))
                continue
            z = max(pc[2], 1e-6)
            u = cam.fx * pc[0] / z + cam.cx
            v = cam.fy * pc[1] / z + cam.cy
            margin = rad * cam.fx / z + 2
            if u < -margin or u > cam.width + margin or v < -margin or v > cam.height + margin:
                out.append(dict(empty))
                continue
            q = self.renderer.target_quality(cam, T)
            out.append({k: v for k, v in q.items() if k != "mask"})
        return out
