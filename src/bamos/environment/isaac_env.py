"""Isaac Sim client environment: observations rendered by the Isaac server, GT geometry from the shared analytic
scene description (candidate quality, solvability, target-only silhouette), with a per-scene render cache and
an analytic-depth consistency check recorded per frame."""
from __future__ import annotations

import hashlib
import io
import json
import os
import socket
import struct
from pathlib import Path

import numpy as np

from bamos.environment.analytic_env import AnalyticEnvironment
from bamos.evaluation.records import _jsonable
from bamos.mapping.camera import CameraModel


class IsaacClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 5555, timeout: float = 600.0):
        self.host, self.port = host, port
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.info = self.call({"cmd": "ping"})[0]

    def _recv(self, n):
        buf = b""
        while len(buf) < n:
            c = self.sock.recv(min(1 << 20, n - len(buf)))
            if not c:
                raise ConnectionError("isaac server closed")
            buf += c
        return buf

    def call(self, header: dict, arrays: dict | None = None):
        payload = b""
        if arrays:
            bio = io.BytesIO()
            np.savez(bio, **arrays)
            payload = bio.getvalue()
        hb = json.dumps(_jsonable(header)).encode()
        self.sock.sendall(struct.pack(">Q", len(hb)) + hb + struct.pack(">Q", len(payload)) + payload)
        (hl,) = struct.unpack(">Q", self._recv(8))
        h = json.loads(self._recv(hl).decode())
        (pl,) = struct.unpack(">Q", self._recv(8))
        arrs = None
        if pl:
            arrs = dict(np.load(io.BytesIO(self._recv(pl))))
        if not h.get("ok", False):
            raise RuntimeError(f"isaac server error: {h.get('error')}")
        return h, arrs

    def build_scene(self, scene_manifest: dict, scene_hash: str):
        return self.call({"cmd": "build_scene", "scene": scene_manifest, "scene_hash": scene_hash})[0]

    def render(self, scene_hash: str, camera: CameraModel, world_from_camera: np.ndarray, hide_instances=(), n_steps: int = 2):
        h, arrs = self.call({"cmd": "render", "scene_hash": scene_hash, "camera": camera.__dict__, "world_from_camera": world_from_camera.tolist(), "hide_instances": list(hide_instances), "n_steps": n_steps})
        return arrs["depth"].astype(np.float64), arrs["ids"].astype(np.int64), arrs["rgb"], h

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass


class IsaacEnvironment(AnalyticEnvironment):
    """Same scene/pool/navigation/GT as the analytic env; `_render` goes to Isaac Sim."""

    @staticmethod
    def wrap(env: AnalyticEnvironment, cfg: dict, client: IsaacClient) -> "IsaacEnvironment":
        env.__class__ = IsaacEnvironment
        env.client = client
        ic = cfg["environment"]["isaac"]
        env.validate = bool(ic.get("validate_against_analytic", True))
        env.cache_dir = Path(ic["render_cache_dir"]) if ic.get("render_cache_dir") else None
        env.scene_manifest = env.scene.to_manifest()
        env.scene_manifest.pop("evaluator_only", None)
        env.scene_hash = hashlib.sha1(json.dumps(_jsonable(env.scene_manifest), sort_keys=True).encode()).hexdigest()[:12]
        client.build_scene(env.scene_manifest, env.scene_hash)
        env.validation_stats = []
        env.render_ms = []
        return env

    def _render(self, camera: CameraModel, world_from_camera: np.ndarray):
        key = hashlib.sha1((self.scene_hash + json.dumps(camera.__dict__, sort_keys=True) + np.round(world_from_camera, 6).tobytes().hex()).encode()).hexdigest()[:16]
        cache = self.cache_dir / self.scene_hash / f"{key}.npz" if self.cache_dir else None
        cached = False
        if cache is not None and cache.exists():
            try:
                with np.load(cache) as z:
                    depth, ids, rgb = z["depth"].astype(np.float64), z["ids"].astype(np.int64), z["rgb"]
                cached = True
            except Exception:  # partially written / corrupt cache entry from a concurrent worker: re-render
                cached = False
        if not cached:
            depth, ids, rgb, h = self.client.render(self.scene_hash, camera, world_from_camera)
            self.render_ms.append(h.get("ms", 0.0))
            if self.validate:
                a_depth, a_ids = self.renderer.render(camera, world_from_camera)
                fin = np.isfinite(depth) & np.isfinite(a_depth)
                med = float(np.median(np.abs(depth[fin] - a_depth[fin]))) if fin.any() else float("nan")
                frac_bad = float(np.mean(np.abs(depth[fin] - a_depth[fin]) > 0.05)) if fin.any() else float("nan")
                tgt = self.scene.gt_target_instance_id
                iou_den = np.logical_or(ids == tgt, a_ids == tgt).sum()
                iou = float(np.logical_and(ids == tgt, a_ids == tgt).sum() / iou_den) if iou_den else 1.0
                if med > 0.02 or frac_bad > 0.02:
                    # one-frame lag after a camera move: render again and re-check
                    depth, ids, rgb, h = self.client.render(self.scene_hash, camera, world_from_camera, n_steps=3)
                    fin = np.isfinite(depth) & np.isfinite(a_depth)
                    med = float(np.median(np.abs(depth[fin] - a_depth[fin]))) if fin.any() else float("nan")
                    frac_bad = float(np.mean(np.abs(depth[fin] - a_depth[fin]) > 0.05)) if fin.any() else float("nan")
                    iou_den = np.logical_or(ids == tgt, a_ids == tgt).sum()
                    iou = float(np.logical_and(ids == tgt, a_ids == tgt).sum() / iou_den) if iou_den else 1.0
                self.validation_stats.append({"median_abs_depth_diff_m": med, "frac_depth_diff_gt_5cm": frac_bad, "target_mask_iou": iou, "isaac_target_px": int((ids == tgt).sum()), "analytic_target_px": int((a_ids == tgt).sum())})
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                tmp = cache.with_suffix(f".{os.getpid()}.tmp.npz")
                np.savez_compressed(tmp, depth=depth.astype(np.float32), ids=ids.astype(np.int32), rgb=rgb)
                os.replace(tmp, cache)  # atomic publish
        return depth, ids, rgb

    def close(self):
        pass
