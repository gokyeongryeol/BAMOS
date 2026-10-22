"""The coloured frames of figures: an episode's scene rendered by an Isaac server (walls, floor, anchors in their
realistic colours, servers/environment/isaac_server.py), while the episode itself may run on the analytic backend,
which shares the geometry but has no colour. Used by the figure scripts only; the experiments choose their backend in
the config."""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

from bamos.environment.isaac_env import IsaacClient


class IsaacFramer:
    def __init__(self, env, port: int | None = None, host: str = "127.0.0.1"):
        from bamos.evaluation.records import _jsonable

        self.client = IsaacClient(host, int(port or os.environ.get("RENDER_ISAAC_PORT", 5555)))
        manifest = env.scene.to_manifest()
        manifest.pop("evaluator_only", None)
        self.scene_hash = hashlib.sha1(json.dumps(_jsonable(manifest), sort_keys=True).encode()).hexdigest()[:12]
        self.client.build_scene(manifest, self.scene_hash)

    def rgb(self, camera, world_from_camera: np.ndarray) -> np.ndarray:
        """[H, W, 3] uint8 of the pose. Rendered twice: the first frame after a camera move can lag."""
        # more render steps let the real-time renderer accumulate and denoise (FIG_RENDER_STEPS, figures only)
        steps = int(os.environ.get("FIG_RENDER_STEPS", 3))
        _, _, rgb, _ = self.client.render(self.scene_hash, camera, np.asarray(world_from_camera), n_steps=steps)
        return np.asarray(rgb)[..., :3]

    def close(self):
        self.client.close()
