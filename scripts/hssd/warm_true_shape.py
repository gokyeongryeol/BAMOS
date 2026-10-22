"""Fill the true-geometry surface cache (hssd.cache_dir/true_shape) for the val scenes, one scene per teleport server.

    uv run python scripts/hssd/warm_true_shape.py --port 5950 --shard 0/16
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/hssd.yaml")
    ap.add_argument("--port", type=int, default=5950)
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    from bamos.config import load_config
    from bamos.environment.hssd_env import HabitatEnvironment, TeleportClient
    from bamos.planning.runner import camera_from_cfg

    cfg = load_config(a.config)
    client = TeleportClient(cfg["hssd"]["host"], a.port, timeout=7200.0)
    cam = camera_from_cfg(cfg["cameras"]["robot"])
    first = {}
    for e in client.episodes():
        first.setdefault(e["scene"], e["episode_id"])
    k, n = (int(x) for x in a.shard.split("/"))
    for s in sorted(first)[k::n]:
        t = time.time()
        HabitatEnvironment(client, cfg, s, first[s], cam).true_points()
        print(f"{s} {time.time() - t:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
