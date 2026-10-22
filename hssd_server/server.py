"""Habitat render server for HSSD ObjectNav with teleport poses (run with the habitat env's python: habitat-sim/lab 0.2.5).

The scene, the episodes and the navmesh are the benchmark's: habitat-lab's objectnav_hssd-hab.yaml (navmesh computed at
load: agent radius 0.18, height 0.88, static objects included, NavMeshSettings defaults otherwise). Physics is on so the
scene's objects can be fetched; each gets its own semantic id (object id + 1, 0 for the stage). No actions are taken: the agent is placed at the requested poses.

Protocol: the Isaac server's (8-byte big-endian header length, JSON header, 8-byte payload length, npz payload).
Commands
  ping                                   benchmark constants
  episodes                               [{scene, episode_id, category}]
  load {scene, episode_id}               resets to the episode; its goals (semantic id, AABB), start, geodesic distance
  lattice {step, start, all_islands}     navmesh points on an xz lattice (every navigable height), on the start's island
                                         (all_islands: on every island),
                                         with the floor surface under each (ray cast down) and a sloped flag
  geodesic {start, ends}                 geodesic distance from start to each end (inf: no path)
  render {poses}                         per pose [x, y, z, yaw, pitch] (agent base on the navmesh, habitat frame, yaw
                                         about +y, 0 looking -z; pitch about the camera's x, + up): rgb, depth (metres
                                         along the optical axis, the sensor's clipping: min_depth below, max_depth beyond),
                                         instance ids, and the depth camera's world transform (habitat camera: -z forward)
  surfaces {lo, hi, spacing, categories} hit points (and their semantic ids) of the stage and of the objects whose
                                         category matches one of ("*": every object), of
                                         `categories` (objects.csv), from axis-aligned ray sweeps over the box (all hits)
  shutdown
Everything is in Habitat's frame (y up); the client converts.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import socket
import struct
import time
import traceback

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=5800)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--split", default="val")
parser.add_argument("--data-root", required=True, help="a directory holding data/ (scripts/hssd/make_data_root.sh)")
parser.add_argument("--scenes", default="", help="comma-separated scene ids to load (default: every scene of the split)")
args = parser.parse_args()

os.chdir(args.data_root)  # the episode files reference ./data/...
# the scene assets: the episodes were made on hssd-hab v0.2.4 (no object colliders; v0.2.5's colliders change the navmesh)
SCENE_DATA = os.path.realpath("data/scene_datasets/hssd-hab")
import numpy as np  # noqa: E402
import magnum as mn  # noqa: E402
import habitat  # noqa: E402
import habitat_sim  # noqa: E402
from habitat.config import read_write  # noqa: E402
from habitat.config.default_structured_configs import HabitatSimSemanticSensorConfig  # noqa: E402
from habitat_sim.utils.common import quat_from_angle_axis  # noqa: E402

cfg = habitat.get_config("benchmark/nav/objectnav/objectnav_hssd-hab.yaml")
with read_write(cfg):
    cfg.habitat.dataset.split = args.split
    if args.scenes:  # only these scenes' episodes (the train split has 120k)
        cfg.habitat.dataset.content_scenes = args.scenes.split(",")
    cfg.habitat.simulator.habitat_sim_v0.enable_physics = True  # the scene's objects become fetchable
    cfg.habitat.simulator.habitat_sim_v0.gpu_device_id = args.gpu
    cfg.habitat.environment.iterator_options.shuffle = False
    cfg.habitat.environment.iterator_options.max_scene_repeat_steps = -1
    agent = cfg.habitat.simulator.agents.main_agent
    agent.sim_sensors.depth_sensor.normalize_depth = False
    rgb = agent.sim_sensors.rgb_sensor
    agent.sim_sensors["semantic_sensor"] = HabitatSimSemanticSensorConfig(
        width=rgb.width, height=rgb.height, hfov=rgb.hfov, position=list(rgb.position))
dcfg = cfg.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor
BENCHMARK = {"width": int(dcfg.width), "height": int(dcfg.height), "hfov": float(dcfg.hfov),
             "min_depth": float(dcfg.min_depth), "max_depth": float(dcfg.max_depth),
             "sensor_height": float(dcfg.position[1]), "agent_radius": float(agent.radius),
             "agent_height": float(agent.height)}
env = habitat.Env(cfg)
sim = env.sim
EPISODES = {(os.path.basename(e.scene_id).split(".")[0], str(e.episode_id)): e for e in env.episodes}
CATEGORY = {r["id"]: (r["main_category"] or r["wnsynsetkey"] or "").lower()
            for r in csv.DictReader(open("data/scene_datasets/hssd-hab/semantics/objects.csv"))}
print(f"[hssd_server] scene data {SCENE_DATA}; {len(EPISODES)} {args.split} episodes, sensors {list(sim._sensors)}", flush=True)
current = {"key": None}


def handle_category(handle: str) -> str:
    return CATEGORY.get(handle.split("_:")[0].split("/")[-1], "")


def world_aabb(o) -> tuple[list, list]:
    bb = o.root_scene_node.cumulative_bb
    T = np.asarray(o.transformation, dtype=np.float64)
    cs = np.array([[x, y, z, 1.0] for x in (bb.left, bb.right) for y in (bb.bottom, bb.top) for z in (bb.back, bb.front)])
    w = (cs @ T.T)[:, :3]
    return w.min(0).tolist(), w.max(0).tolist()


def load(scene: str, episode_id: str) -> dict:
    ep = EPISODES[(scene, str(episode_id))]
    env.current_episode = ep  # habitat.Env.reset uses the episode set here
    env.reset()
    rom = sim.get_rigid_object_manager()
    objects = []  # [semantic id, objects.csv category] of every rigid object (false-positive masks of the noisy channels)
    for h in rom.get_object_handles():
        o = rom.get_object_by_handle(h)
        if o is None:
            raise RuntimeError(f"object {h} not fetchable (physics off?)")
        o.semantic_id = int(o.object_id) + 1
        objects.append([int(o.semantic_id), handle_category(h)])
    goals, missing = [], []
    for g in ep.goals:
        o = rom.get_object_by_handle(g.object_name)
        if o is None:  # a goal the scene does not hold under that handle (seen in train): left out, reported
            missing.append(g.object_name)
            continue
        lo, hi = world_aabb(o)
        goals.append({"handle": g.object_name, "semantic_id": int(o.object_id) + 1, "category": g.object_category,
                      "position": [float(x) for x in g.position], "aabb_lo": lo, "aabb_hi": hi,
                      "n_view_points": len(g.view_points or [])})
    current["key"] = (scene, str(episode_id))
    lo, hi = sim.pathfinder.get_bounds()
    return {"scene": scene, "episode_id": str(ep.episode_id), "category": ep.object_category,
            "start_position": [float(x) for x in ep.start_position], "start_rotation": [float(x) for x in ep.start_rotation],
            "geodesic_distance": float(ep.info.get("geodesic_distance", float("nan"))), "goals": goals,
            "missing_goals": missing, "scene_data": SCENE_DATA, "objects": objects,
            "navmesh_bounds": [list(map(float, lo)), list(map(float, hi))],
            "navigable_area": float(sim.pathfinder.navigable_area)}


def floor_below(p) -> float:
    """The surface under a navmesh point (the navmesh floats above the floor): first hit of a ray cast down."""
    h = sim.cast_ray(habitat_sim.geo.Ray(mn.Vector3(float(p[0]), float(p[1]) + 0.3, float(p[2])), mn.Vector3(0, -1, 0)), 3.0)
    return float(p[1]) + 0.3 - h.hits[0].ray_distance if h.has_hits() else float("nan")


def lattice(step: float, start, all_islands: bool = False) -> dict:
    pf = sim.pathfinder
    s = np.asarray(pf.snap_point(np.asarray(start, np.float32)), np.float64)
    island = pf.get_island(s.astype(np.float32))
    lo, hi = pf.get_bounds()

    def snap_ok(x, h, z):
        p = np.asarray(pf.snap_point(np.array([x, h, z], np.float32)), np.float64)
        return p if np.all(np.isfinite(p)) and np.hypot(p[0] - x, p[2] - z) <= 0.05 else None

    pts = {}
    for x in np.arange(np.floor(lo[0] / step) * step, hi[0] + 1e-6, step):
        for z in np.arange(np.floor(lo[2] / step) * step, hi[2] + 1e-6, step):
            for h in np.arange(lo[1] - 0.1, hi[1] + 0.3, 0.25):
                p = snap_ok(x, h, z)
                if p is not None and (all_islands or pf.get_island(p.astype(np.float32)) == island):
                    pts[(round(float(x), 3), round(float(z), 3), round(float(p[1]), 1))] = p
    P = np.asarray(list(pts.values()), dtype=np.float64).reshape(-1, 3)
    sloped = np.zeros(len(P), dtype=bool)
    for i, p in enumerate(P):
        for dx, dz in ((0.2, 0), (-0.2, 0), (0, 0.2), (0, -0.2)):
            q = snap_ok(p[0] + dx, p[1], p[2] + dz)
            if q is not None and abs(q[1] - p[1]) > 0.04:
                sloped[i] = True
                break
    floor = np.array([floor_below(p) for p in P])
    return {"island": int(island), "start_snapped": s.tolist()}, {"positions": P, "floor_y": floor, "sloped": sloped}


def geodesic(start, ends) -> np.ndarray:
    pf = sim.pathfinder
    out = np.full(len(ends), np.inf)
    s = np.asarray(start, np.float32)
    for i, e in enumerate(np.asarray(ends, np.float32)):
        sp = habitat_sim.ShortestPath()
        sp.requested_start, sp.requested_end = s, e
        if pf.find_path(sp):
            out[i] = sp.geodesic_distance
    return out


def sensor_nodes() -> dict:
    """The sensors' scene nodes, fetched anew: a scene change rebuilds them (a node kept from before is freed)."""
    return {k: s._sensor_object.node for k, s in sim._sensors.items()}


def render(poses) -> dict:
    H, W = BENCHMARK["height"], BENCHMARK["width"]
    n = len(poses)
    rgb = np.zeros((n, H, W, 3), np.uint8)
    depth = np.zeros((n, H, W), np.float32)
    ids = np.zeros((n, H, W), np.int32)
    T = np.zeros((n, 4, 4), np.float64)
    for i, (x, y, z, yaw, pitch) in enumerate(poses):
        sim.set_agent_state(np.array([x, y, z], np.float32), quat_from_angle_axis(float(yaw), np.array([0.0, 1.0, 0.0])))
        nodes = sensor_nodes()
        for node in nodes.values():
            node.rotation = mn.Quaternion.rotation(mn.Rad(float(pitch)), mn.Vector3.x_axis())
        obs = sim.get_sensor_observations()
        rgb[i] = np.asarray(obs["rgb"])[..., :3]
        depth[i] = np.asarray(obs["depth"]).reshape(H, W)
        ids[i] = np.asarray(obs["semantic"]).reshape(H, W).astype(np.int32)
        T[i] = np.asarray(nodes["depth"].absolute_transformation(), np.float64)
    return {"rgb": rgb, "depth": depth, "ids": ids, "T": T}


def surfaces(lo, hi, spacing: float, categories: list[str]) -> tuple[dict, dict]:
    """Hit points of the stage and of the objects whose category contains one of `categories` ("*": every object), and
    each point's semantic id (object_id + 1; 0 for the stage)."""
    rom = sim.get_rigid_object_manager()
    allowed, names = {-1}, []
    every = "*" in categories
    for h in rom.get_object_handles():
        c = handle_category(h)
        if every or any(k in c for k in categories):
            allowed.add(int(rom.get_object_by_handle(h).object_id))
            names.append([h, c])
    lo, hi = np.asarray(lo, np.float64), np.asarray(hi, np.float64)
    pts, sem = [], []
    for a in range(3):
        b, c = [k for k in range(3) if k != a]
        L = float(hi[a] - lo[a]) + 0.2
        for u in np.arange(lo[b], hi[b] + 1e-9, spacing):
            for v in np.arange(lo[c], hi[c] + 1e-9, spacing):
                o = np.zeros(3)
                o[a], o[b], o[c] = lo[a] - 0.1, u, v
                d = np.zeros(3)
                d[a] = 1.0
                r = sim.cast_ray(habitat_sim.geo.Ray(mn.Vector3(*o), mn.Vector3(*d)), L)
                for h in r.hits:
                    if int(h.object_id) in allowed:
                        pts.append(o + d * h.ray_distance)
                        sem.append(int(h.object_id) + 1)
    return {"objects": names}, {"points": np.asarray(pts, np.float64).reshape(-1, 3), "semantic_ids": np.asarray(sem, np.int64)}


# --------------------------------------------------------------------------------------------------- transport
def recv_exact(conn, n):
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(min(1 << 20, n - len(buf)))
        if not chunk:
            raise ConnectionError("socket closed")
        buf += chunk
    return buf


def recv_msg(conn):
    (hl,) = struct.unpack(">Q", recv_exact(conn, 8))
    header = json.loads(recv_exact(conn, hl).decode())
    (pl,) = struct.unpack(">Q", recv_exact(conn, 8))
    payload = recv_exact(conn, pl) if pl else b""
    return header, payload


def send_msg(conn, header: dict, arrays: dict | None = None):
    payload = b""
    if arrays:
        bio = io.BytesIO()
        np.savez(bio, **arrays)
        payload = bio.getvalue()
    hb = json.dumps(header).encode()
    conn.sendall(struct.pack(">Q", len(hb)) + hb + struct.pack(">Q", len(payload)) + payload)


srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("127.0.0.1", args.port))
srv.listen(1)
print(f"[hssd_server] listening on {args.port}", flush=True)
running = True
while running:
    conn, _ = srv.accept()
    conn.settimeout(3600.0)
    try:
        while True:
            header, payload = recv_msg(conn)
            arrs = dict(np.load(io.BytesIO(payload))) if payload else {}
            cmd = header.get("cmd")
            t0 = time.time()
            try:
                if cmd == "ping":
                    send_msg(conn, {"ok": True, "habitat_version": habitat.__version__, "split": args.split,
                                    "benchmark": BENCHMARK, "n_episodes": len(EPISODES), "scene_data": SCENE_DATA})
                elif cmd == "episodes":
                    send_msg(conn, {"ok": True, "episodes": [{"scene": k[0], "episode_id": k[1], "category": e.object_category}
                                                             for k, e in EPISODES.items()]})
                elif cmd == "load":
                    send_msg(conn, {"ok": True, **load(header["scene"], str(header["episode_id"])), "ms": (time.time() - t0) * 1000})
                elif cmd == "lattice":
                    h, a = lattice(float(header["step"]), header["start"], bool(header.get("all_islands", False)))
                    send_msg(conn, {"ok": True, **h, "ms": (time.time() - t0) * 1000}, a)
                elif cmd == "geodesic":
                    send_msg(conn, {"ok": True, "ms": (time.time() - t0) * 1000},
                             {"dist": geodesic(header["start"], arrs["ends"])})
                elif cmd == "render":
                    send_msg(conn, {"ok": True, "ms": (time.time() - t0) * 1000}, render(arrs["poses"]))
                elif cmd == "surfaces":
                    h, a = surfaces(header["lo"], header["hi"], float(header["spacing"]), list(header["categories"]))
                    send_msg(conn, {"ok": True, **h, "ms": (time.time() - t0) * 1000}, a)
                elif cmd == "shutdown":
                    send_msg(conn, {"ok": True})
                    running = False
                    break
                else:
                    send_msg(conn, {"ok": False, "error": f"unknown cmd {cmd}"})
            except Exception as e:
                traceback.print_exc()
                send_msg(conn, {"ok": False, "error": repr(e)})
    except (ConnectionError, OSError) as e:
        print(f"[hssd_server] connection ended: {e}", flush=True)
    finally:
        conn.close()
print("[hssd_server] shutting down", flush=True)
env.close()
