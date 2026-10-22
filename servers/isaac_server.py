"""Isaac Sim render server (runs in Isaac Sim 6.1.0's Python, 3.12: $ISAAC_PY).

Protocol (TCP, one request per message): 8-byte big-endian header length, JSON header, 8-byte payload length,
raw payload (npz bytes). Commands: ping | build_scene | render | shutdown.

Conventions: world z up, meters. The core sends world-from-optical (x right, y down, z forward) transforms; USD
cameras look down -Z with +Y up, so world_from_usdcam = world_from_optical @ diag(1,-1,-1). Depth returned is the
`distance_to_image_plane` annotator (optical-axis depth, float32, inf for background). Instance ids are mapped
from the instance_segmentation_fast idToLabels (prim path) back to the scene's integer instance ids.
"""
from __future__ import annotations

import argparse
import io
import json
import socket
import struct
import time
import traceback

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=5555)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--width", type=int, default=320)
parser.add_argument("--height", type=int, default=240)
args = parser.parse_args()

t_start = time.time()
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "renderer": "RayTracedLighting", "width": args.width, "height": args.height, "multi_gpu": False, "active_gpu": args.gpu, "physics_gpu": args.gpu})
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.semantics import add_labels  # noqa: E402
from isaacsim.core.version import get_version  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade  # noqa: E402

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
dome = UsdLux.DomeLight.Define(stage, "/World/dome")
dome.CreateIntensityAttr(600.0)
dome.CreateColorAttr(Gf.Vec3f(0.95, 0.97, 1.0))
key = UsdLux.DistantLight.Define(stage, "/World/key")
key.CreateIntensityAttr(1300.0)
key.CreateAngleAttr(2.0)
UsdGeom.Xformable(key.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-55.0, 25.0, 0.0))
UsdGeom.Xform.Define(stage, "/World/Looks")

# --------------------------------------------------------------------------- materials (RGB only; depth/segmentation unaffected)
PALETTE = {  # (base colour, roughness, metallic)
    "floor": ((0.42, 0.42, 0.40), 0.85, 0.0),
    "wall": ((0.82, 0.82, 0.78), 0.9, 0.0),
    # walls and floor per archetype
    "wall/warehouse_aisles": ((0.58, 0.58, 0.56), 1.0, 0.0),      # painted concrete block, light grey
    "floor/warehouse_aisles": ((0.46, 0.46, 0.44), 0.6, 0.0),     # sealed concrete
    "wall/industrial_workcell": ((0.68, 0.66, 0.60), 1.0, 0.0),  # off-white painted plaster
    "floor/industrial_workcell": ((0.36, 0.43, 0.39), 0.45, 0.0), # green-grey epoxy
    "wall/dense_storage": ((0.62, 0.58, 0.52), 1.0, 0.0),         # warm grey
    "floor/dense_storage": ((0.34, 0.34, 0.33), 0.75, 0.0),       # dark concrete
    "wall/large_warehouse": ((0.70, 0.71, 0.72), 1.0, 0.0),       # cladding, cool grey
    "floor/large_warehouse": ((0.50, 0.49, 0.45), 0.55, 0.0),     # polished concrete
    # anchors, per part
    "shelf/post": ((0.10, 0.24, 0.52), 0.45, 0.5),   # blue painted steel upright
    "shelf/board": ((0.50, 0.51, 0.52), 0.65, 0.25),   # galvanised steel deck
    "shelf": ((0.10, 0.24, 0.52), 0.45, 0.5),
    "table/top": ((0.50, 0.35, 0.21), 0.6, 0.0),     # wood-laminate top
    "table/leg": ((0.20, 0.20, 0.22), 0.4, 0.7),     # dark steel legs
    "table": ((0.50, 0.35, 0.21), 0.6, 0.0),
    "cabinet": ((0.62, 0.64, 0.66), 0.45, 0.35),     # light grey powder-coated steel
    "box": ((0.72, 0.55, 0.34), 0.9, 0.0),         # cardboard
    "surface_box": ((0.66, 0.50, 0.30), 0.9, 0.0),
    "pallet": ((0.60, 0.48, 0.30), 0.85, 0.0),
    "barrel": ((0.15, 0.30, 0.65), 0.35, 0.3),     # blue drum
    "pillar": ((0.55, 0.55, 0.53), 0.9, 0.0),      # concrete
    "target_cube": ((0.80, 0.12, 0.10), 0.5, 0.0),    # red painted box ("the red box")
    "target_sphere": ((0.80, 0.12, 0.10), 0.4, 0.0),  # red ball ("the red ball")
    "default": ((0.6, 0.6, 0.6), 0.7, 0.0),
}
_materials = {}


def _hash_jitter(name: str, amt: float = 0.06):
    h = abs(hash(name)) % 1000 / 1000.0
    return (h - 0.5) * 2 * amt


def get_material(kind: str, name: str = ""):
    keyname = kind if kind in PALETTE else "default"
    jitter = _hash_jitter(name) if keyname in ("box", "surface_box", "pallet") else 0.0
    mkey = (keyname, round(jitter, 3))
    if mkey in _materials:
        return _materials[mkey]
    base, rough, metal = PALETTE[keyname]
    col = tuple(min(1.0, max(0.0, c + jitter)) for c in base)
    path = f"/World/Looks/{keyname.replace('/', '_')}_{len(_materials)}"
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*col))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(rough))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metal))
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    _materials[mkey] = mat
    return mat


def material_kind(obj: dict, k: int = 0) -> str:
    """The palette key of prim k of an object: walls and floor by archetype, anchor prims by part."""
    role = obj.get("role", "")
    meta = obj.get("meta", {}) or {}
    if role in ("floor", "wall"):
        key = f"{role}/{meta.get('archetype', '')}"
        return key if key in PALETTE else role
    if role == "target":
        return f"target_{meta.get('shape', 'cube')}"
    kind = meta.get("kind", "default")
    parts = meta.get("parts") or []
    if k < len(parts) and f"{kind}/{parts[k]}" in PALETTE:
        return f"{kind}/{parts[k]}"
    return kind
SCENE_ROOT = "/World/scene"
cam = UsdGeom.Camera.Define(stage, "/World/cam")
cam.CreateFocalLengthAttr(15.0)
cam.CreateHorizontalApertureAttr(20.955)
cam.CreateVerticalApertureAttr(20.955 * args.height / args.width)
cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 100.0))
xf = UsdGeom.Xformable(cam.GetPrim())
xf.ClearXformOpOrder()
cam_op = xf.AddTransformOp()
render_product = rep.create.render_product("/World/cam", (args.width, args.height))
annotators = {}
for name in ["rgb", "distance_to_image_plane", "instance_id_segmentation_fast", "instance_segmentation_fast"]:
    a = rep.AnnotatorRegistry.get_annotator(name)
    a.attach([render_product])
    annotators[name] = a
world.reset()
for _ in range(3):
    world.step(render=True)
print(f"[isaac_server] ready on 127.0.0.1:{args.port} isaac={get_version()[0]} gpu={args.gpu} startup={time.time()-t_start:.1f}s", flush=True)

path_to_id: dict[str, int] = {}
current_scene_hash = None
render_count = 0


def set_camera(K: dict, world_from_optical: np.ndarray):
    W, H = K["width"], K["height"]
    if (W, H) != (args.width, args.height):
        raise ValueError(f"server resolution {(args.width, args.height)} != requested {(W, H)}")
    Hap = 20.955
    cam.GetFocalLengthAttr().Set(float(Hap * K["fx"] / W))
    cam.GetHorizontalApertureAttr().Set(Hap)
    cam.GetVerticalApertureAttr().Set(float(Hap * K["fy"] / K["fx"] * H / W))
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(float(K["near"]) * 0.5, float(K["far"]) * 2.0))
    T = np.asarray(world_from_optical, dtype=np.float64)
    Tu = T.copy()
    Tu[:3, :3] = T[:3, :3] @ np.diag([1.0, -1.0, -1.0])
    cam_op.Set(Gf.Matrix4d(*[float(x) for x in Tu.T.flatten()]))


def build_scene(scene: dict):
    global path_to_id
    root = stage.GetPrimAtPath(SCENE_ROOT)
    if root and root.IsValid():
        stage.RemovePrim(SCENE_ROOT)
    UsdGeom.Xform.Define(stage, SCENE_ROOT)
    path_to_id = {}
    for obj in scene["objects"]:
        opath = f"{SCENE_ROOT}/obj_{obj['instance_id']}"
        UsdGeom.Xform.Define(stage, opath)
        path_to_id[opath] = int(obj["instance_id"])
        for k, prim in enumerate(obj["prims"]):
            ppath = f"{opath}/p{k}"
            if prim["kind"] == "box":
                g = UsdGeom.Cube.Define(stage, ppath)
                g.CreateSizeAttr(1.0)
                x = UsdGeom.Xformable(g.GetPrim())
                x.ClearXformOpOrder()
                x.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in prim["center"]]))
                x.AddRotateZOp().Set(float(np.rad2deg(prim.get("yaw", 0.0))))
                x.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in prim["size"]]))
            elif prim["kind"] == "sphere":
                g = UsdGeom.Sphere.Define(stage, ppath)
                g.CreateRadiusAttr(float(prim["radius"]))
                x = UsdGeom.Xformable(g.GetPrim())
                x.ClearXformOpOrder()
                x.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in prim["center"]]))
            else:
                g = UsdGeom.Cylinder.Define(stage, ppath)
                g.CreateRadiusAttr(float(prim["radius"]))
                g.CreateHeightAttr(float(prim["height"]))
                g.CreateAxisAttr("Z")
                x = UsdGeom.Xformable(g.GetPrim())
                x.ClearXformOpOrder()
                x.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in prim["center"]]))
            add_labels(g.GetPrim(), labels=[obj["role"]], instance_name="class")
            path_to_id[ppath] = int(obj["instance_id"])
            mat = get_material(material_kind(obj, k), obj.get("name", ""))
            UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(mat)
    for _ in range(3):
        world.step(render=True)


def set_visibility(hidden_ids: list[int]):
    for path, iid in path_to_id.items():
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid() or not prim.IsA(UsdGeom.Gprim):
            continue
        img = UsdGeom.Imageable(prim)
        if iid in hidden_ids:
            img.MakeInvisible()
        else:
            img.MakeVisible()


def render(K: dict, world_from_optical, hide_instances: list[int], n_steps: int = 2):
    global render_count
    set_camera(K, world_from_optical)
    if hide_instances:
        set_visibility(hide_instances)
    for _ in range(n_steps):
        world.step(render=True)
    rgb = annotators["rgb"].get_data()
    depth = annotators["distance_to_image_plane"].get_data()
    seg = annotators["instance_id_segmentation_fast"].get_data()
    if hide_instances:
        set_visibility([])
    seg_data = seg["data"] if isinstance(seg, dict) else seg
    id_to_labels = seg["info"]["idToLabels"] if isinstance(seg, dict) else {}
    ids = np.full(seg_data.shape, -1, dtype=np.int32)
    for sid, label in id_to_labels.items():
        sid = int(sid)
        # label is a prim path like /World/scene/obj_7/p0 (or a parent); map by longest matching prefix
        iid = -1
        for path, val in path_to_id.items():
            if label == path or label.startswith(path + "/"):
                iid = val
                break
        ids[seg_data == sid] = iid
    render_count += 1
    depth = np.asarray(depth, dtype=np.float32)
    uniq = np.unique(seg_data).tolist()
    return {"rgb": np.asarray(rgb)[..., :3].astype(np.uint8), "depth": depth, "ids": ids}, {str(k): str(v) for k, v in id_to_labels.items()}, uniq


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
running = True
while running:
    conn, _ = srv.accept()
    try:
        while True:
            header, payload = recv_msg(conn)
            cmd = header.get("cmd")
            t0 = time.time()
            try:
                if cmd == "ping":
                    send_msg(conn, {"ok": True, "isaac_version": get_version()[0], "render_count": render_count, "resolution": [args.width, args.height], "depth_kind": "optical_axis", "annotators": list(annotators)})
                elif cmd == "build_scene":
                    build_scene(header["scene"])
                    current_scene_hash = header.get("scene_hash")
                    send_msg(conn, {"ok": True, "n_prims": len(path_to_id), "ms": (time.time() - t0) * 1000})
                    print(f"[isaac_server] build_scene {current_scene_hash} n_prims={len(path_to_id)} {1000*(time.time()-t0):.0f} ms", flush=True)
                elif cmd == "render":
                    if header.get("scene_hash") != current_scene_hash:
                        send_msg(conn, {"ok": False, "error": f"scene mismatch: server has {current_scene_hash}"})
                        continue
                    out, labels, uniq = render(header["camera"], header["world_from_camera"], header.get("hide_instances", []), int(header.get("n_steps", 2)))
                    send_msg(conn, {"ok": True, "ms": (time.time() - t0) * 1000, "render_count": render_count, "id_to_labels": labels, "seg_unique": uniq}, out)
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
        print(f"[isaac_server] connection ended: {e}", flush=True)
    finally:
        conn.close()
print("[isaac_server] shutting down", flush=True)
app.close()
