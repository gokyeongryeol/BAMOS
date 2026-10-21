"""Layout archetypes, anchor assemblies, clutter, target placement and the movable pose pool.

Anchor/clutter geometry is built from primitives (see ASSUMPTIONS.md). GT placement fields are written only to the
evaluator-only part of the SceneDescription.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bamos.environment.scene import Box, Cylinder, SceneObject, Sphere


@dataclass
class Surface:
    """A horizontal placement region: rectangle [x0,x1]x[y0,y1] at height z_top with vertical clearance."""

    anchor: str
    mode: str  # on_top | inside
    x0: float
    x1: float
    y0: float
    y1: float
    z_top: float
    clearance: float  # free height above the surface (inf for on_top)
    occupied: list[tuple[float, float, float, float]] = field(default_factory=list)  # footprints already used


def _target_spec(shape: str, s: float) -> dict:
    """A target whose size scale sqrt(surface area / 4) is exactly s (formulation §3): a cube of side s sqrt(2/3)
    (area 6 c^2) or a sphere of diameter 2 s / sqrt(pi) (area pi D^2)."""
    d = s * np.sqrt(2.0 / 3.0) if shape == "cube" else 2.0 * s / np.sqrt(np.pi)
    return {"shape": shape, "s": s, "dim": float(d), "extent": (float(d),) * 3}


# the benchmark's targets: isotropic shapes only, at size scales that are multiples of the lattice voxel (0.1 m), so
# each target is exactly one size hypothesis of planner.size_grid_m
TARGET_SPECS = {f"{shape}{round(100 * s)}": _target_spec(shape, s) for shape in ("cube", "sphere") for s in (0.1, 0.2, 0.3)}


def make_target(kind: str, center_xy, z_bottom: float, instance_id: int, yaw: float) -> SceneObject:
    x, y = center_xy
    spec = TARGET_SPECS[kind]
    d = spec["dim"]
    meta = {"type": kind, "shape": spec["shape"], "size_scale": spec["s"]}
    if spec["shape"] == "cube":
        return SceneObject(instance_id, f"target_{kind}", "target", [Box((x, y, z_bottom + d / 2), (d, d, d), yaw)], meta)
    if spec["shape"] == "sphere":
        return SceneObject(instance_id, f"target_{kind}", "target", [Sphere((x, y, z_bottom + d / 2), d / 2)], meta)
    raise ValueError(kind)


class LayoutBuilder:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.objects: list[SceneObject] = []
        self.surfaces: list[Surface] = []
        self.next_id = 1

    def add(self, name: str, role: str, prims, meta=None) -> SceneObject:
        o = SceneObject(self.next_id, name, role, list(prims), meta or {})
        self.next_id += 1
        self.objects.append(o)
        return o

    # ------------------------------------------------------------ anchors
    def shelf(self, cx: float, cy: float, yaw: float, width=1.8, depth=0.6, levels=(0.30, 0.80), top=1.30, name="shelf") -> SceneObject:
        post = 0.08
        board_t = 0.04
        cy_, sy_ = np.cos(yaw), np.sin(yaw)

        def loc(dx, dy, dz, sx, sy, sz):
            return Box((cx + dx * cy_ - dy * sy_, cy + dx * sy_ + dy * cy_, dz), (sx, sy, sz), yaw)

        prims = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                prims.append(loc(sx * (width / 2 - post / 2), sy * (depth / 2 - post / 2), top / 2, post, post, top))
        zs = list(levels) + [top]
        for z in zs:
            prims.append(loc(0, 0, z - board_t / 2, width, depth, board_t))
        # "parts" names each prim for the renderer's materials (RGB only: geometry, depth and ids are unaffected)
        o = self.add(name, "anchor", prims, {"kind": "shelf", "yaw": yaw, "parts": ["post"] * 4 + ["board"] * len(zs)})
        # surfaces (axis aligned approximation of the rotated rectangle is exact for yaw in {0, pi/2, pi, 3pi/2})
        hw, hd = (width / 2 - post, depth / 2 - 0.05)
        if abs(np.sin(yaw)) > 0.5:
            hw, hd = hd, hw
        for i, z in enumerate(zs):
            mode = "on_top" if i == len(zs) - 1 else "inside"
            clearance = np.inf if mode == "on_top" else zs[i + 1] - board_t - z
            self.surfaces.append(Surface(name, mode, cx - hw, cx + hw, cy - hd, cy + hd, z, clearance))
        return o

    def table(self, cx, cy, yaw, width=1.2, depth=0.8, height=0.75, name="table") -> SceneObject:
        cy_, sy_ = np.cos(yaw), np.sin(yaw)
        top_t = 0.05
        leg = 0.06

        def loc(dx, dy, dz, sx, sy, sz):
            return Box((cx + dx * cy_ - dy * sy_, cy + dx * sy_ + dy * cy_, dz), (sx, sy, sz), yaw)

        prims = [loc(0, 0, height - top_t / 2, width, depth, top_t)]
        for sx in (-1, 1):
            for sy in (-1, 1):
                prims.append(loc(sx * (width / 2 - leg), sy * (depth / 2 - leg), (height - top_t) / 2, leg, leg, height - top_t))
        o = self.add(name, "anchor", prims, {"kind": "table", "yaw": yaw, "parts": ["top"] + ["leg"] * 4})
        hw, hd = width / 2 - 0.08, depth / 2 - 0.08
        if abs(np.sin(yaw)) > 0.5:
            hw, hd = hd, hw
        self.surfaces.append(Surface(name, "on_top", cx - hw, cx + hw, cy - hd, cy + hd, height, np.inf))
        return o

    def cabinet(self, cx, cy, yaw, width=0.9, depth=0.5, height=1.1, name="cabinet") -> SceneObject:
        """Open-front cabinet; the opening faces local -y rotated by yaw. Interior shelves at bottom and middle."""
        t = 0.03
        cy_, sy_ = np.cos(yaw), np.sin(yaw)

        def loc(dx, dy, dz, sx, sy, sz):
            return Box((cx + dx * cy_ - dy * sy_, cy + dx * sy_ + dy * cy_, dz), (sx, sy, sz), yaw)

        mid = height / 2
        prims = [
            loc(0, depth / 2 - t / 2, height / 2, width, t, height),  # back
            loc(-(width / 2 - t / 2), 0, height / 2, t, depth, height),  # sides
            loc(width / 2 - t / 2, 0, height / 2, t, depth, height),
            loc(0, 0, t / 2, width, depth, t),  # bottom
            loc(0, 0, height - t / 2, width, depth, t),  # top
            loc(0, 0, mid, width - 2 * t, depth - t, t),  # middle shelf
        ]
        o = self.add(name, "anchor", prims, {"kind": "cabinet", "yaw": yaw})
        hw, hd = width / 2 - t - 0.05, depth / 2 - t - 0.04
        if abs(np.sin(yaw)) > 0.5:
            hw, hd = hd, hw
        self.surfaces.append(Surface(name, "on_top", cx - hw, cx + hw, cy - hd, cy + hd, height, np.inf))
        self.surfaces.append(Surface(name, "inside", cx - hw, cx + hw, cy - hd, cy + hd, t, mid - t / 2 - t))
        self.surfaces.append(Surface(name, "inside", cx - hw, cx + hw, cy - hd, cy + hd, mid + t / 2, height - t - (mid + t / 2)))
        return o

    # ------------------------------------------------------------ clutter
    def box_clutter(self, cx, cy, sx, sy, sz, z_bottom=0.0, yaw=0.0, name="box") -> SceneObject:
        return self.add(name, "clutter", [Box((cx, cy, z_bottom + sz / 2), (sx, sy, sz), yaw)], {"kind": "box"})

    def barrel(self, cx, cy, r=0.28, h=0.85, name="barrel") -> SceneObject:
        return self.add(name, "clutter", [Cylinder((cx, cy, h / 2), r, h)], {"kind": "barrel"})

    def pallet(self, cx, cy, yaw=0.0, name="pallet") -> SceneObject:
        return self.add(name, "clutter", [Box((cx, cy, 0.075), (1.2, 0.8, 0.15), yaw)], {"kind": "pallet"})

    def pillar(self, cx, cy, s=0.3, h=2.8, name="pillar") -> SceneObject:
        return self.add(name, "clutter", [Box((cx, cy, h / 2), (s, s, h))], {"kind": "pillar"})

    def surface_clutter(self, surf: Surface, n: int, size_range=(0.15, 0.35)):
        """Small boxes on an anchor surface (create occlusion for targets placed there)."""
        for k in range(n):
            s = self.rng.uniform(*size_range, size=3)
            s[2] = min(s[2], surf.clearance - 0.02) if np.isfinite(surf.clearance) else s[2]
            if s[2] <= 0.05:
                continue
            xy = self._sample_on_surface(surf, s[:2])
            if xy is None:
                continue
            self.add(f"{surf.anchor}_clutter{k}", "clutter", [Box((xy[0], xy[1], surf.z_top + s[2] / 2), tuple(s))], {"kind": "surface_box"})
            surf.occupied.append((xy[0] - s[0] / 2, xy[0] + s[0] / 2, xy[1] - s[1] / 2, xy[1] + s[1] / 2))

    def _sample_on_surface(self, surf: Surface, footprint, tries: int = 40):
        fx, fy = footprint[0] / 2 + 0.01, footprint[1] / 2 + 0.01
        if surf.x1 - surf.x0 < 2 * fx or surf.y1 - surf.y0 < 2 * fy:
            return None
        for _ in range(tries):
            x = self.rng.uniform(surf.x0 + fx, surf.x1 - fx)
            y = self.rng.uniform(surf.y0 + fy, surf.y1 - fy)
            ok = True
            for (a0, a1, b0, b1) in surf.occupied:
                if x + fx > a0 and x - fx < a1 and y + fy > b0 and y - fy < b1:
                    ok = False
                    break
            if ok:
                return (x, y)
        return None

    # ------------------------------------------------------------ difficulty (hard benchmark)
    def shift_target(self, tgt: SceneObject, kind: str, xy, z_bottom: float, yaw: float) -> SceneObject:
        """Rebuild the target at a new xy (used to place it off the candidate lattice)."""
        idx = self.objects.index(tgt)
        new = make_target(kind, xy, z_bottom, tgt.instance_id, yaw)
        self.objects[idx] = new
        return new

    def occlude_target(self, tgt: SceneObject, surf: "Surface", facing, cover: float = 0.75, gap: float = 0.03) -> SceneObject | None:
        """Put a clutter box on the same surface between the target and the open side.

        `facing` is the unit xy direction a viewer comes from. The box spans the target footprint plus a margin and
        reaches `cover` of the target height, so the target stays partly visible from above and from the sides but
        is cut off in the frontal view.
        """
        lo, hi = tgt.aabb()
        tw = float(max(hi[0] - lo[0], hi[1] - lo[1]))
        th = float(hi[2] - lo[2])
        h = max(0.06, cover * th)
        if np.isfinite(surf.clearance) and h > surf.clearance - 0.02:
            h = surf.clearance - 0.02
        if h <= 0.05:
            return None
        c = 0.5 * (lo[:2] + hi[:2]) + np.asarray(facing, dtype=float) * (0.5 * tw + 0.5 * 0.10 + gap)
        if not (surf.x0 + 0.05 <= c[0] <= surf.x1 - 0.05 and surf.y0 + 0.05 <= c[1] <= surf.y1 - 0.05):
            return None
        sx, sy = (0.10, max(0.12, tw + 0.08)) if abs(facing[0]) > abs(facing[1]) else (max(0.12, tw + 0.08), 0.10)
        o = self.add(f"{surf.anchor}_occluder", "clutter", [Box((c[0], c[1], surf.z_top + h / 2), (sx, sy, h))], {"kind": "occluder"})
        surf.occupied.append((c[0] - sx / 2, c[0] + sx / 2, c[1] - sy / 2, c[1] + sy / 2))
        return o

    # ------------------------------------------------------------ target
    def place_target(self, kind: str, mode: str, anchor_kinds=("shelf", "table", "cabinet")) -> tuple[SceneObject, Surface] | None:
        ext = TARGET_SPECS[kind]["extent"]
        cands = [s for s in self.surfaces if s.mode == mode and s.clearance > ext[2] + 0.02]
        anchors = {o.name: o.meta.get("kind") for o in self.objects if o.role == "anchor"}
        cands = [s for s in cands if anchors.get(s.anchor) in anchor_kinds]
        if not cands:
            return None
        order = self.rng.permutation(len(cands))
        for i in order:
            surf = cands[i]
            yaw = float(self.rng.choice([0.0, np.pi / 2]))
            fp = (ext[0], ext[1]) if yaw == 0.0 else (ext[1], ext[0])
            xy = self._sample_on_surface(surf, fp)
            if xy is None:
                continue
            tgt = make_target(kind, xy, surf.z_top, self.next_id, yaw)
            self.next_id += 1
            self.objects.append(tgt)
            surf.occupied.append((xy[0] - fp[0] / 2, xy[0] + fp[0] / 2, xy[1] - fp[1] / 2, xy[1] + fp[1] / 2))
            return tgt, surf
        return None


# ---------------------------------------------------------------------------------------------------- archetypes
ARCHETYPES = {
    "warehouse_aisles": {"size": (8.0, 6.0), "ceiling": 2.8},
    "industrial_workcell": {"size": (6.0, 6.0), "ceiling": 2.8},
    "dense_storage": {"size": (6.0, 6.0), "ceiling": 2.8},
    # a room larger than the detection range of a small target (2026-10-06): two rack rows with cross aisles. 12 x 8 m
    # rather than 16 x 10 m: planning cost grows with poses x cells, and 16 x 10 took 14 s per view on an A100
    "large_warehouse": {"size": (12.0, 8.0), "ceiling": 3.2},
}


def _walls_and_floor(b: LayoutBuilder, W: float, D: float, Hc: float, archetype: str):
    t = 0.1
    m = {"archetype": archetype}  # the renderer paints walls and floor per archetype
    b.add("floor", "floor", [Box((W / 2, D / 2, -t / 2), (W + 2 * t, D + 2 * t, t))], m)
    b.add("wall_x0", "wall", [Box((-t / 2, D / 2, Hc / 2), (t, D + 2 * t, Hc))], m)
    b.add("wall_x1", "wall", [Box((W + t / 2, D / 2, Hc / 2), (t, D + 2 * t, Hc))], m)
    b.add("wall_y0", "wall", [Box((W / 2, -t / 2, Hc / 2), (W + 2 * t, t, Hc))], m)
    b.add("wall_y1", "wall", [Box((W / 2, D + t / 2, Hc / 2), (W + 2 * t, t, Hc))], m)


def _free_floor_spot(b: LayoutBuilder, rng, W, D, fx, fy, margin=0.35, tries=60, keep_clear=None):
    """Sample an xy for a floor object with footprint fx x fy not overlapping existing footprints (2D)."""
    for _ in range(tries):
        x = rng.uniform(margin + fx / 2, W - margin - fx / 2)
        y = rng.uniform(margin + fy / 2, D - margin - fy / 2)
        ok = True
        if keep_clear is not None and keep_clear(x, y, fx, fy):
            continue
        for o in b.objects:
            if o.role in ("floor", "wall"):
                continue
            lo, hi = o.aabb()
            if x + fx / 2 + 0.3 > lo[0] and x - fx / 2 - 0.3 < hi[0] and y + fy / 2 + 0.3 > lo[1] and y - fy / 2 - 0.3 < hi[1]:
                ok = False
                break
        if ok:
            return x, y
    return None


def build_layout(archetype: str, layout_seed: int) -> LayoutBuilder:
    rng = np.random.default_rng(layout_seed)
    W, D = ARCHETYPES[archetype]["size"]
    Hc = ARCHETYPES[archetype]["ceiling"]
    b = LayoutBuilder(rng)
    _walls_and_floor(b, W, D, Hc, archetype)
    if archetype == "warehouse_aisles":
        # two rack rows along x at y ~ 1.4 and y ~ 4.6, aisle in the middle; racks in 1.8 m segments
        for row, y in enumerate((1.4, D - 1.4)):
            n_seg = 3
            x0 = 0.9 + rng.uniform(0.0, 0.3)
            for s in range(n_seg):
                if rng.uniform() < 0.85:
                    b.shelf(x0 + s * 1.95, y, 0.0, name=f"shelf_r{row}_{s}")
        # floor clutter: pallets/boxes/barrels in the aisle sides
        for k in range(int(rng.integers(3, 6))):
            kind = rng.choice(["pallet", "box", "barrel"])
            if kind == "pallet":
                spot = _free_floor_spot(b, rng, W, D, 1.2, 0.8)
                if spot:
                    b.pallet(*spot, name=f"pallet{k}")
                    if rng.uniform() < 0.7:
                        b.box_clutter(spot[0], spot[1], 0.8, 0.6, rng.uniform(0.4, 1.2), z_bottom=0.15, name=f"pallet{k}_box")
            elif kind == "box":
                s = rng.uniform(0.4, 0.9, size=3)
                spot = _free_floor_spot(b, rng, W, D, s[0], s[1])
                if spot:
                    b.box_clutter(*spot, *s, name=f"box{k}")
            else:
                spot = _free_floor_spot(b, rng, W, D, 0.6, 0.6)
                if spot:
                    b.barrel(*spot, name=f"barrel{k}")
    elif archetype == "industrial_workcell":
        # tables in the middle, cabinets against walls, a pillar
        for k in range(int(rng.integers(2, 4))):
            spot = _free_floor_spot(b, rng, W, D, 1.2, 0.8, margin=1.0)
            if spot:
                b.table(spot[0], spot[1], float(rng.choice([0.0, np.pi / 2])), name=f"table{k}")
        for k in range(int(rng.integers(1, 3))):
            side = rng.integers(0, 4)
            pos_along = rng.uniform(1.2, (W if side < 2 else D) - 1.2)
            if side == 0:
                cx, cy, yaw = pos_along, 0.35, 0.0  # opening faces +y? local -y is opening; yaw=pi flips
                yaw = np.pi
            elif side == 1:
                cx, cy, yaw = pos_along, D - 0.35, 0.0
            elif side == 2:
                cx, cy, yaw = 0.35, pos_along, -np.pi / 2
            else:
                cx, cy, yaw = W - 0.35, pos_along, np.pi / 2
            b.cabinet(cx, cy, yaw, name=f"cabinet{k}")
        if rng.uniform() < 0.6:
            spot = _free_floor_spot(b, rng, W, D, 0.3, 0.3, margin=1.5)
            if spot:
                b.pillar(*spot)
        for k in range(int(rng.integers(2, 5))):
            s = rng.uniform(0.3, 0.7, size=3)
            spot = _free_floor_spot(b, rng, W, D, s[0], s[1])
            if spot:
                b.box_clutter(*spot, *s, name=f"box{k}")
    elif archetype == "dense_storage":
        for k in range(int(rng.integers(2, 4))):
            spot = _free_floor_spot(b, rng, W, D, 1.8, 0.6, margin=0.6)
            if spot:
                b.shelf(spot[0], spot[1], float(rng.choice([0.0, np.pi / 2])), name=f"shelf{k}")
        if rng.uniform() < 0.7:
            side = rng.integers(0, 2)
            cx, cy, yaw = (rng.uniform(1.2, W - 1.2), 0.35, np.pi) if side == 0 else (0.35, rng.uniform(1.2, D - 1.2), -np.pi / 2)
            b.cabinet(cx, cy, yaw, name="cabinet0")
        for k in range(int(rng.integers(6, 11))):
            kind = rng.choice(["box", "stack", "barrel", "pallet"], p=[0.4, 0.25, 0.2, 0.15])
            if kind == "box":
                s = rng.uniform(0.35, 0.9, size=3)
                spot = _free_floor_spot(b, rng, W, D, s[0], s[1], margin=0.3)
                if spot:
                    b.box_clutter(*spot, *s, name=f"box{k}")
            elif kind == "stack":
                s = rng.uniform(0.5, 0.8, size=2)
                h = rng.uniform(1.2, 2.0)
                spot = _free_floor_spot(b, rng, W, D, s[0], s[1], margin=0.3)
                if spot:
                    b.box_clutter(*spot, s[0], s[1], h, name=f"stack{k}")
            elif kind == "barrel":
                spot = _free_floor_spot(b, rng, W, D, 0.6, 0.6, margin=0.3)
                if spot:
                    b.barrel(*spot, name=f"barrel{k}")
            else:
                spot = _free_floor_spot(b, rng, W, D, 1.2, 0.8, margin=0.3)
                if spot:
                    b.pallet(*spot, name=f"pallet{k}")
                    b.box_clutter(spot[0], spot[1], 0.9, 0.6, rng.uniform(0.5, 1.4), z_bottom=0.15, name=f"pallet{k}_box")
    elif archetype == "large_warehouse":
        # two rack rows along x; a segment is left out now and then, which opens a cross aisle
        for row, y in enumerate((2.6, 5.4)):
            x0 = 1.0 + rng.uniform(0.0, 0.4)
            for k in range(5):
                if rng.uniform() < 0.8:
                    b.shelf(x0 + k * 1.95, y, 0.0, name=f"shelf_r{row}_{k}")
        for k in range(int(rng.integers(5, 9))):
            kind = rng.choice(["pallet", "box", "barrel", "stack"], p=[0.3, 0.35, 0.2, 0.15])
            if kind == "pallet":
                spot = _free_floor_spot(b, rng, W, D, 1.2, 0.8)
                if spot:
                    b.pallet(*spot, name=f"pallet{k}")
                    if rng.uniform() < 0.7:
                        b.box_clutter(spot[0], spot[1], 0.8, 0.6, rng.uniform(0.4, 1.2), z_bottom=0.15, name=f"pallet{k}_box")
            elif kind == "box":
                s_ = rng.uniform(0.4, 0.9, size=3)
                spot = _free_floor_spot(b, rng, W, D, s_[0], s_[1])
                if spot:
                    b.box_clutter(*spot, *s_, name=f"box{k}")
            elif kind == "stack":
                s_ = rng.uniform(0.5, 0.8, size=2)
                spot = _free_floor_spot(b, rng, W, D, s_[0], s_[1])
                if spot:
                    b.box_clutter(*spot, s_[0], s_[1], rng.uniform(1.2, 2.0), name=f"stack{k}")
            else:
                spot = _free_floor_spot(b, rng, W, D, 0.6, 0.6)
                if spot:
                    b.barrel(*spot, name=f"barrel{k}")
    else:
        raise ValueError(archetype)
    # surface clutter on some anchor surfaces
    for surf in b.surfaces:
        if rng.uniform() < 0.5:
            b.surface_clutter(surf, int(rng.integers(1, 3)))
    b.meta = {"W": W, "D": D, "Hc": Hc}
    return b

