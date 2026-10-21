"""Exact analytic ray caster for the primitive scene description.

Produces optical-axis depth and an instance-id image for a CameraModel/pose, and target-only silhouettes.
Used as the evaluator's GT geometry (silhouette, solvability) and to validate the Isaac adapter's depth.
"""
from __future__ import annotations

import numpy as np

from bamos.environment.scene import Box, Cylinder, SceneDescription, SceneObject, Sphere
from bamos.mapping.camera import CameraModel, inverse_transform


def _ray_box(o: np.ndarray, d: np.ndarray, box: Box) -> np.ndarray:
    """Entry parameter t (>0) per ray for an oriented box; inf if missed. o [N,3] d [N,3]."""
    c = np.asarray(box.center)
    h = 0.5 * np.asarray(box.size)
    cy, sy = np.cos(box.yaw), np.sin(box.yaw)
    R = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])  # local->world
    ol = (o - c) @ R  # world->local
    dl = d @ R
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / dl
        t1 = (-h - ol) * inv
        t2 = (h - ol) * inv
    tmin = np.where(np.isfinite(inv), np.minimum(t1, t2), np.where((np.abs(ol) <= h), -np.inf, np.inf))
    tmax = np.where(np.isfinite(inv), np.maximum(t1, t2), np.where((np.abs(ol) <= h), np.inf, -np.inf))
    t_in = tmin.max(axis=1)
    t_out = tmax.min(axis=1)
    hit = (t_out >= np.maximum(t_in, 0.0)) & (t_out > 0)
    t = np.where(t_in > 0, t_in, t_out)  # origin inside -> exit face
    return np.where(hit, t, np.inf)


def _ray_cylinder(o: np.ndarray, d: np.ndarray, cyl: Cylinder) -> np.ndarray:
    c = np.asarray(cyl.center)
    r = cyl.radius
    hz = 0.5 * cyl.height
    ox, oy, oz = (o - c).T
    dx, dy, dz = d.T
    a = dx * dx + dy * dy
    b = 2.0 * (ox * dx + oy * dy)
    cc = ox * ox + oy * oy - r * r
    disc = b * b - 4 * a * cc
    t_best = np.full(len(o), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        sq = np.sqrt(np.maximum(disc, 0.0))
        ta = (-b - sq) / (2 * a)
        tb = (-b + sq) / (2 * a)
        for t in (ta, tb):
            z = oz + t * dz
            ok = (disc >= 0) & (a > 1e-15) & (t > 0) & (np.abs(z) <= hz)
            t_best = np.where(ok & (t < t_best), t, t_best)
        # caps
        for zc in (-hz, hz):
            t = (zc - oz) / dz
            x = ox + t * dx
            y = oy + t * dy
            ok = np.isfinite(t) & (t > 0) & (x * x + y * y <= r * r)
            t_best = np.where(ok & (t < t_best), t, t_best)
    return t_best


def _ray_sphere(o: np.ndarray, d: np.ndarray, sph: Sphere) -> np.ndarray:
    """Nearest t > 0 per ray on the sphere surface (the exit point if the origin is inside); inf if missed."""
    rel = o - np.asarray(sph.center)
    a = np.sum(d * d, axis=1)
    b = 2.0 * np.sum(rel * d, axis=1)
    cc = np.sum(rel * rel, axis=1) - sph.radius ** 2
    disc = b * b - 4 * a * cc
    with np.errstate(invalid="ignore"):
        sq = np.sqrt(np.maximum(disc, 0.0))
        t0 = (-b - sq) / (2 * a)
        t1 = (-b + sq) / (2 * a)
    t = np.where(t0 > 0, t0, t1)
    return np.where((disc >= 0) & (t > 0), t, np.inf)


def _prim_aabb_cull(o: np.ndarray, d: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Boolean [N] whether the ray may hit the AABB (slab test), used to prune ray/primitive work."""
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        t1 = (lo - o) * inv
        t2 = (hi - o) * inv
    tmin = np.where(np.isfinite(inv), np.minimum(t1, t2), np.where((o >= lo) & (o <= hi), -np.inf, np.inf))
    tmax = np.where(np.isfinite(inv), np.maximum(t1, t2), np.where((o >= lo) & (o <= hi), np.inf, -np.inf))
    return (tmax.min(axis=1) >= np.maximum(tmin.max(axis=1), 0.0)) & (tmax.min(axis=1) > 0)


def cast_rays(origins: np.ndarray, dirs: np.ndarray, objects: list[SceneObject]) -> tuple[np.ndarray, np.ndarray]:
    """Nearest hit parameter t [N] (inf if none) and instance id [N] (-1 if none). dirs need not be unit."""
    N = len(dirs)
    t_best = np.full(N, np.inf)
    id_best = np.full(N, -1, dtype=np.int64)
    for obj in objects:
        for prim in obj.prims:
            lo, hi = prim.aabb()
            cand = _prim_aabb_cull(origins, dirs, lo, hi)
            if not cand.any():
                continue
            idx = np.nonzero(cand)[0]
            if isinstance(prim, Box):
                t = _ray_box(origins[idx], dirs[idx], prim)
            elif isinstance(prim, Sphere):
                t = _ray_sphere(origins[idx], dirs[idx], prim)
            else:
                t = _ray_cylinder(origins[idx], dirs[idx], prim)
            better = t < t_best[idx]
            t_best[idx[better]] = t[better]
            id_best[idx[better]] = obj.instance_id
    return t_best, id_best


class AnalyticRenderer:
    def __init__(self, scene: SceneDescription):
        self.scene = scene

    def render(self, camera: CameraModel, world_from_camera: np.ndarray, objects: list[SceneObject] | None = None, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Returns (depth [H,W] optical-axis, instance id image [H,W]); misses -> inf depth, id -1.

        Depth is z along the optical axis: with camera-frame direction (x, y, 1) the hit point is t*(x,y,1) so
        depth = t (dirs have unit z).
        """
        objs = self.scene.objects if objects is None else objects
        dirs_cam, uv = camera.pixel_rays(stride)
        R = world_from_camera[:3, :3]
        o = world_from_camera[:3, 3]
        dirs_w = dirs_cam @ R.T
        origins = np.repeat(o[None, :], len(dirs_w), axis=0)
        t, ids = cast_rays_auto(origins, dirs_w, objs)
        H = (camera.height + stride - 1) // stride
        W = (camera.width + stride - 1) // stride
        depth = t.reshape(H, W)
        depth = np.where(depth < camera.near, np.inf, depth)  # hits closer than near are invalid
        return depth, ids.reshape(H, W)

    def target_silhouette(self, camera: CameraModel, world_from_camera: np.ndarray) -> np.ndarray:
        """Target-only render (occluders removed): bool mask [H,W]."""
        tgt = self.scene.target()
        lo, hi = tgt.aabb()
        # restrict to the target's projected bounding window for speed
        depth, ids = self.render_window(camera, world_from_camera, [tgt], lo, hi)
        return ids == tgt.instance_id

    def render_window(self, camera: CameraModel, world_from_camera: np.ndarray, objects: list[SceneObject], lo: np.ndarray, hi: np.ndarray):
        """Render full-size images but only cast rays inside the projected AABB window of [lo, hi]."""
        H, W = camera.height, camera.width
        depth = np.full((H, W), np.inf)
        ids = np.full((H, W), -1, dtype=np.int64)
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        Tc = inverse_transform(world_from_camera)
        pc = corners @ Tc[:3, :3].T + Tc[:3, 3]
        if np.all(pc[:, 2] <= camera.near):
            return depth, ids
        if np.any(pc[:, 2] <= camera.near):
            u0, u1, v0, v1 = 0, W - 1, 0, H - 1  # crosses the image plane: cast everything
        else:
            uv, _ = camera.project(pc)
            u0 = int(max(0, np.floor(uv[:, 0].min())))
            u1 = int(min(W - 1, np.ceil(uv[:, 0].max())))
            v0 = int(max(0, np.floor(uv[:, 1].min())))
            v1 = int(min(H - 1, np.ceil(uv[:, 1].max())))
            if u1 < u0 or v1 < v0:
                return depth, ids
        vs, us = np.meshgrid(np.arange(v0, v1 + 1), np.arange(u0, u1 + 1), indexing="ij")
        us = us.ravel().astype(np.float64)
        vs = vs.ravel().astype(np.float64)
        dirs_cam = np.stack([(us - camera.cx) / camera.fx, (vs - camera.cy) / camera.fy, np.ones_like(us)], axis=1)
        R = world_from_camera[:3, :3]
        o = world_from_camera[:3, 3]
        t, idh = cast_rays_auto(np.repeat(o[None], len(dirs_cam), 0), dirs_cam @ R.T, objects)
        t = np.where(t < camera.near, np.inf, t)
        depth[vs.astype(int), us.astype(int)] = t
        ids[vs.astype(int), us.astype(int)] = idh
        return depth, ids

    def target_quality(self, camera: CameraModel, world_from_camera: np.ndarray) -> dict:
        """GT visible fraction nu = A_GT / A_iso, normalized areas and truncation for the target at this pose.

        A_GT: pixels where the target is the nearest surface in the full scene; A_iso: target-only silhouette.
        Computed on the target window only (full-scene occlusion test restricted to silhouette pixels).
        """
        tgt = self.scene.target()
        lo, hi = tgt.aabb()
        _, id_iso = self.render_window(camera, world_from_camera, [tgt], lo, hi)
        sil = id_iso == tgt.instance_id
        a_iso = int(sil.sum())
        if a_iso == 0:
            return {"visible_fraction": 0.0, "area_px": 0, "iso_area_px": 0, "normalized_area": 0.0, "truncated": False, "mask": sil}
        vs, us = np.nonzero(sil)
        dirs_cam = np.stack([(us - camera.cx) / camera.fx, (vs - camera.cy) / camera.fy, np.ones(len(us))], axis=1)
        R = world_from_camera[:3, :3]
        o = world_from_camera[:3, 3]
        _, ids = cast_rays_auto(np.repeat(o[None], len(dirs_cam), 0), dirs_cam @ R.T, self.scene.objects)
        vis = ids == tgt.instance_id
        mask = np.zeros_like(sil)
        mask[vs[vis], us[vis]] = True
        # truncation: silhouette touches the image border -> part of the target may be outside the frame
        truncated = bool(sil[0, :].any() or sil[-1, :].any() or sil[:, 0].any() or sil[:, -1].any())
        a_gt = int(vis.sum())
        return {
            "visible_fraction": a_gt / a_iso,
            "area_px": a_gt,
            "iso_area_px": a_iso,
            "normalized_area": a_gt / float(camera.n_pixels),
            "truncated": truncated,
            "mask": mask,
        }


# ----------------------------------------------------------------------------------------------- torch fast path
_TORCH_CACHE: dict = {}


def _prim_tensors(objects: list[SceneObject], device):
    """Pack boxes, cylinders and spheres of the object list into batched tensors (cached per object-list identity).

    The entry holds the objects themselves: an id() is only unique while its object is alive, and a key of bare
    ids let an object of a later episode, allocated at a freed address, read an earlier episode's geometry.
    """
    import torch

    key = (tuple(id(o) for o in objects), str(device))
    hit = _TORCH_CACHE.get(key)
    if hit is not None and len(hit[0]) == len(objects) and all(a is b for a, b in zip(hit[0], objects)):
        return hit[1]
    boxes, box_ids, cyls, cyl_ids, sphs, sph_ids = [], [], [], [], [], []
    for o in objects:
        for p in o.prims:
            if isinstance(p, Box):
                boxes.append((*p.center, *p.size, p.yaw))
                box_ids.append(o.instance_id)
            elif isinstance(p, Sphere):
                sphs.append((*p.center, p.radius))
                sph_ids.append(o.instance_id)
            else:
                cyls.append((*p.center, p.radius, p.height))
                cyl_ids.append(o.instance_id)
    f = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64).reshape(len(a), -1), dtype=torch.float64, device=device)
    out = {
        "boxes": f(boxes) if boxes else None, "box_ids": torch.as_tensor(np.asarray(box_ids, dtype=np.int64), device=device),
        "cyls": f(cyls) if cyls else None, "cyl_ids": torch.as_tensor(np.asarray(cyl_ids, dtype=np.int64), device=device),
        "sphs": f(sphs) if sphs else None, "sph_ids": torch.as_tensor(np.asarray(sph_ids, dtype=np.int64), device=device),
    }
    if len(_TORCH_CACHE) > 64:
        _TORCH_CACHE.clear()
    _TORCH_CACHE[key] = (tuple(objects), out)
    return out


def cast_rays_torch(origins: np.ndarray, dirs: np.ndarray, objects: list[SceneObject], device="cuda:0") -> tuple[np.ndarray, np.ndarray]:
    """Same semantics as cast_rays, batched over primitives on the GPU."""
    import torch

    P = _prim_tensors(objects, device)
    o = torch.as_tensor(np.asarray(origins, dtype=np.float64), device=device)
    d = torch.as_tensor(np.asarray(dirs, dtype=np.float64), device=device)
    N = o.shape[0]
    inf = float("inf")
    t_best = torch.full((N,), inf, dtype=torch.float64, device=device)
    id_best = torch.full((N,), -1, dtype=torch.int64, device=device)
    if P["boxes"] is not None:
        B = P["boxes"]  # [B, 7]
        c = B[:, 0:3]
        h = 0.5 * B[:, 3:6]
        yaw = B[:, 6]
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        # local coords: x' = R^T (x - c); R = [[cy,-sy,0],[sy,cy,0],[0,0,1]]
        rel = o[:, None, :] - c[None, :, :]  # [N,B,3]
        olx = rel[..., 0] * cy[None] + rel[..., 1] * sy[None]
        oly = -rel[..., 0] * sy[None] + rel[..., 1] * cy[None]
        olz = rel[..., 2]
        dlx = d[:, None, 0] * cy[None] + d[:, None, 1] * sy[None]
        dly = -d[:, None, 0] * sy[None] + d[:, None, 1] * cy[None]
        dlz = d[:, None, 2].expand(-1, B.shape[0])
        ol = torch.stack([olx, oly, olz], -1)
        dl = torch.stack([dlx, dly, dlz], -1)
        par = dl.abs() < 1e-15
        inv = 1.0 / torch.where(par, torch.full_like(dl, 1e-15), dl)
        t1 = (-h[None] - ol) * inv
        t2 = (h[None] - ol) * inv
        inside = ol.abs() <= h[None]
        tmin = torch.where(par, torch.where(inside, torch.full_like(t1, -inf), torch.full_like(t1, inf)), torch.minimum(t1, t2))
        tmax = torch.where(par, torch.where(inside, torch.full_like(t1, inf), torch.full_like(t1, -inf)), torch.maximum(t1, t2))
        t_in = tmin.max(dim=-1).values
        t_out = tmax.min(dim=-1).values
        hit = (t_out >= torch.clamp(t_in, min=0.0)) & (t_out > 0)
        t = torch.where(t_in > 0, t_in, t_out)
        t = torch.where(hit, t, torch.full_like(t, inf))  # [N,B]
        tb, ib = t.min(dim=1)
        better = tb < t_best
        t_best = torch.where(better, tb, t_best)
        id_best = torch.where(better, P["box_ids"][ib], id_best)
    if P["cyls"] is not None:
        C = P["cyls"]  # [C,5]
        c = C[:, 0:3]
        r = C[:, 3]
        hz = 0.5 * C[:, 4]
        rel = o[:, None, :] - c[None]
        ox, oy, oz = rel[..., 0], rel[..., 1], rel[..., 2]
        dx, dy, dz = d[:, None, 0], d[:, None, 1], d[:, None, 2]
        a = dx * dx + dy * dy
        b = 2.0 * (ox * dx + oy * dy)
        cc = ox * ox + oy * oy - (r * r)[None]
        disc = b * b - 4 * a * cc
        sq = torch.sqrt(torch.clamp(disc, min=0.0))
        a_safe = torch.where(a.abs() < 1e-15, torch.full_like(a, 1e-15), a)
        t_c = torch.full(disc.shape, inf, dtype=torch.float64, device=device)
        for tt in ((-b - sq) / (2 * a_safe), (-b + sq) / (2 * a_safe)):
            z = oz + tt * dz
            ok = (disc >= 0) & (a > 1e-15) & (tt > 0) & (z.abs() <= hz[None])
            t_c = torch.where(ok & (tt < t_c), tt, t_c)
        dz_safe = torch.where(dz.abs() < 1e-15, torch.full_like(dz, 1e-15), dz)
        for zc in (-hz[None], hz[None]):
            tt = (zc - oz) / dz_safe
            x = ox + tt * dx
            y = oy + tt * dy
            ok = (dz.abs() >= 1e-15) & (tt > 0) & (x * x + y * y <= (r * r)[None])
            t_c = torch.where(ok & (tt < t_c), tt, t_c)
        tc, ic = t_c.min(dim=1)
        better = tc < t_best
        t_best = torch.where(better, tc, t_best)
        id_best = torch.where(better, P["cyl_ids"][ic], id_best)
    if P["sphs"] is not None:
        Sp = P["sphs"]  # [S,4]
        rel = o[:, None, :] - Sp[None, :, 0:3]
        a = (d * d).sum(-1)[:, None]
        b = 2.0 * (rel * d[:, None, :]).sum(-1)
        cc = (rel * rel).sum(-1) - (Sp[:, 3] ** 2)[None]
        disc = b * b - 4 * a * cc
        sq = torch.sqrt(torch.clamp(disc, min=0.0))
        t0, t1 = (-b - sq) / (2 * a), (-b + sq) / (2 * a)
        tt = torch.where(t0 > 0, t0, t1)
        tt = torch.where((disc >= 0) & (tt > 0), tt, torch.full_like(tt, inf))
        ts, is_ = tt.min(dim=1)
        better = ts < t_best
        t_best = torch.where(better, ts, t_best)
        id_best = torch.where(better, P["sph_ids"][is_], id_best)
    return t_best.cpu().numpy(), id_best.cpu().numpy()


def set_cast_backend(device: str | None):
    """Select the ray-casting backend for AnalyticRenderer (None -> numpy reference)."""
    global _CAST_DEVICE
    _CAST_DEVICE = device


_CAST_DEVICE: str | None = None


def cast_rays_auto(origins, dirs, objects):
    if _CAST_DEVICE is not None and len(dirs) >= 2048:
        return cast_rays_torch(origins, dirs, objects, _CAST_DEVICE)
    return cast_rays(origins, dirs, objects)
