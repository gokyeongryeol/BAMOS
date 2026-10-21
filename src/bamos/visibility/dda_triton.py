"""Triton kernel: exact Amanatides-Woo DDA transmittance from one camera position to many points.

Each program instance handles BLOCK rays. For ray m (to the point y_m):
  log T_m = sum_{s in Pre} log_free[s], Pre = voxels the segment p -> y_m enters before the last `margin` of its length,
  excluding (optionally) the voxel containing p; segments are clipped to the grid. The margin keeps the surface the
  point rests on (a shelf board sharing the point's voxel) from counting as in front of it.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except Exception:  # pragma: no cover
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _dda_kernel(
        ends_ptr, logfree_ptr, out_ptr, params_ptr,
        nx, ny, nz, cam_lin, n_rays, max_iter,
        BLOCK: tl.constexpr,
    ):
        # float64 scalars are passed through a small tensor (Python float args would be fp32)
        px = tl.load(params_ptr + 0)
        py = tl.load(params_ptr + 1)
        pz = tl.load(params_ptr + 2)
        lx = tl.load(params_ptr + 3)
        ly = tl.load(params_ptr + 4)
        lz = tl.load(params_ptr + 5)
        vs = tl.load(params_ptr + 6)
        margin = tl.load(params_ptr + 7)
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_rays
        cx = tl.load(ends_ptr + offs * 3 + 0, mask=mask, other=0.0)
        cy = tl.load(ends_ptr + offs * 3 + 1, mask=mask, other=0.0)
        cz = tl.load(ends_ptr + offs * 3 + 2, mask=mask, other=0.0)
        dx = cx - px
        dy = cy - py
        dz = cz - pz
        L = tl.sqrt(dx * dx + dy * dy + dz * dz)
        Ls = tl.where(L > 0, L, 1.0)
        ux = dx / Ls
        uy = dy / Ls
        uz = dz / Ls
        tiny = 1e-15
        big = 1e30
        parx = tl.abs(ux) < tiny
        pary = tl.abs(uy) < tiny
        parz = tl.abs(uz) < tiny
        sux = tl.where(parx, tiny, ux)
        suy = tl.where(pary, tiny, uy)
        suz = tl.where(parz, tiny, uz)
        hx = lx + nx * vs
        hy = ly + ny * vs
        hz = lz + nz * vs
        # slab clipping
        tax = (lx - px) / sux
        tbx = (hx - px) / sux
        tay = (ly - py) / suy
        tby = (hy - py) / suy
        taz = (lz - pz) / suz
        tbz = (hz - pz) / suz
        tminx = tl.minimum(tax, tbx)
        tmaxx = tl.maximum(tax, tbx)
        tminy = tl.minimum(tay, tby)
        tmaxy = tl.maximum(tay, tby)
        tminz = tl.minimum(taz, tbz)
        tmaxz = tl.maximum(taz, tbz)
        inx = (px >= lx) & (px < hx)
        iny = (py >= ly) & (py < hy)
        inz = (pz >= lz) & (pz < hz)
        tminx = tl.where(parx, tl.where(inx, -big, big), tminx)
        tmaxx = tl.where(parx, tl.where(inx, big, -big), tmaxx)
        tminy = tl.where(pary, tl.where(iny, -big, big), tminy)
        tmaxy = tl.where(pary, tl.where(iny, big, -big), tmaxy)
        tminz = tl.where(parz, tl.where(inz, -big, big), tminz)
        tmaxz = tl.where(parz, tl.where(inz, big, -big), tmaxz)
        t0 = tl.maximum(tl.maximum(tminx, tminy), tl.maximum(tminz, 0.0))
        t1 = tl.minimum(tl.minimum(tmaxx, tmaxy), tl.minimum(tmaxz, L - margin))
        active = mask & (t0 <= t1) & (L > 0)
        eps = 1e-9
        sx = px + ux * (t0 + eps)
        sy = py + uy * (t0 + eps)
        sz = pz + uz * (t0 + eps)
        i = tl.floor((sx - lx) / vs).to(tl.int64)
        j = tl.floor((sy - ly) / vs).to(tl.int64)
        k = tl.floor((sz - lz) / vs).to(tl.int64)
        i = tl.minimum(tl.maximum(i, 0), nx - 1)
        j = tl.minimum(tl.maximum(j, 0), ny - 1)
        k = tl.minimum(tl.maximum(k, 0), nz - 1)
        stx = tl.where(ux > 0, 1, -1).to(tl.int64)
        sty = tl.where(uy > 0, 1, -1).to(tl.int64)
        stz = tl.where(uz > 0, 1, -1).to(tl.int64)
        tdx = tl.where(parx, big, vs / tl.abs(sux))
        tdy = tl.where(pary, big, vs / tl.abs(suy))
        tdz = tl.where(parz, big, vs / tl.abs(suz))
        nbx = lx + (i + tl.where(ux > 0, 1, 0)).to(tl.float64) * vs
        nby = ly + (j + tl.where(uy > 0, 1, 0)).to(tl.float64) * vs
        nbz = lz + (k + tl.where(uz > 0, 1, 0)).to(tl.float64) * vs
        tmx = tl.where(parx, big, (nbx - px) / sux)
        tmy = tl.where(pary, big, (nby - py) / suy)
        tmz = tl.where(parz, big, (nbz - pz) / suz)
        acc = tl.zeros([BLOCK], dtype=tl.float64)
        for _ in range(max_iter):
            inb = (i >= 0) & (i < nx) & (j >= 0) & (j < ny) & (k >= 0) & (k < nz)
            lin = (i * ny + j) * nz + k
            lin_safe = tl.where(inb, lin, 0)
            use = active & inb & (lin != cam_lin)
            val = tl.load(logfree_ptr + lin_safe, mask=use, other=0.0)
            acc = acc + tl.where(use, val, 0.0)
            # choose axis
            ax = (tmx <= tmy) & (tmx <= tmz)
            ay = (~ax) & (tmy <= tmz)
            az = (~ax) & (~ay)
            tnext = tl.where(ax, tmx, tl.where(ay, tmy, tmz))
            stop = tnext > t1
            active = active & (~stop)
            i = i + tl.where(active & ax, stx, 0)
            j = j + tl.where(active & ay, sty, 0)
            k = k + tl.where(active & az, stz, 0)
            tmx = tmx + tl.where(active & ax, tdx, 0.0)
            tmy = tmy + tl.where(active & ay, tdy, 0.0)
            tmz = tmz + tl.where(active & az, tdz, 0.0)
        tl.store(out_ptr + offs, acc, mask=mask)


def dda_log_transmittance(ends: torch.Tensor, log_free: torch.Tensor, position, lower, voxel_size: float, shape,
                          cam_lin: int, max_iter: int, margin: float) -> torch.Tensor:
    """ends [M,3] float64 cuda; log_free [n_vox] float64 -> out [M] float64."""
    assert HAVE_TRITON and ends.is_cuda
    M = ends.shape[0]
    out = torch.empty((M,), dtype=torch.float64, device=ends.device)
    if M == 0:
        return out
    BLOCK = 128
    grid = (triton.cdiv(M, BLOCK),)
    lf = log_free if log_free.dtype == torch.float64 else log_free.to(torch.float64)
    e = ends.to(torch.float64).contiguous()
    params = torch.tensor(
        [float(position[0]), float(position[1]), float(position[2]), float(lower[0]), float(lower[1]), float(lower[2]),
         float(voxel_size), float(margin)], dtype=torch.float64, device=ends.device,
    )
    with torch.cuda.device(ends.device):  # Triton launches on the current device; tensors may live on cuda:k
        _dda_kernel[grid](
            e, lf, out, params,
            int(shape[0]), int(shape[1]), int(shape[2]), int(cam_lin), M, int(max_iter),
            BLOCK=BLOCK,
        )
    return out
