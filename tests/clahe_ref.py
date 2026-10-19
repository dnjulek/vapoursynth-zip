"""NumPy reference for vszip.CLAHE on one plane, any bin count.

Mirrors the algorithm (not the code layout) of src/filters/clahe.zig with the
same f32 operation order, so integer outputs must match exactly and float
outputs bit-for-bit:

- bins: integer samples by value (min(v, peak) for 9..15-bit), float samples
  quantized to 16-bit bins as trunc(clamp((v + off) * 65535 + 0.5, 0, 65535)),
  off = 0.5 on YUV chroma, NaN -> bin 0;
- per tile: histogram, clip at clip_limit = max(limit * tile_area // hist, 1),
  redistribute (batch to every bin, residual one-per-step from bin 0),
  inclusive cumsum -> lut = trunc(cdf * (peak / tile_area) + 0.5);
- bilinear blend of the 4 nearest tile LUTs at each pixel's bin, rounded
  (int) or mapped back as res / 65535 - off (float, unrounded).
"""

import numpy as np

f32 = np.float32


def bins_of(plane, bits, is_float, off):
    peak = (1 << bits) - 1
    if is_float:
        x = (plane.astype(f32) + f32(off)) * f32(peak) + f32(0.5)
        x = np.where(np.isnan(x), f32(0), x)
        return np.trunc(np.minimum(np.maximum(x, f32(0)), f32(peak))).astype(np.int64)
    return np.minimum(plane.astype(np.int64), peak)


def clahe_plane(plane, bits, limit, tiles, is_float=False, off=0.0, out_dtype=None):
    h, w = plane.shape
    hist = 1 << bits
    peak = f32(hist - 1)
    tx_n, ty_n = tiles
    tw, th = w // tx_n, h // ty_n
    area = tw * th
    clip_limit = max(limit * area // hist, 1)
    scale = f32(peak / f32(area))
    b = bins_of(plane, bits, is_float, off)

    luts = np.empty((ty_n, tx_n, hist), dtype=np.int64)
    for ty in range(ty_n):
        for tx in range(tx_n):
            tile = b[ty * th:min((ty + 1) * th, h), tx * tw:min((tx + 1) * tw, w)]
            hg = np.bincount(tile.ravel(), minlength=hist).astype(np.int64)
            clipped = int(np.maximum(hg - clip_limit, 0).sum())
            hg = np.minimum(hg, clip_limit)
            batch = clipped // hist
            residual = clipped - batch * hist
            hg += batch
            if residual:
                step = max(hist // residual, 1)
                i = 0
                while i < hist and residual > 0:
                    hg[i] += 1
                    residual -= 1
                    i += step
            cdf = np.cumsum(hg).astype(f32)
            luts[ty, tx] = np.trunc(cdf * scale + f32(0.5)).astype(np.int64)

    inv_tw, inv_th = f32(1) / f32(tw), f32(1) / f32(th)
    txf = np.arange(w, dtype=f32) * inv_tw - f32(0.5)
    tx1 = np.floor(txf).astype(np.int64)
    xa = (txf - tx1.astype(f32)).astype(f32)
    tx2 = np.minimum(tx1 + 1, tx_n - 1)
    tx1 = np.clip(tx1, 0, tx_n - 1)
    out = np.empty((h, w), dtype=f32)
    for y in range(h):
        tyf = f32(y) * inv_th - f32(0.5)
        ty1 = int(np.floor(tyf))
        ya = f32(tyf - f32(ty1))
        ty2 = min(ty1 + 1, ty_n - 1)
        ty1 = min(max(ty1, 0), ty_n - 1)
        v = b[y]
        l0 = luts[ty1, tx1, v].astype(f32)
        l1 = luts[ty1, tx2, v].astype(f32)
        l2 = luts[ty2, tx1, v].astype(f32)
        l3 = luts[ty2, tx2, v].astype(f32)
        one_m_xa = f32(1) - xa
        out[y] = (l0 * one_m_xa + l1 * xa) * (f32(1) - ya) + (l2 * one_m_xa + l3 * xa) * ya
    if is_float:
        res = out * (f32(1) / peak) - f32(off)
        return res.astype(out_dtype or f32)
    return np.trunc(out + f32(0.5)).astype(out_dtype or np.uint16)
