"""Pure-numpy (float32) reference implementation of the WNNM denoiser.

Mirrors VapourSynth-WNNM (source/wnnm.cpp, v3-0-g52c764c) for a single plane,
following the spec in scratchpad/wnnm_spec.md. It is a *reference*: written
for readability and numerical fidelity, not speed.

Known, deliberate deviations from the compiled C++ oracle (tolerance-tested):
  1. Block-distance accumulation is strictly sequential over the patch element
     index p (f32), vectorized across candidates. The AVX2 C++ build uses
     per-lane FMA + horizontal add (different rounding).
  2. Sorting is stable by (error, insertion index); C++ std::partial_sort has
     unspecified order for exact float ties. This matters in practice for
     radius > 0 at clip boundaries: temporal clamping duplicates frame data,
     so the same block position ties EXACTLY across several t values, and the
     winning t decides which temporal slab receives the (identical) patch.
     Observed effect: up to ~3e-4 abs diff on the first/last `radius` output
     frames vs the oracle; interior frames match to ~1e-5.
  3. SVD is np.linalg.svd (LAPACK sgesdd from numpy's BLAS); the oracle links
     MKL sgesdd. Same algorithm family, not bit-identical.
  4. radius==0 final aggregation uses exact division `wdst / weight`; the AVX2
     C++ build uses approx_recipr (_mm256_rcp_ps, ~2^-12 rel. error) for the
     SIMD-width-aligned part of each row and exact division for the tail.
     (radius>0 VAggregate division is exact scalar in C++ and here.)
"""

import numpy as np

FLT_EPSILON = np.float32(np.finfo(np.float32).eps)

__all__ = ["wnnm_ref"]


def _stable_sort_errors(entries):
    """Sort (error, ...) tuples ascending by error; stable for exact ties
    (Python's sort is stable, so ties keep insertion order)."""
    return sorted(entries, key=lambda e: e[0])


def _block_distances(current, ref, coords, bs):
    """Squared-difference block distances of `current` (flattened bs*bs, f32)
    against blocks of `ref` at (x, y) in `coords`.

    Accumulation is strictly sequential over the patch element index p, in f32,
    vectorized across candidates (spec: port deviation 4)."""
    m = bs * bs
    ncand = len(coords)
    nb = np.empty((ncand, m), dtype=np.float32)
    for i, (bx, by) in enumerate(coords):
        nb[i] = ref[by:by + bs, bx:bx + bs].reshape(m)
    acc = np.zeros(ncand, dtype=np.float32)
    for p in range(m):
        d = current[p] - nb[:, p]
        acc += d * d
    return acc


def _search_locations(seeds, ps_range, bs, width, height):
    """Union of clamped (2*ps_range+1)^2 windows around the seed positions,
    sorted ascending by (y, x), deduplicated (C++ generate_search_locations)."""
    locs = set()
    for (_err, cx, cy) in seeds:
        left = max(cx - ps_range, 0)
        right = min(cx + ps_range, width - bs)
        top = max(cy - ps_range, 0)
        bottom = min(cy + ps_range, height - bs)
        for k in range(top, bottom + 1):
            for l in range(left, right + 1):
                locs.add((l, k))
    return sorted(locs, key=lambda t: (t[1], t[0]))


def _block_matching(refps, x, y, bs, gs, bm_range, ps_num, ps_range):
    """Returns the selected group: list of (error, bm_x, bm_y, bm_t), length
    active_group_size, plus the full stable-sorted error list (for debugging).
    """
    radius = (len(refps) - 1) // 2
    H, W = refps[0].shape
    m = bs * bs

    current = refps[radius][y:y + bs, x:x + bs].reshape(m).copy()

    # spatial search on the center frame, scan order (row-major)
    top = max(y - bm_range, 0)
    bottom = min(y + bm_range, H - bs)
    left = max(x - bm_range, 0)
    right = min(x + bm_range, W - bs)
    coords = [(bx, by) for by in range(top, bottom + 1)
              for bx in range(left, right + 1)]
    dists = _block_distances(current, refps[radius], coords, bs)
    center_errors = [(dists[i], bx, by) for i, (bx, by) in enumerate(coords)]

    if radius == 0:
        errors = [(e, bx, by, 0) for (e, bx, by) in center_errors]
    else:
        active_ps_num = min(ps_num, len(center_errors))
        active_num = min(max(gs, ps_num), len(center_errors))
        center_sorted = _stable_sort_errors(center_errors)[:active_num]
        errors = [(e, bx, by, radius) for (e, bx, by) in center_sorted]

        for direction in (-1, 1):
            temporal_errors = list(center_sorted)
            for i in range(1, radius + 1):
                t_idx = radius + direction * i
                seeds = temporal_errors[:min(active_ps_num, len(temporal_errors))]
                locs = _search_locations(seeds, ps_range, bs, W, H)
                tdists = _block_distances(current, refps[t_idx], locs, bs)
                temporal_errors = [(tdists[j], lx, ly)
                                   for j, (lx, ly) in enumerate(locs)]
                atn = min(max(gs, ps_num), len(temporal_errors))
                temporal_errors = _stable_sort_errors(temporal_errors)[:atn]
                errors.extend((e, lx, ly, t_idx)
                              for (e, lx, ly) in temporal_errors)

    active_group_size = min(gs, len(errors))
    errors_sorted = _stable_sort_errors(errors)
    sel = errors_sorted[:active_group_size]

    # center-block guarantee: if (x, y, t=center) missing, overwrite the BEST entry
    if not any(bx == x and by == y and bt == radius
               for (_e, bx, by, bt) in sel):
        sel[0] = (np.float32(0.0), x, y, radius)

    return sel, errors_sorted


def _patch_estimation(A, sigma_w, residual, adaptive_aggregation):
    """WNNP shrinkage on the patch matrix A (m x n, f32, columns = blocks).
    Returns (A_denoised, adaptive_weight, dbg) or (None, None, dbg) if the SVD
    failed (caller must use the patch_estimation_skip path)."""
    m, n = A.shape
    try:
        U, s, Vt = np.linalg.svd(A, full_matrices=False)  # sgesdd, f32
    except np.linalg.LinAlgError:
        return None, None, {"svd_failed": True}

    s = np.ascontiguousarray(s, dtype=np.float32)
    s_pre = s.copy()

    # constant = 8.f * sqrtf(2.0f * n) * square(sigma)   (all f32)
    c = (np.float32(8.0) * np.float32(np.sqrt(np.float32(2.0) * np.float32(n)))) \
        * (sigma_w * sigma_w)

    k0 = 0 if residual else 1  # largest SV untouched when not residual
    k = k0
    mn = min(m, n)
    while k < mn:
        sv = s[k]
        tmp = sv * sv - c
        if tmp > np.float32(0.0):
            s[k] = (sv + np.float32(np.sqrt(tmp))) * np.float32(0.5)
            k += 1
        else:
            break
    # k = number of KEPT components (s[0] counts as kept when not residual)

    if adaptive_aggregation:
        aw = np.float32(1.0) / np.float32(k) if k > 0 else np.float32(1.0)
    else:
        aw = np.float32(1.0)

    if k > 0:
        Ap = (U[:, :k] * s[:k][None, :]) @ Vt[:k, :]
        Ap = np.ascontiguousarray(Ap, dtype=np.float32)
    else:
        Ap = np.zeros((m, n), dtype=np.float32)

    dbg = {"svd_failed": False, "s_pre": s_pre, "s_post": s.copy(), "k": k,
           "c": c, "aw": aw}
    return Ap, aw, dbg


def _wnnm_raw_frame(srcps, refps, sigma_w, bs, step, gs, bm_range,
                    ps_num, ps_range, residual, adaptive_aggregation,
                    trace=None):
    """Process one output frame of WNNMRaw for a single plane.

    srcps/refps: lists of (H, W) f32 arrays, length 2*radius+1 (center=radius).
    Returns the intermediate accumulator of shape (2*(2*radius+1), H, W):
    slab 2*t = weighted sums, slab 2*t+1 = weights. For radius==0 the caller
    divides slab 0 by slab 1; for radius>0 this is the tall raw frame content.
    """
    nt = len(srcps)
    radius = (nt - 1) // 2
    H, W = srcps[0].shape
    m = bs * bs

    inter = np.zeros((2 * nt, H, W), dtype=np.float32)

    temp_r = H - bs
    temp_c = W - bs

    _y = 0
    while _y < temp_r + step:
        y = min(_y, temp_r)
        _x = 0
        while _x < temp_c + step:
            x = min(_x, temp_c)

            sel, errors_sorted = _block_matching(
                refps, x, y, bs, gs, bm_range, ps_num, ps_range)
            ags = len(sel)

            # patch load: column i = block i flattened row-major, from srcps
            A = np.empty((m, ags), dtype=np.float32)
            for i, (_e, bx, by, bt) in enumerate(sel):
                A[:, i] = srcps[bt][by:by + bs, bx:bx + bs].reshape(m)

            mean_patch = None
            if residual:
                mean_patch = np.zeros(m, dtype=np.float32)
                for i in range(ags):          # sequential accumulation (C++ order)
                    mean_patch += A[:, i]
                mean_patch /= np.float32(ags)
                for i in range(ags):
                    A[:, i] -= mean_patch

            Ap, aw, dbg = _patch_estimation(
                A, sigma_w, residual, adaptive_aggregation)

            if Ap is None:
                # patch_estimation_skip: aggregate ORIGINAL src blocks, weight 1
                for (_e, bx, by, bt) in sel:
                    inter[2 * bt, by:by + bs, bx:bx + bs] += \
                        srcps[bt][by:by + bs, bx:bx + bs]
                    inter[2 * bt + 1, by:by + bs, bx:bx + bs] += np.float32(1.0)
            else:
                if residual:
                    Ap = Ap + mean_patch[:, None]
                # col2im
                for i, (_e, bx, by, bt) in enumerate(sel):
                    patch = Ap[:, i].reshape(bs, bs)
                    inter[2 * bt, by:by + bs, bx:bx + bs] += patch * aw
                    inter[2 * bt + 1, by:by + bs, bx:bx + bs] += aw

            if trace is not None:
                cutoff = None
                if len(errors_sorted) > ags:
                    cutoff = (errors_sorted[ags - 1][0], errors_sorted[ags][0])
                trace.setdefault((x, y), []).append({
                    "sel": [(float(e), bx, by, bt) for (e, bx, by, bt) in sel],
                    "cutoff": cutoff,
                    **{k: v for k, v in dbg.items()},
                })

            _x += step
        _y += step

    return inter


def _vaggregate(raws, radius):
    """Combine the tall WNNMRaw frames into output frames (C++ VAggregate)."""
    T = len(raws)
    _, H, W = raws[0].shape
    out = np.empty((T, H, W), dtype=np.float32)
    for n in range(T):
        num = np.zeros((H, W), dtype=np.float32)
        den = np.zeros((H, W), dtype=np.float32)
        for i in range(2 * radius + 1):
            fid = min(max(n - radius + i, 0), T - 1)
            lo = n - T + 1 + radius
            hi = n + radius
            slab = min(max(2 * radius - i, lo), hi)
            num += raws[fid][2 * slab]
            den += raws[fid][2 * slab + 1]
        out[n] = num / den
    return out


def wnnm_ref(src, sigma=3.0, block_size=8, block_step=None, group_size=8,
             bm_range=7, radius=0, ps_num=2, ps_range=4, residual=False,
             adaptive_aggregation=True, rclip=None, trace=None):
    """Reference WNNM for a single plane.

    src: (T, H, W) or (H, W) array (converted to f32).
    rclip: optional same-shape array; used only for block matching.
    trace: optional dict; filled with per-frame, per-position debug info
           (selected block group, singular values, k, cutoff gap).
    Returns the denoised array with the same leading shape as `src`.
    """
    src = np.ascontiguousarray(np.asarray(src, dtype=np.float32))
    single = src.ndim == 2
    if single:
        src = src[None]
    if src.ndim != 3:
        raise ValueError("src must be (T, H, W) or (H, W)")
    T, H, W = src.shape

    if rclip is not None:
        ref = np.ascontiguousarray(np.asarray(rclip, dtype=np.float32))
        if ref.ndim == 2:
            ref = ref[None]
        if ref.shape != src.shape:
            raise ValueError("rclip must have the same shape as src")
    else:
        ref = src

    bs = int(block_size)
    step = bs if block_step is None else int(block_step)
    gs = int(group_size)

    # parameter validation (mirrors WNNMRawCreate; bs>dims rejected per port note)
    if bs <= 0 or bs > min(H, W):
        raise ValueError("block_size must be in 1..min(H, W)")
    if step < 1 or step > bs:
        raise ValueError("block_step must be in 1..block_size")
    if gs < 1:
        raise ValueError("group_size must be positive")
    if bm_range < 0 or radius < 0 or ps_range < 0:
        raise ValueError("bm_range/radius/ps_range must be non-negative")
    if ps_num < 1:
        raise ValueError("ps_num must be positive")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    if radius > 0 and T < 1:
        raise ValueError("empty clip")

    # plane processed iff sigma >= FLT_EPSILON, checked BEFORE scaling
    if np.float32(sigma) < FLT_EPSILON:
        out = src.copy()
        return out[0] if single else out

    sigma_w = np.float32(sigma) / np.float32(255.0)

    if radius == 0:
        out = np.empty_like(src)
        for n in range(T):
            tr = None if trace is None else trace.setdefault(n, {})
            inter = _wnnm_raw_frame(
                [src[n]], [ref[n]], sigma_w, bs, step, gs, bm_range,
                ps_num, ps_range, residual, adaptive_aggregation, tr)
            out[n] = inter[0] / inter[1]
        return out[0] if single else out

    # radius > 0: tall raw frames, then VAggregate
    raws = []
    for n in range(T):
        idxs = [min(max(n + i, 0), T - 1) for i in range(-radius, radius + 1)]
        tr = None if trace is None else trace.setdefault(n, {})
        raws.append(_wnnm_raw_frame(
            [src[j] for j in idxs], [ref[j] for j in idxs], sigma_w,
            bs, step, gs, bm_range, ps_num, ps_range, residual,
            adaptive_aggregation, tr))
    out = _vaggregate(raws, radius)
    return out[0] if single else out
