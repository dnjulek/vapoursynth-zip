"""Independent behavioral tests for the SIMD-heavy LimitFilter kernel."""

import pytest
import vapoursynth as vs

from helpers import assert_same_clip, diff, write_png


def _limit_scalar(flt: int, src: int, ref: int, dark: float, bright: float, elast: float) -> int:
    """Specification written independently of the vector implementation."""
    signed = float(flt - ref)
    distance = abs(signed)
    threshold = bright if signed > 0.0 else dark
    outer = threshold * elast
    if distance <= threshold:
        value = float(flt)
    elif distance >= outer:
        value = float(src)
    else:
        value = src + (flt - src) * (outer - distance) / (outer - threshold)
    return int(value + 0.5)


def _gray8(core, tmp_path, name: str, values: list[int], width: int):
    rows = [values[:width], values[width:]]
    return core.vszip.ImageRead(str(write_png(tmp_path / f"{name}.png", rows, color="gray")))


@pytest.mark.parametrize("with_ref", [False, True], ids=["implicit-ref-src", "explicit-ref"])
@pytest.mark.parametrize("width", [1, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65])
def test_scalar_oracle_across_vector_boundaries(core, tmp_path, width, with_ref):
    """Exercise scalar-only, exact-vector and vector-plus-tail row widths.

    The generated differences cover both signs and all three soft-limit
    regions: pass filtered, interpolate, and fall back to source.
    """
    count = width * 2
    deltas = (-40, -25, -17, -16, -15, -9, -8, -7, -1, 0, 1, 7, 8, 9, 15, 16, 17, 25, 40)
    reference = [64 + (i * 73) % 128 for i in range(count)]
    filtered = [max(0, min(255, reference[i] + deltas[i % len(deltas)])) for i in range(count)]
    source = [(31 + i * 47) % 256 for i in range(count)]

    flt = _gray8(core, tmp_path, "flt", filtered, width)
    src = _gray8(core, tmp_path, "src", source, width)
    ref_values = reference if with_ref else source
    ref = _gray8(core, tmp_path, "ref", reference, width)
    kwargs = dict(dark_thr=8, bright_thr=16, elast=3)
    out = flt.vszip.LimitFilter(src, ref, **kwargs) if with_ref else flt.vszip.LimitFilter(src, **kwargs)

    expected = [
        _limit_scalar(filtered[i], source[i], ref_values[i], 8, 16, 3)
        for i in range(count)
    ]
    with out.get_frame(0) as frame:
        actual = [frame[0][y, x] for y in range(2) for x in range(width)]
    assert actual == expected


def test_selected_planes_and_passthrough(core):
    src = core.std.BlankClip(None, 64, 32, vs.YUV444P8, length=2, color=[80, 100, 140])
    flt = core.std.BlankClip(src, color=[120, 130, 170])
    out = flt.vszip.LimitFilter(src, dark_thr=1, bright_thr=1, elast=2, planes=[0])

    # The selected plane is outside the soft window and therefore falls back
    # to src. Unselected planes are frame-shared directly from flt.
    assert diff(out, src, plane=0) == 0.0
    assert diff(out, flt, plane=1) == 0.0
    assert diff(out, flt, plane=2) == 0.0


def test_identity_is_exact(make_clip):
    src = make_clip(vs.YUV420P16, "odd")
    assert_same_clip(src.vszip.LimitFilter(src), src)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda c: c.std.Crop(right=2), "same width and height"),
        (lambda c: vs.core.std.BlankClip(None, 64, 32, vs.RGB24, length=2), "same color family"),
        (lambda c: c.resize.Point(format=vs.YUV444P8), "same subsampling"),
        (lambda c: c.resize.Point(format=vs.YUV420P10), "same bit depth"),
        (lambda c: c.std.Trim(0, 0), "same length"),
    ],
)
def test_clip_validation(core, change, message):
    flt = core.std.BlankClip(None, 64, 32, vs.YUV420P8, length=2)
    with pytest.raises(vs.Error, match=message):
        flt.vszip.LimitFilter(change(flt))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"planes": [-1]}, "plane index out of range"),
        ({"planes": [0, 0]}, "plane specified twice"),
        ({"dark_thr": [-1]}, "below minimum"),
        ({"bright_thr": [256]}, "above maximum"),
        ({"elast": [-1]}, "below minimum"),
        ({"dark_thr": [1, 2, 3, 4]}, "too many elements"),
    ],
)
def test_argument_validation(core, kwargs, message):
    clip = core.std.BlankClip(None, 64, 32, vs.YUV420P8, length=1)
    with pytest.raises(vs.Error, match=message):
        clip.vszip.LimitFilter(clip, **kwargs)


# ---------------------------------------------------------------------------
# float32-exact oracle: every format, every plane, both thresholds paths
# ---------------------------------------------------------------------------
#
# The kernel works in f32 for all sample types: f32 inputs, thresholds scaled
# from 8-bit units by hz.scaleValue, thr2 = thr1 * elast, and
# out = sf + (ff - sf) * (thr2 - |d|) / (thr2 - thr1) with no FMA contraction.
# NumPy float32 element-wise ops round exactly like that, so the oracle is
# bit-exact rather than tolerance-based. Integer output is trunc(out + 0.5).


def _np():
    return pytest.importorskip("numpy")


def _scaled_thr(np, thr8: float, fmt: vs.VideoFormat, full: bool):
    """hz.scaleValue(thr, clip, .{}) for an 8-bit-unit threshold. Chroma
    planes use the luma (chroma=false) scale too, exactly like the filter."""
    f32 = np.float32
    if fmt.bits_per_sample == 8 and fmt.sample_type == vs.INTEGER:
        return f32(thr8)
    in_peak, in_low = (255.0, 0.0) if full else (235.0, 16.0)
    if fmt.sample_type == vs.FLOAT:
        out_peak, out_low = 1.0, 0.0
    elif full:
        out_peak, out_low = float((1 << fmt.bits_per_sample) - 1), 0.0
    else:
        shift = fmt.bits_per_sample - 8
        out_peak, out_low = float(235 << shift), float(16 << shift)
    v = f32(thr8) * (f32(out_peak - out_low) / f32(in_peak - in_low))
    if fmt.sample_type == vs.INTEGER:
        v = f32(min(max(float(np.round(v)), 0.0), float((1 << fmt.bits_per_sample) - 1)))
    return v


def _oracle_plane(np, flt, src, ref, dark, bright, elast, is_int):
    f32 = np.float32
    ff, sf, rf = (a.astype(f32) for a in (flt, src, ref))
    signed = ff - rf
    dist = np.abs(signed)
    thr1 = np.where(signed > f32(0), bright, dark).astype(f32)
    thr2 = (thr1 * f32(elast)).astype(f32)
    with np.errstate(divide="ignore", invalid="ignore"):
        interp = sf + (ff - sf) * (thr2 - dist) / (thr2 - thr1)
    out = np.where(dist <= thr1, ff, np.where(dist >= thr2, sf, interp)).astype(f32)
    if is_int:
        return np.trunc(out + f32(0.5)).astype(flt.dtype)
    return out


def _clip_from_planes(core, fmt: vs.VideoFormat, planes, frame_props: dict):
    """One-frame clip whose planes are exactly the given NumPy arrays."""
    np = _np()
    h, w = planes[0].shape
    blank = core.std.BlankClip(None, w, h, fmt.id, length=1)

    def fill(n, f):
        fout = f.copy()
        for p, arr in enumerate(planes):
            np.asarray(fout[p])[...] = arr
        for k, v in frame_props.items():
            fout.props[k] = v
        return fout

    return blank.std.ModifyFrame(blank, fill)


def _planes_of(np, clip):
    with clip.get_frame(0) as f:
        return [np.array(f[p], copy=True) for p in range(f.format.num_planes)]


def _random_triplet(np, rng, fmt: vs.VideoFormat, w: int, h: int, thr_span: float):
    """flt/src/ref planes whose |flt - ref| sweeps all three soft-limit regions
    (pass, interpolate, fall back) for thresholds around `thr_span` (in the
    format's own units)."""
    out = {"flt": [], "src": [], "ref": []}
    for p in range(fmt.num_planes):
        pw = w >> (fmt.subsampling_w if p else 0)
        ph = h >> (fmt.subsampling_h if p else 0)
        if fmt.sample_type == vs.FLOAT:
            lo, hi = (-0.5, 0.5) if (p and fmt.color_family == vs.YUV) else (0.0, 1.0)
            ref = rng.uniform(lo + 0.2, hi - 0.2, (ph, pw)).astype(np.float32)
            flt = (ref + rng.uniform(-6, 6, (ph, pw)) * thr_span).astype(np.float32)
            src = rng.uniform(lo, hi, (ph, pw)).astype(np.float32)
        else:
            peak = (1 << fmt.bits_per_sample) - 1
            dt = np.uint8 if fmt.bits_per_sample == 8 else np.uint16
            ref = rng.integers(peak // 4, 3 * peak // 4 + 1, (ph, pw))
            flt = np.clip(ref + np.round(rng.uniform(-6, 6, (ph, pw)) * thr_span), 0, peak)
            src = rng.integers(0, peak + 1, (ph, pw))
            ref, flt, src = (a.astype(dt) for a in (ref, flt, src))
        out["flt"].append(flt)
        out["src"].append(src)
        out["ref"].append(ref)
    return out


THRESHOLDS = [
    pytest.param(dict(dark_thr=8, bright_thr=16, elast=3), id="dark8-bright16-e3"),
    pytest.param(dict(dark_thr=4, bright_thr=4, elast=2.5), id="equal4-e2.5"),
    pytest.param(dict(), id="defaults"),  # thr 1, elast 2: no integer inside (1, 2) for 8-bit
    pytest.param(dict(dark_thr=5, bright_thr=9, elast=1), id="elast1-no-window"),
    pytest.param(dict(dark_thr=5, bright_thr=9, elast=0), id="elast0-inverted"),
    pytest.param(dict(dark_thr=0.5, bright_thr=2.25, elast=65535), id="elast-max"),
    pytest.param(dict(dark_thr=0, bright_thr=0, elast=2), id="zero-thr"),
]

FORMATS = [vs.GRAY8, vs.GRAY16, vs.GRAYS, vs.YUV420P8, vs.YUV444P16, vs.YUV420PS, vs.RGB24, vs.RGBS]


@pytest.mark.parametrize("with_ref", [False, True], ids=["implicit-ref-src", "explicit-ref"])
@pytest.mark.parametrize("kwargs", THRESHOLDS)
@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: vs.PresetVideoFormat(f).name)
def test_float32_oracle_all_planes(core, fmt, kwargs, with_ref):
    """Bit-exact against the f32 oracle on every plane, at a width that
    leaves a scalar tail after every vector width (8/16/32/64 lanes)."""
    np = _np()
    vf = core.get_video_format(fmt)
    w, h = 2 * 67, 6  # 134: not a multiple of 4/8/16/32/64; chroma 67
    full = vf.color_family == vs.RGB or vf.sample_type == vs.FLOAT
    props = {"_Range": 1} if full else {}
    dark8 = kwargs.get("dark_thr", 1.0)
    bright8 = kwargs.get("bright_thr", 1.0)
    elast = kwargs.get("elast", 2.0)
    dark = _scaled_thr(np, dark8, vf, full)
    bright = _scaled_thr(np, bright8, vf, full)
    span = float(max(dark, bright, _scaled_thr(np, 1, vf, full)))

    rng = np.random.default_rng(1234 + fmt % 1000)
    t = _random_triplet(np, rng, vf, w, h, span)
    flt = _clip_from_planes(core, vf, t["flt"], props)
    src = _clip_from_planes(core, vf, t["src"], props)
    ref = _clip_from_planes(core, vf, t["ref"], props)
    out = flt.vszip.LimitFilter(src, ref, **kwargs) if with_ref else flt.vszip.LimitFilter(src, **kwargs)
    got = _planes_of(np, out)

    is_int = vf.sample_type == vs.INTEGER
    for p in range(vf.num_planes):
        r = t["ref"][p] if with_ref else t["src"][p]
        want = _oracle_plane(np, t["flt"][p], t["src"][p], r, dark, bright, elast, is_int)
        if not is_int:
            # identical bits, NaN-free by construction
            assert np.array_equal(got[p].view(np.uint32), want.view(np.uint32)), f"plane {p}"
        else:
            bad = np.argwhere(got[p] != want)
            assert bad.size == 0, f"plane {p}: {len(bad)} mismatches, first at {bad[0].tolist()}: got {got[p][tuple(bad[0])]} want {want[tuple(bad[0])]}"


@pytest.mark.parametrize("sign", [1, -1], ids=["brighter", "darker"])
def test_exact_window_boundaries(core, tmp_path, sign):
    """Hand-checked region boundaries, independent of any oracle code:
    |d| == thr1 keeps flt, |d| == thr2 takes src, strictly inside interpolates.
    Brighter pixels use bright_thr, darker ones dark_thr."""
    dark, bright, elast = 4, 10, 3
    thr1 = bright if sign > 0 else dark
    thr2 = thr1 * elast
    ref_v, src_v = 120, 60
    dists = [thr1 - 1, thr1, thr1 + 1, (thr1 + thr2) // 2, thr2 - 1, thr2, thr2 + 1]
    flt_vals = [ref_v + sign * d for d in dists]
    width = len(dists)
    flt = _gray8(core, tmp_path, "bflt", flt_vals * 2, width)
    src = _gray8(core, tmp_path, "bsrc", [src_v] * width * 2, width)
    ref = _gray8(core, tmp_path, "bref", [ref_v] * width * 2, width)
    out = flt.vszip.LimitFilter(src, ref, dark_thr=dark, bright_thr=bright, elast=elast)
    with out.get_frame(0) as f:
        row = [f[0][0, x] for x in range(width)]

    assert row[0] == flt_vals[0]  # inside thr1
    assert row[1] == flt_vals[1]  # == thr1 still keeps flt
    assert row[5] == src_v  # == thr2 falls back to src
    assert row[6] == src_v  # beyond thr2
    for x in (2, 3, 4):  # strictly inside: between src and flt, never equal to either end
        lo, hi = sorted((src_v, flt_vals[x]))
        assert lo < row[x] < hi, f"x={x}: {row[x]} not strictly between {lo} and {hi}"
    # interpolation weight falls monotonically from flt towards src
    assert abs(row[2] - flt_vals[2]) <= abs(row[3] - flt_vals[3]) <= abs(row[4] - flt_vals[4])


def test_stride_independent(core, make_clip):
    """Cropped (wide parent stride) and repacked inputs give identical output."""
    from helpers import repack

    base = make_clip(vs.YUV420P16)
    src = base.std.Crop(left=6, right=10, top=2, bottom=4)
    flt = src.std.BoxBlur(hradius=2, vradius=2)
    wide = flt.vszip.LimitFilter(src, dark_thr=3, bright_thr=5, elast=2.5)
    tight = repack(flt).vszip.LimitFilter(repack(src), dark_thr=3, bright_thr=5, elast=2.5)
    assert_same_clip(wide, tight)


@pytest.mark.xfail(
    strict=True,
    reason="known bug: hz.scaleValue takes the range of a float clip from _Range "
    "(missing -> LIMITED), so float thresholds scale by 1/219 instead of 1/255; "
    "float clips are always full range",
)
@pytest.mark.parametrize("props", [{}, {"_Range": 0}], ids=["no-range-prop", "range-limited-prop"])
def test_float_threshold_scale_ignores_range_prop(core, props):
    # flt - src = 0.042 lies between 10/255 (0.0392) and 10/219 (0.0457)
    src = core.std.BlankClip(None, 8, 2, vs.GRAYS, length=1, color=0.5)
    flt = core.std.BlankClip(src, color=0.542)
    if props:
        src = src.std.SetFrameProps(**props)
        flt = flt.std.SetFrameProps(**props)
    full = core.std.BlankClip(src, color=0.5).std.SetFrameProps(_Range=1)
    flt_full = core.std.BlankClip(src, color=0.542).std.SetFrameProps(_Range=1)
    got = flt.vszip.LimitFilter(src, dark_thr=10, bright_thr=10, elast=4)
    want = flt_full.vszip.LimitFilter(full, dark_thr=10, bright_thr=10, elast=4)
    with got.get_frame(0) as a, want.get_frame(0) as b:
        assert a[0][0, 0] == b[0][0, 0]
