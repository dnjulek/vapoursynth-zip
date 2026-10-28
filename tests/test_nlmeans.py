"""NLMeans (KNLMeansCL-compatible non-local means) tests.

Contract (src/vapoursynth/nlmeans.zig, src/filters/nlmeans.zig):
- 8-16 bit integer, 16-bit half and 32-bit float; output keeps the input
  format, size, length and the center frame's props.
- channels: Gray -> "Y"; YUV -> "Y" (default, chroma copied), "UV" (luma
  copied, joint U+V distance) or "YUV" (4:4:4 only); RGB -> "RGB".
- Values are normalized to [0, 1] (integers divided by their own peak, like
  KNLMeansCL), the frame border is zero, frame n uses the temporal radius
  m = min(d, n) with neighbours clamped to the last frame.
- wmode 1..3 must be bit-exact with the vszipcu CUDA port (same tap order,
  rounding points and fused multiply-adds) for every `channels`; wmode 0
  computes exp as a degree-4 exp2 polynomial and stays within ~1e-5 of an
  exact exp.
- Finite input gives finite output: the RGB joint distance is clamped at 0
  (it goes negative only for extended-range input, e.g. PQ in linear light,
  where the GPU itself overflows), and an h whose wmode-0 exp2 constant
  would overflow is rejected.

Nets:
  1. an independent float64 oracle written from the algorithm, not from the
     tiled/SIMD kernel (whole-image shifts, an integral-image box sum, exact
     exp), checked across formats, parameters, temporal edges and
     SIMD/tile/batch boundary geometries;
  2. metamorphic checks (crop vs repack, plane passthrough, rclip=clip,
     f16 vs f32, 10-bit vs float, scheduling order);
  3. optional parity with vszipcu (the CUDA port) when it is installed;
  4. golden snapshots.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import vapoursynth as vs

from conftest import _geometry
from golden import Case, grid, sweep
from helpers import assert_long_clip_edges, assert_same_clip, props, repack

HAS_CU = hasattr(vs.core, "vszipcu")

CASES = sweep(
    base_fmt=vs.GRAYS,
    base_args={},
    formats=[vs.GRAY8, vs.GRAY16, vs.GRAYH, vs.GRAYS, vs.YUV420P8, vs.YUV420P16, vs.RGB24, vs.RGBS],
    args=grid(d=[0, 2]) + grid(a=[1, 4]) + grid(s=[0, 2]) + grid(wmode=[1, 2, 3]) + [dict(h=3.0), dict(wref=0.3)],
    geometries=["odd", "tiny"],
) + [
    Case(vs.YUV420P16, args=dict(channels="UV")),
    Case(vs.YUV444PS, args=dict(channels="YUV")),
    Case(vs.YUV420P16, "tiny", dict(channels="UV", a=1)),
]


@pytest.fixture(scope="module")
def temporal(make_temporal_clip, temporal_rgb):
    """make_temporal_clip, plus RGB formats (which it can't convert to)."""

    def make(fmt, geometry="full"):
        if vs.core.get_video_format(fmt).color_family == vs.RGB:
            clip = temporal_rgb if fmt == vs.RGB24 else temporal_rgb.resize.Point(format=fmt)
            return _geometry(clip, geometry)
        return make_temporal_clip(fmt, geometry)

    return make


@pytest.mark.parametrize("case", CASES, ids=str)
def test_golden_cases(golden, temporal, case):
    # tiny is 13x7 luma (fits a=2); the tiny UV case uses a=1 for 6x3 chroma
    src = temporal(case.fmt, case.geometry)
    # last frame: d=2 reaches radius 2 there (the future side clamped), so
    # d=0/1/2 lock different windows
    golden.check("nlmeans", case, src.vszip.NLMeans(**case.args), n=2)


# --- independent float64 oracle ------------------------------------------------


def _norm_planes(clip, n, planes):
    """(C, H, W) float64 of the processed planes, normalized to [0, 1]."""
    with clip.get_frame(n) as f:
        arr = np.stack([np.asarray(f[p]).astype(np.float64) for p in planes])
    if clip.format.sample_type == vs.INTEGER:
        arr /= (1 << clip.format.bits_per_sample) - 1
    return arr


def _pix_dist(kind, A, B):
    if kind == "Y":
        return 3.0 * (A[0] - B[0]) ** 2
    if kind == "UV":
        return 1.5 * ((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)
    if kind == "YUV":
        return ((A - B) ** 2).sum(axis=0)
    m_red = (A[0] + B[0]) / 6.0
    return (2.0 / 3.0 + m_red) * (A[0] - B[0]) ** 2 + (4.0 / 3.0) * (A[1] - B[1]) ** 2 + (1.0 - m_red) * (A[2] - B[2]) ** 2


def _box(D, s):
    """Sum over the (2s+1)^2 patch around every pixel; zero outside D."""
    H, W = D.shape
    P = np.zeros((H + 2 * s + 1, W + 2 * s + 1))
    P[s + 1 : s + 1 + H, s + 1 : s + 1 + W] = D
    I = P.cumsum(0).cumsum(1)
    k = 2 * s + 1
    return I[k:, k:] - I[:-k, k:] - I[k:, :-k] + I[:-k, :-k]


def nlm_oracle(frames, n, *, d=1, a=2, s=4, h=1.2, wmode=0, wref=1.0, kind="Y", ref_frames=None):
    """Spec-level NLMeans of frame n. `frames`/`ref_frames`: per frame a
    (C, H, W) normalized array of the processed planes."""
    nf = len(frames)
    m = min(d, n)
    pick = lambda fr, k: fr[min(max(n + k, 0), nf - 1)]
    C, H, W = frames[0].shape
    P = a + s
    pad = lambda x: np.pad(x, ((0, 0), (P, P), (P, P)))
    val = {k: pad(pick(frames, k)) for k in range(-m, m + 1)}
    wsrc = {k: pad(pick(ref_frames or frames, k)) for k in range(-m, m + 1)}
    at = lambda X, qy, qx: X[:, P + qy : P + qy + H, P + qx : P + qx + W]
    kk = 65025.0 / (3.0 * h * h * (2 * s + 1) ** 2)

    def weight_image(t1, t2, qy, qx):
        # first operand in frame (D is frame-sized, zero outside), second at +q
        S = _box(_pix_dist(kind, at(wsrc[t1], 0, 0), at(wsrc[t2], qy, qx)), s)
        arg = kk * S
        if wmode == 0:
            return np.exp(-arg)
        c = np.maximum(1.0 - arg, 0.0)
        return c ** {1: 1, 2: 2, 3: 8}[wmode]

    def shifted_weight(Wimg, qy, qx):
        # W(x - q), zero where x - q is outside the frame
        Wp = np.pad(Wimg, a)
        return Wp[a - qy : a - qy + H, a - qx : a - qx + W]

    acc = np.zeros((C, H, W))
    accw = np.zeros((H, W))
    u5 = np.full((H, W), 2.0**-23)
    side = 2 * a + 1
    for qz in range(-m, 1):
        for qy in range(-a, a + 1):
            for qx in range(-a, a + 1):
                if qz * side * side + qy * side + qx >= 0:
                    continue
                wc = weight_image(0, qz, qy, qx)
                wm = shifted_weight(weight_image(-qz, 0, qy, qx), qy, qx)
                acc += wc * at(val[qz], qy, qx) + wm * at(val[-qz], -qy, -qx)
                accw += wc + wm
                u5 = np.maximum(u5, np.maximum(wc, wm))
    mm = wref * u5
    den = mm + accw
    with np.errstate(invalid="ignore", divide="ignore"):
        return (at(val[0], 0, 0) * mm + acc) / den, den


def _kind_and_planes(clip, channels):
    fam = clip.format.color_family
    if fam == vs.RGB:
        return "RGB", [0, 1, 2]
    if channels == "UV":
        return "UV", [1, 2]
    if channels == "YUV":
        return "YUV", [0, 1, 2]
    return "Y", [0]


def check_against_oracle(src, n, rclip=None, atol_float=5e-6, **kw):
    kind, planes = _kind_and_planes(src, kw.get("channels"))
    frames = [_norm_planes(src, i, planes) for i in range(src.num_frames)]
    ref_frames = [_norm_planes(rclip, i, planes) for i in range(rclip.num_frames)] if rclip is not None else None
    args = {k: v for k, v in kw.items() if k != "channels"}
    expect, den = nlm_oracle(frames, n, kind=kind, ref_frames=ref_frames, **args)
    # With wref=0 a pixel without any similar patch has a weight total below
    # f32 range (weights < 1e-38 are 0 in f32): the result there is
    # ill-conditioned and covered by test_zero_wref_without_matches_keeps_source.
    valid = np.broadcast_to(den >= 1e-30, expect.shape)
    assert valid.mean() > 0.5
    out = src.vszip.NLMeans(**kw, **({"rclip": rclip} if rclip is not None else {}))
    with out.get_frame(n) as f:
        got = np.stack([np.asarray(f[p]).astype(np.float64) for p in planes])
    fmt = src.format
    if fmt.sample_type == vs.INTEGER:
        peak = (1 << fmt.bits_per_sample) - 1
        want = np.clip(np.rint(expect * peak), 0, peak)
        bad = np.where(valid, np.abs(got - want), 0)
        assert bad.max() <= 1, f"max |diff| {bad.max()} LSB at {np.unravel_index(bad.argmax(), bad.shape)}"
        # off-by-one only where the exact value sits within the float error
        # of a rounding edge
        edge = np.abs(expect * peak - np.floor(expect * peak) - 0.5)
        assert (edge[bad > 0] < atol_float * peak + 1e-3).all(), "a 1-LSB difference away from a rounding edge"
    else:
        tol = 2e-3 if fmt.bits_per_sample == 16 else atol_float
        bad = np.where(valid, np.abs(got - expect), 0)
        assert bad.max() <= tol, f"max |diff| {bad.max():.3g} at {np.unravel_index(bad.argmax(), bad.shape)}"


@pytest.fixture(scope="module")
def small_gray(make_temporal_clip):
    """3 moving frames of a 41x27 GRAYS region with real detail."""
    return make_temporal_clip(vs.GRAYS).std.CropAbs(width=41, height=27, left=200, top=100)


ORACLE_ARGS = [
    {},
    dict(d=0),
    dict(d=2),
    dict(a=1, s=0),
    dict(a=3, s=1),
    dict(s=2, h=0.6),
    dict(h=4.0),
    dict(wmode=1),
    dict(wmode=2, h=2.0),
    dict(wmode=3, h=3.0),
    dict(wref=0.25),
    dict(wref=0.0, h=2.0),
]


@pytest.mark.parametrize("n", [0, 1, 2])
@pytest.mark.parametrize("kw", ORACLE_ARGS, ids=lambda k: ",".join(f"{a}={b}" for a, b in k.items()) or "default")
def test_oracle_gray_float(small_gray, n, kw):
    # n=0: radius clipped to 0; n=2 with d=2: both future neighbours clamp to
    # the last frame (KNLMeansCL's window)
    check_against_oracle(small_gray, n, **kw)


@pytest.mark.parametrize(
    ("fmt", "kw"),
    [
        (vs.GRAY8, {}),
        (vs.GRAY16, {}),
        (vs.GRAY10, {}),
        (vs.GRAYH, {}),
        (vs.YUV420P16, {}),
        (vs.YUV420P16, dict(channels="UV")),
        (vs.YUV444P8, dict(channels="YUV")),
        (vs.YUV444PS, dict(channels="YUV")),
        (vs.RGB24, {}),
        (vs.RGBS, {}),
        (vs.RGB48, dict(d=0, wmode=2)),
    ],
    ids=lambda v: vs.core.get_video_format(v).name if isinstance(v, int) else (",".join(f"{a}={b}" for a, b in v.items()) or "default"),
)
def test_oracle_formats(temporal, fmt, kw):
    src = temporal(fmt).std.CropAbs(width=36, height=22, left=200, top=100)
    check_against_oracle(src, 1, **kw)


@pytest.mark.parametrize(
    ("w", "h"),
    # 8- and 16-lane vector tails, a 512-wide tile boundary plus tail, the
    # 64-row tile boundary and 4-row batch remainders (h mod 4 = 1, 2, 3)
    [(5, 9), (7, 5), (8, 6), (9, 7), (15, 9), (16, 9), (17, 9), (31, 9), (32, 9), (33, 9), (63, 9), (64, 9), (65, 9), (530, 13), (40, 70)],
)
def test_oracle_geometry_boundaries(make_temporal_clip, w, h):
    src = make_temporal_clip(vs.GRAYS).std.CropAbs(width=w, height=h, left=20, top=60)
    check_against_oracle(src, 1, d=1, a=2, s=2)


def test_oracle_rclip(small_gray):
    rclip = small_gray.std.BoxBlur(hradius=1, vradius=1)
    check_against_oracle(small_gray, 1, rclip=rclip)


# --- metamorphic and contract checks -------------------------------------------


def test_rclip_equal_to_clip_is_identity_of_rclip(small_gray):
    assert_same_clip(small_gray.vszip.NLMeans(rclip=small_gray), small_gray.vszip.NLMeans())


def test_stride_and_crop(make_temporal_clip):
    cropped = make_temporal_clip(vs.GRAY16).std.Crop(left=27, top=5)
    assert_same_clip(cropped.vszip.NLMeans(), repack(cropped).vszip.NLMeans())


@pytest.mark.parametrize(
    ("fmt", "channels", "processed"),
    [
        (vs.YUV420P16, None, [0]),
        (vs.YUV420P16, "Y", [0]),
        (vs.YUV420P16, "UV", [1, 2]),
        (vs.YUV444P16, "YUV", [0, 1, 2]),
        (vs.RGB48, None, [0, 1, 2]),
    ],
)
def test_plane_routing(temporal, fmt, channels, processed):
    src = temporal(fmt)
    out = src.vszip.NLMeans(**({"channels": channels} if channels else {}))
    assert (out.format.id, out.width, out.height, out.num_frames) == (src.format.id, src.width, src.height, src.num_frames)
    for n in range(src.num_frames):
        with src.get_frame(n) as fs, out.get_frame(n) as fo:
            changed = [not np.array_equal(np.asarray(fs[p]), np.asarray(fo[p])) for p in range(src.format.num_planes)]
        # unprocessed planes are copied; a processed plane may legitimately
        # stay put (the test image's blue is nearly constant), not all of them
        assert not any(changed[p] for p in range(src.format.num_planes) if p not in processed), f"frame {n}: {changed}"
        assert any(changed[p] for p in processed), f"frame {n}: nothing processed"


def test_props_from_center_frame(make_temporal_clip):
    src = make_temporal_clip(vs.GRAY8)
    src = src.std.SetFrameProps(Marker=0)
    src = src[0].std.SetFrameProps(Marker=10) + src[1].std.SetFrameProps(Marker=11) + src[2].std.SetFrameProps(Marker=12)
    out = src.vszip.NLMeans(d=1)
    assert [props(out, n)["Marker"] for n in range(3)] == [10, 11, 12]


def test_f16_matches_f32(make_temporal_clip):
    h16 = make_temporal_clip(vs.GRAYH).vszip.NLMeans().resize.Point(format=vs.GRAYS)
    f32 = make_temporal_clip(vs.GRAYH).resize.Point(format=vs.GRAYS).vszip.NLMeans()
    for n in range(3):
        with h16.get_frame(n) as a, f32.get_frame(n) as b:
            # one f16 rounding of the output (values <= 1: <= 2^-11 relative)
            assert np.abs(np.asarray(a[0]) - np.asarray(b[0])).max() <= 2**-11


def test_10bit_is_normalized_by_its_peak(make_temporal_clip):
    """9-15 bit values are divided by their own peak (KNLMeansCL), so h means
    the same at every depth."""
    c10 = make_temporal_clip(vs.GRAY10)
    flt = c10.std.Expr("x 1023 /", format=vs.GRAYS).vszip.NLMeans()
    out = c10.vszip.NLMeans()
    for n in range(3):
        with out.get_frame(n) as a, flt.get_frame(n) as b:
            want = np.clip(np.rint(np.asarray(b[0]).astype(np.float64) * 1023), 0, 1023)
            assert np.abs(np.asarray(a[0]).astype(np.float64) - want).max() <= 1


def test_zero_wref_without_matches_keeps_source(core):
    """wref=0 and no similar patch: every weight underflows, den would be 0.
    The source pixel comes through instead of NaN (the GPU writes NaN/0)."""
    data = np.random.default_rng(3).random((24, 32), dtype=np.float32)
    noise = core.std.BlankClip(None, 32, 24, vs.GRAYS, length=1)

    def fill(n, f):
        fo = f.copy()
        np.asarray(fo[0])[:] = data
        return fo

    noise = noise.std.ModifyFrame(noise, fill)
    out = noise.vszip.NLMeans(d=0, wref=0.0, h=1e-3)
    with out.get_frame(0) as a, noise.get_frame(0) as b:
        got, src = np.asarray(a[0]), np.asarray(b[0])
        assert np.isfinite(got).all()
        assert np.array_equal(got, src)


def _array_clip(core, values, fmt):
    """One-frame clip holding `values` (planes, h, w)."""
    base = core.std.BlankClip(None, values.shape[2], values.shape[1], fmt, length=1)

    def fill(n, f):
        fo = f.copy()
        for p in range(values.shape[0]):
            np.asarray(fo[p])[:] = values[p]
        return fo

    return base.std.ModifyFrame(base, fill)


def _planes(clip):
    with clip.get_frame(0) as f:
        return np.stack([np.asarray(f[p]).copy() for p in range(clip.format.num_planes)])


@pytest.mark.parametrize("wmode", [0, 1, 2, 3])
@pytest.mark.parametrize("kw", [dict(d=0, a=1, s=0), {}], ids=["d0a1s0", "default"])
def test_extended_range_rgb_stays_finite(core, wmode, kw):
    """A red mean above 3 turns the RGB distance's blue coefficient negative.
    Weights must stay <= 1 so finite input never overflows to inf/NaN."""
    hdr = np.zeros((3, 9, 17), np.float32)
    hdr[0], hdr[1], hdr[2, :, 1::2] = 4.0, 0.1, 0.5
    got = _planes(_array_clip(core, hdr, vs.RGBS).vszip.NLMeans(wmode=wmode, **kw))
    assert np.isfinite(got).all()
    # every weight <= 1 and the source is constant along rows: a convex mix
    assert got.min() >= hdr.min() - 1e-6 and got.max() <= hdr.max() + 1e-6


def test_pq_to_linear_rgb_stays_finite(core):
    """Legal 10-bit PQ YUV converted to linear light reaches red ~4."""
    patches = [core.std.BlankClip(None, 8, 32, vs.YUV444P10, color=list(c), length=1)
               for c in [(383, 308, 705), (410, 534, 687)]]
    rgb = core.std.StackHorizontal(patches * 4).resize.Point(
        format=vs.RGBS, matrix_in_s="2020ncl", range_in_s="full", range_s="full",
        transfer_in_s="st2084", transfer_s="linear",
        primaries_in_s="2020", primaries_s="2020", nominal_luminance=100,
    )
    assert _planes(rgb)[0].max() > 3.5
    assert np.isfinite(_planes(rgb.vszip.NLMeans())).all()


@pytest.mark.parametrize("wmode", [0, 1, 3])
def test_tiny_h_keeps_perfect_matches(core, wmode):
    """With a constant rclip every patch distance is exactly 0, so every
    weight is 1 whatever h is: the output must not depend on h, down to the
    smallest h accepted (wmode 0 folds log2(e) into k, which overflows first)."""
    noise = _array_clip(core, np.random.default_rng(2).random((1, 9, 17), dtype=np.float32), vs.GRAYS)
    ref = core.std.BlankClip(noise, color=[0])
    want = _planes(noise.vszip.NLMeans(d=0, a=1, s=4, h=1.0, wmode=wmode, rclip=ref))
    assert np.abs(want - _planes(noise)).max() > 0.05  # it does smooth
    got = _planes(noise.vszip.NLMeans(d=0, a=1, s=4, h=1.2e-18, wmode=wmode, rclip=ref))
    assert np.array_equal(got, want)


def test_h_too_small_for_exp2_constant(core):
    src = core.std.BlankClip(None, 64, 48, vs.GRAYS, length=1)
    with pytest.raises(vs.Error, match="h must be > 0"):
        src.vszip.NLMeans(s=4, h=1e-18)
    # k itself is finite there: the linear weight modes accept it
    src.vszip.NLMeans(s=4, h=1e-18, wmode=1).get_frame(0).close()


def test_scheduling_order_does_not_matter(make_temporal_clip):
    src = make_temporal_clip(vs.GRAY16)
    src = src + src.std.FlipHorizontal() + src.std.FlipVertical()  # 9 frames
    seq = src.vszip.NLMeans(d=2)
    expected = []
    for n in range(src.num_frames):
        with seq.get_frame(n) as f:
            expected.append(np.asarray(f[0]).copy())

    shuffled = src.vszip.NLMeans(d=2)
    for n in [8, 0, 4, 4, 7, 1, 2, 6, 3, 5]:
        with shuffled.get_frame(n) as f:
            assert np.array_equal(np.asarray(f[0]), expected[n]), f"shuffled frame {n}"

    concurrent = src.vszip.NLMeans(d=2)

    def get(n):
        with concurrent.get_frame(n) as f:
            return n, np.asarray(f[0]).copy()

    with ThreadPoolExecutor(max_workers=8) as ex:
        for n, arr in ex.map(get, list(range(src.num_frames)) * 3):
            assert np.array_equal(arr, expected[n]), f"concurrent frame {n}"


@pytest.mark.parametrize(
    ("fmt", "args", "msg"),
    [
        (vs.GRAYS, dict(d=-1), "d must be 0..16"),
        (vs.GRAYS, dict(d=17), "d must be 0..16"),
        (vs.GRAYS, dict(a=0), "a must be 1..64"),
        (vs.GRAYS, dict(a=65), "a must be 1..64"),
        (vs.GRAYS, dict(s=-1), "s must be 0..8"),
        (vs.GRAYS, dict(s=9), "s must be 0..8"),
        (vs.GRAYS, dict(h=0.0), "h must be > 0"),
        (vs.GRAYS, dict(h=-1.0), "h value -1 is below minimum 0"),
        (vs.GRAYS, dict(wmode=4), "wmode must be 0..3"),
        (vs.GRAYS, dict(wref=-0.5), "wref value -0.5 is below minimum 0"),
        (vs.GRAYS, dict(channels="UV"), "'channels' must be 'Y' with Gray"),
        (vs.YUV420P8, dict(channels="YUV"), "requires 4:4:4"),
        (vs.YUV420P8, dict(channels="RGB"), "'channels' must be 'YUV', 'Y' or 'UV' with YUV"),
        (vs.RGB24, dict(channels="Y"), "'channels' must be 'RGB' with RGB"),
        (vs.GRAY32, {}, "not supported Int format"),
    ],
)
def test_validation_errors(core, fmt, args, msg):
    src = core.std.BlankClip(None, 64, 48, fmt, length=1)
    with pytest.raises(vs.Error, match=msg):
        src.vszip.NLMeans(**args)


def test_window_larger_than_plane(core):
    with pytest.raises(vs.Error, match=r"research window \(2\*a\+1\) larger"):
        core.std.BlankClip(None, 64, 8, vs.GRAY8, length=1).vszip.NLMeans(a=4)
    # UV works on the 4:2:0 chroma planes (32x4 here)
    with pytest.raises(vs.Error, match="research window"):
        core.std.BlankClip(None, 64, 8, vs.YUV420P8, length=1).vszip.NLMeans(channels="UV", a=2)


def test_rclip_mismatch(core):
    src = core.std.BlankClip(None, 64, 48, vs.GRAY8, length=3)
    for bad in (
        core.std.BlankClip(None, 64, 48, vs.GRAY16, length=3),
        core.std.BlankClip(None, 62, 48, vs.GRAY8, length=3),
        core.std.BlankClip(None, 64, 48, vs.GRAY8, length=2),
    ):
        with pytest.raises(vs.Error, match="'rclip' must match"):
            src.vszip.NLMeans(rclip=bad)


# --- optional: parity with the CUDA port --------------------------------------


@pytest.mark.skipif(not HAS_CU, reason="vszipcu (CUDA port) not installed")
@pytest.mark.parametrize("wmode", [1, 2, 3])
@pytest.mark.parametrize("fmt", [vs.RGBS, vs.YUV444PS])
def test_joint_channels_bit_exact_with_cuda_port(core, fmt, wmode):
    """Dense joint-channel distances: these depend on which multiply-adds
    nvcc fuses, so every term order matters (YUV chroma centered on 0)."""
    vals = np.random.default_rng(6).random((3, 23, 37), dtype=np.float32)
    channels = "RGB"
    if fmt == vs.YUV444PS:
        vals[1:] -= 0.5
        channels = "YUV"
    src = _array_clip(core, vals, fmt)
    for h in (0.6, 50.0):
        kw = dict(d=0, a=2, s=1, h=h, wmode=wmode, channels=channels)
        cpu = _planes(src.vszip.NLMeans(**kw))
        gpu = _planes(src.vszipcu.NLMeans(**kw))
        assert np.array_equal(cpu, gpu), f"h={h}: {np.count_nonzero(cpu != gpu)} samples differ"


@pytest.mark.skipif(not HAS_CU, reason="vszipcu (CUDA port) not installed")
@pytest.mark.parametrize(
    ("fmt", "kw", "tol"),
    [
        # same tap order and rounding points: bit-exact without exp
        (vs.GRAYS, dict(wmode=1), 0),
        (vs.GRAYS, dict(wmode=3, h=3.0, d=2), 0),
        (vs.YUV444PS, dict(channels="YUV", wmode=2), 0),
        (vs.RGBS, dict(wmode=1), 0),
        (vs.RGBS, dict(wmode=3, h=3.0, d=2), 0),
        # exp differs (degree-4 exp2 here, CUDA expf there)
        (vs.GRAYS, {}, 1e-5),
        (vs.RGBS, {}, 1e-5),
        (vs.GRAY16, {}, 1),
        (vs.YUV420P8, dict(channels="UV"), 1),
    ],
)
def test_matches_cuda_port(temporal, fmt, kw, tol):
    src = temporal(fmt)
    a = src.vszip.NLMeans(**kw)
    b = src.vszipcu.NLMeans(**kw)
    for n in range(src.num_frames):
        with a.get_frame(n) as fa, b.get_frame(n) as fb:
            for p in range(src.format.num_planes):
                d = np.abs(np.asarray(fa[p]).astype(np.float64) - np.asarray(fb[p]).astype(np.float64)).max()
                assert d <= tol, f"frame {n} plane {p}: max |diff| {d}"


@pytest.mark.parametrize("d", [1, 3])
def test_last_frames_of_the_longest_clip(d):
    """KNLMeansCL's window near frame 2^31 - 1 is formed without i32
    overflow."""
    assert_long_clip_edges(f"{{c}}.vszip.NLMeans(d={d}, a=1)")
