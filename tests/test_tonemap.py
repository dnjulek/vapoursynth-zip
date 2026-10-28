"""Contract and oracles for Tonemap.

Tonemap ports vs-placebo's Tonemap: libplacebo's renderer path for one frame
(YCbCr BT.2020 NCL decode with lanczos chroma upsampling, PQ/HLG linearize,
pl_shader_color_map_ex in IPT with a 1D tone LUT and a 3D gamut LUT,
delinearize, encode), following libplacebo v7.387. It is not 1:1
with the GPU: PQ transfers are LUT-interpolated, pow/exp/log approximated,
the gamut LUT stays f32 and 16-bit output is rounded instead of dithered.

Observable contract (src/vapoursynth/tonemap.zig, src/filters/tonemap.zig):

- input: 16-bit integer or 32-bit float, RGB or YUV with 2x or no
  subsampling per axis; at least 64x64. Output: RGB48 / YUV444P16 (float:
  RGBS / YUV444PS), same size and length.
- YUV is always decoded as BT.2020 NCL; RGB out is full range, YUV out keeps
  the source's range (_Range, else _ColorRange, else limited); float is full.
  Integer output is clipped to the range's codes (limited: Y 16-235, C
  16-240, << 8).
- props: _Matrix (0 RGB, 1 SDR, 9 HDR), _Transfer (1 / 16 / 18), _Primaries,
  _Range; _ColorRange and _ChromaLocation are removed.
- src_csp 1 (HDR10) / 2 (HLG), dst_csp 0 (SDR) / 1 (HDR10) / 2 (HLG);
  Dolby Vision (3, use_dovi) is rejected.
- peak detection: frame n is computed from frames n - window + 1 .. n
  (window = 5 smoothing periods, smoothing_period <= 60), whatever order
  frames are requested in; no window when the settings rule detection out.

Oracles: an independent numpy model of the clip/clip pipeline (no tone curve,
no gamut LUT), pointwise metamorphic properties (cropping commutes with the
filter for RGB/4:4:4), request-order determinism of peak detection, goldens,
and, when vs-placebo is installed, parity with it.
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import vapoursynth as vs

from golden import Case, sweep
from helpers import assert_same_clip, max_abs_diff_all_planes, props, repack

META = dict(MasteringDisplayMaxLuminance=1000.0, ContentLightLevelMax=1200.0, ContentLightLevelAverage=300.0)


def _hdr(src_rgb, fmt, trc="st2084", gain=12.0, meta=True):
    """The test image as HDR: linear BT.2020 light expanded to ~1200 nits
    (x^2 * 12, 1.0 = 100 nits), encoded with `trc`. The image's blue is
    nearly constant (252..255), so blue takes the mirrored green instead."""
    lin = src_rgb.resize.Bicubic(format=vs.RGBS, transfer_in_s="709", transfer_s="linear",
                                 primaries_in_s="709", primaries_s="2020")
    lin = vs.core.std.ShufflePlanes([lin, lin.std.FlipHorizontal()], planes=[0, 1, 1], colorfamily=vs.RGB)
    scale = gain if trc == "st2084" else gain / 10.0  # HLG: scene light 0..1.2
    lin = lin.std.Expr(f"x 0 max 2 pow {scale} *")
    f = vs.core.get_video_format(fmt)
    kw = dict(format=fmt, transfer_in_s="linear", transfer_s=trc, primaries_in_s="2020", primaries_s="2020",
              nominal_luminance=100)
    if f.color_family == vs.YUV:
        kw["matrix_s"] = "2020ncl"
    out = lin.resize.Bicubic(**kw)
    return out.std.SetFrameProps(**META) if meta else out


@pytest.fixture(scope="module")
def make_hdr(src_rgb):
    cache = {}

    def make(fmt, geometry="full", trc="st2084"):
        key = (int(fmt), geometry, trc)
        if key not in cache:
            c = _hdr(src_rgb, fmt, trc)
            f = c.format
            if geometry == "odd":
                c = c.std.Crop(right=1 << f.subsampling_w, bottom=1 << f.subsampling_h)
            cache[key] = c
        return cache[key]

    return make


# --- goldens ---------------------------------------------------------------

CASES = (
    sweep(
        base_fmt=vs.YUV420P16,
        base_args={},
        formats=[vs.YUV420P16, vs.YUV422P16, vs.YUV444P16, vs.YUV440P16, vs.RGB48, vs.YUV420PS, vs.YUV444PS, vs.RGBS],
        args=[dict(dynamic_peak_detection=0)]
        + [dict(dynamic_peak_detection=0, tone_mapping_function=t) for t in range(12)]
        + [dict(dynamic_peak_detection=0, gamut_mapping=g) for g in range(10)]
        + [
            dict(dst_csp=1),
            dict(dst_csp=2),
            dict(dst_csp=1, dst_max=400.0),
            dict(dst_prim=12),
            dict(dst_max=100.0, dst_min=0.1),
            dict(src_max=4000.0, src_min=0.05),
            dict(percentile=99.5),
            dict(smoothing_period=0.0, scene_threshold_low=0.0),
            dict(black_cutoff=0.0),
            dict(metadata=2),
            dict(contrast_recovery=0.3),
            dict(contrast_recovery=0.3, contrast_smoothness=2.0),
            dict(show_clipping=1),
            dict(tone_mapping_function_s="bt2390", tone_mapping_param=1.5),
            dict(tone_mapping_function=5, tone_mapping_param=0.0),
            # libplacebo parameters vs-placebo doesn't expose
            dict(dynamic_peak_detection=0, film_strength=0.0),
            dict(dynamic_peak_detection=0, film_strength=1.5),
            dict(mastering_clip=0.0),
            dict(dynamic_peak_detection=0, spline_contrast=1.0, knee_default=0.3, slope_tuning=3.0),
            dict(dynamic_peak_detection=0, gamut_mapping=2, softclip_knee=0.4, softclip_desat=0.6),
            dict(dynamic_peak_detection=0, perceptual_deadzone=0.1, perceptual_strength=0.5),
            dict(dynamic_peak_detection=0, lut3d_size=[17, 17, 65], lut_size=64),
            dict(dynamic_peak_detection=0, lut3d_tricubic=1),
            dict(dst_csp=1, gamut_expansion=1),
            dict(dst_csp=1, dst_max=4000.0, inverse_tone_mapping=1, dynamic_peak_detection=0),
        ],
        geometries=["odd"],
    )
    + [
        Case(vs.RGB48, "odd", dict(dynamic_peak_detection=0)),
        Case(vs.YUV444PS, "odd", dict(gamut_mapping=2)),
        Case(vs.YUV420P16, args=dict(src_csp=2, dst_csp=0), variant="hlg"),
        Case(vs.YUV420P16, args=dict(src_csp=2, dst_csp=1), variant="hlg"),
        Case(vs.RGBS, args=dict(src_csp=2, dst_csp=0, dynamic_peak_detection=0), variant="hlg"),
    ]
)


@pytest.mark.parametrize("case", CASES, ids=str)
def test_golden_cases(golden, make_hdr, case):
    src = make_hdr(case.fmt, case.geometry, "std-b67" if case.variant == "hlg" else "st2084")
    args = {"src_csp": 1, **case.args}
    out = src.vszip.Tonemap(**args)
    golden.check("tonemap", case, out)


# --- format, size, props -----------------------------------------------------


@pytest.mark.parametrize(
    ("fmt", "out_fmt"),
    [
        (vs.YUV420P16, vs.YUV444P16),
        (vs.YUV422P16, vs.YUV444P16),
        (vs.YUV440P16, vs.YUV444P16),
        (vs.YUV444P16, vs.YUV444P16),
        (vs.RGB48, vs.RGB48),
        (vs.YUV420PS, vs.YUV444PS),
        (vs.RGBS, vs.RGBS),
    ],
)
def test_output_format(make_hdr, fmt, out_fmt):
    src = make_hdr(fmt)
    out = src.vszip.Tonemap()
    assert out.format.id == out_fmt
    assert (out.width, out.height, out.num_frames) == (src.width, src.height, src.num_frames)
    with out.get_frame(0) as f:
        for p in range(3):
            a = np.asarray(f[p])
            assert np.isfinite(a).all()
            # a tone-mapped picture, not a constant plane
            assert a.max() > a.min()


@pytest.mark.parametrize(
    ("fmt", "kw", "want"),
    [
        (vs.YUV420P16, {}, dict(_Matrix=1, _Transfer=1, _Primaries=1, _Range=0)),
        (vs.YUV420P16, dict(dst_csp=1), dict(_Matrix=9, _Transfer=16, _Primaries=9, _Range=0)),
        (vs.YUV420P16, dict(dst_csp=2), dict(_Matrix=9, _Transfer=18, _Primaries=9, _Range=0)),
        (vs.YUV420P16, dict(dst_prim=12), dict(_Matrix=1, _Transfer=1, _Primaries=12)),
        (vs.RGB48, {}, dict(_Matrix=0, _Transfer=1, _Primaries=1, _Range=1)),
        (vs.YUV444PS, {}, dict(_Matrix=1, _Range=1)),
    ],
)
def test_output_props(make_hdr, fmt, kw, want):
    src = make_hdr(fmt).std.SetFrameProps(_Range=0, _ChromaLocation=0)
    p = props(src.vszip.Tonemap(**kw))
    for k, v in want.items():
        assert p[k] == v, (k, p[k])
    # (VapourSynth R77+ aliases _ColorRange to _Range, which props() hides)
    assert "_ChromaLocation" not in p
    # HDR metadata props pass through untouched
    assert p["MasteringDisplayMaxLuminance"] == META["MasteringDisplayMaxLuminance"]


@pytest.mark.parametrize("value", [1e300, -1e300, float("inf")], ids=str)
def test_out_of_range_metadata_reads_as_absent(make_hdr, value):
    """Props that don't fit f32 (non-finite, or finite beyond its range)
    count as absent instead of overflowing to inf."""
    src = make_hdr(vs.YUV420P16)
    plain = src.std.RemoveFrameProps(["MasteringDisplayMaxLuminance", "ContentLightLevelMax"])
    bad = src.std.SetFrameProps(MasteringDisplayMaxLuminance=value, ContentLightLevelMax=value)
    for kw in (dict(dynamic_peak_detection=0), {}):
        assert_same_clip(plain.vszip.Tonemap(**kw), bad.vszip.Tonemap(**kw))


def test_full_range_yuv_stays_full(make_hdr):
    src = make_hdr(vs.YUV444P16)
    full = src.resize.Point(range_in=0, range=1).std.SetFrameProps(_Range=1)
    out = full.vszip.Tonemap(dynamic_peak_detection=0)
    assert props(out)["_Range"] == 1
    with out.get_frame(0) as f:
        # full-range SDR Y reaches below the limited-range black code 4096
        assert np.asarray(f[0]).min() < 4096


# --- independent oracle ---------------------------------------------------

PQ_M1, PQ_M2 = 2610 / 4096 / 4, 2523 / 4096 * 128
PQ_C1, PQ_C2, PQ_C3 = 3424 / 4096, 2413 / 4096 * 32, 2392 / 4096 * 32


def _pq_eotf(x):
    xp = np.power(np.clip(x, 0, 1), 1 / PQ_M2)
    return np.power(np.maximum(xp - PQ_C1, 0) / (PQ_C2 - PQ_C3 * xp), 1 / PQ_M1)


def _pq_oetf(y):
    yp = np.power(np.maximum(y, 0), PQ_M1)
    return np.power((PQ_C1 + PQ_C2 * yp) / (1 + PQ_C3 * yp), PQ_M2)


def _rgb2xyz(prim):
    (rx, ry), (gx, gy), (bx, by), (wx, wy) = prim
    m = np.array([[rx / ry, gx / gy, bx / by], [1, 1, 1], [(1 - rx - ry) / ry, (1 - gx - gy) / gy, (1 - bx - by) / by]])
    s = np.linalg.solve(m, np.array([wx / wy, 1, (1 - wx - wy) / wy]))
    return m * s


def _rgb2lms(prim):
    hpe = np.array([[0.40024, 0.70760, -0.08081], [-0.22630, 1.16532, 0.04570], [0, 0, 0.91822]])
    c = 0.04
    cross = np.array([[1 - 2 * c, c, c], [c, 1 - 2 * c, c], [c, c, 1 - 2 * c]])
    return cross @ hpe @ _rgb2xyz(prim)


BT2020 = ((0.708, 0.292), (0.170, 0.797), (0.131, 0.046), (0.3127, 0.3290))
BT709 = ((0.640, 0.330), (0.300, 0.600), (0.150, 0.060), (0.3127, 0.3290))
P3 = ((0.680, 0.320), (0.265, 0.690), (0.150, 0.060), (0.3127, 0.3290))
LMS2IPT = np.array([[0.4, 0.4, 0.2], [4.4550, -4.8510, 0.3960], [0.8056, 0.3572, -1.1628]])


def _nits2pq(n):
    return _pq_oetf(n / 10000.0)


def _reference_clip_clip(rgb16):
    """HDR10 (full-range RGB48, PQ, BT.2020, 1000-nit mastering peak, no
    mastering black) -> SDR RGB48 with tone_mapping_function=0 and
    gamut_mapping=0, from libplacebo v7.387's formulas: the pixel moves to
    the mastering display's RGB (vs-placebo's default, Display P3), the
    curve, which only clamps to [0, PQ(1000)] (the infinite-contrast black
    is snapped to 0), applies per cone of PQ LMS and is blended with the
    maxRGB channel's gain by the film response (strength 0.3), then LMS ->
    BT.709, BT.1886 with the SDR target's 203 / 0.203 nits."""
    rgb = rgb16.astype(np.float64) / 65535.0
    lin = _pq_eotf(rgb) * (10000.0 / 203.0)
    to_p3 = np.linalg.inv(_rgb2xyz(P3)) @ _rgb2xyz(BT2020)
    lin = np.einsum("ij,jhw->ihw", to_p3, lin)
    lms = np.einsum("ij,jhw->ihw", _rgb2lms(P3) * (203.0 / 10000.0), lin)
    pk = _nits2pq(1000.0)
    tone = lambda x: _pq_eotf(np.clip(x, 0, pk))  # noqa: E731
    o = tone(_pq_oetf(lms))
    lx = np.clip(lin.max(axis=0) * (203.0 / 10000.0), 0, 1)
    fl = 0.203 / 10000.0
    ga = np.maximum(tone(_pq_oetf(lx)) - fl, 0) / np.maximum(lx, 1e-9)
    # the knee (knee_default of the PQ input range) maps onto itself
    knee_in = _pq_eotf(0.4 * pk)
    knee_gain = (knee_in - fl) / knee_in
    u = np.where(lx > knee_in, np.clip(1 - ga / knee_gain, 0, 1), 0) ** 2
    s = np.clip(lms / np.maximum(lms.max(axis=0), 1e-9), 0, 1)
    w = 0.3 * (s + (1 - s) * u)
    lms = fl + lms * ga + (o - fl - lms * ga) * w
    out = np.einsum("ij,jhw->ihw", np.linalg.inv(_rgb2lms(BT709)), np.maximum(lms, 0) * (10000.0 / 203.0))
    lb, lw = 0.001 ** (1 / 2.4), 1.0
    a, b = (lw - lb) ** 2.4, lb / (lw - lb)
    out = np.power(np.maximum(out, 0) / a, 1 / 2.4) - b
    return np.clip(np.floor(np.clip(out, 0, 1) * 65535 + 0.5), 0, 65535)


def test_clip_pipeline_matches_numpy_model(make_hdr):
    src = make_hdr(vs.RGB48).std.SetFrameProps(_Range=1)
    out = src.vszip.Tonemap(dynamic_peak_detection=0, tone_mapping_function=0, gamut_mapping=0)
    with src.get_frame(0) as fs, out.get_frame(0) as fo:
        x = np.stack([np.asarray(fs[p]) for p in range(3)])
        got = np.stack([np.asarray(fo[p]) for p in range(3)]).astype(np.float64)
    want = _reference_clip_clip(x)
    d = np.abs(got - want)
    # LUT-interpolated PQ (<= 8.4e-7 in the PQ domain) and approximated
    # pow: a few 16-bit codes (saturated colors amplify them most); a wrong
    # matrix or curve is off by thousands. Clipped (out-of-gamut) pixels keep
    # a near-black channel, where BT.1886's slope magnifies those errors, so
    # the tight bound skips them.
    unclipped = np.broadcast_to(((want > 0) & (want < 65535)).all(axis=0), d.shape)
    assert d[unclipped].max() <= 16, d[unclipped].max()
    assert np.percentile(d, 99.9) <= 8
    assert d.mean() <= 0.25


# --- metamorphic: pointwise paths, crops, strides ---------------------------


@pytest.mark.parametrize("fmt", [vs.RGB48, vs.RGBS, vs.YUV444P16, vs.YUV444PS])
@pytest.mark.parametrize("width", [64, 65, 71, 72, 73, 79, 80, 81, 95, 96, 97])
def test_crop_commutes_on_pointwise_paths(make_hdr, fmt, width):
    """Without chroma upsampling or peak detection every output pixel
    depends on its input pixel only, so cropping commutes with the filter:
    this checks the SIMD tails at widths around 8 and 16 lanes."""
    src = make_hdr(fmt)
    kw = dict(dynamic_peak_detection=0)
    crop = src.std.CropAbs(width=width, height=67, left=101, top=33)
    assert_same_clip(crop.vszip.Tonemap(**kw), src.vszip.Tonemap(**kw).std.CropAbs(width=width, height=67, left=101, top=33))


@pytest.mark.parametrize("fmt", [vs.YUV420P16, vs.YUV422P16, vs.YUV420PS, vs.RGB48])
def test_cropped_source_matches_repacked(make_hdr, fmt):
    src = make_hdr(fmt).std.Crop(left=6, right=10, top=4, bottom=2)
    assert_same_clip(src.vszip.Tonemap(), repack(src).vszip.Tonemap())


def test_float_and_16bit_paths_agree(make_hdr):
    a = make_hdr(vs.RGBS).vszip.Tonemap(dynamic_peak_detection=0)
    b = make_hdr(vs.RGBS).resize.Point(format=vs.RGB48).vszip.Tonemap(dynamic_peak_detection=0)
    with a.get_frame(0) as fa, b.get_frame(0) as fb:
        x = np.stack([np.asarray(fa[p]) for p in range(3)]).astype(np.float64)
        y = np.stack([np.asarray(fb[p]) for p in range(3)]).astype(np.float64) / 65535
    # input quantization to 16 bit (and output rounding) is the only
    # difference; it grows only where BT.1886 is steep, near black
    d = np.abs(np.clip(x, 0, 1) - y)
    assert np.percentile(d, 99.9) < 1e-4
    assert d.max() < 1e-2


def test_hdr10_to_hdr10_is_near_identity(make_hdr):
    """Same space in and out with the target peak at the mastering peak and
    gamut clipping: tone and gamut mapping are no-ops, only decode/encode and
    the LUTs run. (The default perceptual mapping is bidirectional: it maps
    the BT.2020 container into the P3 mastering gamut, so it is not a no-op.)"""
    src = make_hdr(vs.YUV444P16)
    out = src.vszip.Tonemap(dst_csp=1, dst_max=1000.0, dynamic_peak_detection=0, gamut_mapping=0)
    assert max_abs_diff_all_planes(out, src) <= 2
    mapped = src.vszip.Tonemap(dst_csp=1, dst_max=1000.0, dynamic_peak_detection=0)
    assert max_abs_diff_all_planes(mapped, src) > 1000


# --- peak detection scheduling ----------------------------------------------


@pytest.fixture(scope="module")
def scene_clip(src_rgb):
    """12 frames, a bright scene and a dark one, with slight changes per frame
    so the IIR smoothing and scene-change hysteresis both act."""
    frames = []
    for n in range(12):
        gain = (12.0 if n < 6 else 1.5) * (1 + 0.03 * (n % 6))
        frames.append(_hdr(src_rgb, vs.YUV420P16, gain=gain))
    clip = frames[0]
    for f in frames[1:]:
        clip = clip + f
    return clip


def _frame_arrays(clip, order, threads=1):
    def get(n):
        with clip.get_frame(n) as f:
            return n, [np.asarray(f[p]).copy() for p in range(3)]

    if threads == 1:
        return dict(get(n) for n in order)
    with ThreadPoolExecutor(threads) as ex:
        return dict(ex.map(get, order))


def test_peak_detection_ignores_request_order(scene_clip):
    n = scene_clip.num_frames
    seq = _frame_arrays(scene_clip.vszip.Tonemap(), range(n))
    rev = _frame_arrays(scene_clip.vszip.Tonemap(), reversed(range(n)))
    mid = _frame_arrays(scene_clip.vszip.Tonemap(), [7, 0, 11, 3, 3, 9, 1, 5, 2, 10, 4, 6, 8])
    par = _frame_arrays(scene_clip.vszip.Tonemap(), list(range(n)) * 2, threads=8)
    for k in range(n):
        for other in (rev, mid, par):
            for p in range(3):
                assert np.array_equal(seq[k][p], other[k][p]), (k, p)


def test_peak_detection_depends_on_earlier_frames_only(scene_clip):
    """Frame n replays frames up to n: changing later frames leaves it as is,
    changing an earlier one in the window does not."""
    base = scene_clip.vszip.Tonemap()
    later = (scene_clip[:8] + scene_clip[0] * 4).vszip.Tonemap()
    assert_same_clip(base[:8], later[:8])
    earlier = (scene_clip[:3] + scene_clip[9] + scene_clip[4:]).vszip.Tonemap()
    assert max_abs_diff_all_planes(base, earlier, n=5) > 0


def test_peak_detection_changes_output_and_follows_scenes(scene_clip):
    static = scene_clip.vszip.Tonemap(dynamic_peak_detection=0)
    dyn = scene_clip.vszip.Tonemap()
    # the dark scene is tone mapped against its own (much lower) peak
    assert max_abs_diff_all_planes(static, dyn, n=11) > 1000


@pytest.mark.parametrize(
    ("kw", "frames"),
    [
        ({}, 100),
        (dict(smoothing_period=60.0), 111),  # window 300, cut at frame 0
        (dict(smoothing_period=0.0), 1),
        (dict(dynamic_peak_detection=0), 1),
        (dict(metadata=1), 1),  # none
        (dict(metadata=2), 1),  # hdr10
        (dict(metadata=3), 1),  # hdr10plus
        (dict(metadata=4), 100),  # cie_y
        (dict(dst_csp=1), 1),  # 10000-nit target: never above it
        (dict(dst_csp=1, dst_max=1000.0), 100),
        (dict(src_csp=2), 100),  # HLG to SDR: peak from per-frame metadata
        (dict(src_csp=2, dst_csp=1, dst_max=1000.0), 1),  # HLG peak = target's
    ],
    ids=str,
)
def test_peak_window_requests(core, kw, frames):
    """Frame n fetches source frames n - window + 1 .. n (window = 5
    smoothing periods), and only its own when no frame can use detection."""
    seen, lock = set(), threading.Lock()

    def count(n, f):
        with lock:
            seen.add(n)
        return f

    base = core.std.BlankClip(width=64, height=64, format=vs.RGB48, length=120, color=[30000, 40000, 20000])
    base = base.std.SetFrameProps(_Range=1, MasteringDisplayMaxLuminance=1000.0)
    src = base.std.ModifyFrame(clips=base, selector=count)
    src.vszip.Tonemap(gamut_mapping=0, **kw).get_frame(110)
    assert len(seen) == frames
    assert max(seen) == 110 and min(seen) == 111 - frames


def test_unsmoothed_detection_is_per_frame(scene_clip):
    kw = dict(smoothing_period=0.0, scene_threshold_low=0.0)
    whole = scene_clip.vszip.Tonemap(**kw)
    for n in (0, 6, 11):
        assert_same_clip(whole[n], scene_clip[n].vszip.Tonemap(**kw))


@pytest.mark.parametrize("scene_max", [[700.0], [700.0, 650.0, 600.0]], ids=["cie_y", "hdr10plus"])
def test_scene_metadata_and_detection(make_hdr, scene_max):
    """PLSceneMax/PLSceneAvg (CIE Y with one value, HDR10+ with three) are
    used as given. libplacebo v7.387 skips its own detection only when both
    kinds are present, and the measurement fills in the other kind's fields
    (and drives the mastering-peak clip), so detection still changes the
    output."""
    src = make_hdr(vs.YUV420P16).std.SetFrameProps(PLSceneAvg=150.0, PLSceneMax=scene_max)
    plain = make_hdr(vs.YUV420P16).vszip.Tonemap(dynamic_peak_detection=0)
    assert max_abs_diff_all_planes(src.vszip.Tonemap(dynamic_peak_detection=0), plain) > 0
    assert max_abs_diff_all_planes(src.vszip.Tonemap(), src.vszip.Tonemap(dynamic_peak_detection=0)) > 0


# --- libplacebo parameters vs-placebo does not expose ---------------------------

# every one at libplacebo's default (spline, perceptual: the defaults)
LIBPLACEBO_DEFAULTS = dict(
    lut_size=256,
    mastering_clip=1.0,
    film_strength=0.3,
    inverse_tone_mapping=0,
    knee_adaptation=0.4,
    knee_minimum=0.1,
    knee_maximum=0.8,
    knee_default=0.4,
    knee_offset=1.0,
    slope_tuning=1.5,
    slope_offset=0.2,
    spline_contrast=0.5,
    reinhard_contrast=0.5,
    linear_knee=0.3,
    exposure=1.0,
    gamut_expansion=0,
    lut3d_size=[48, 32, 256],
    lut3d_tricubic=0,
    perceptual_deadzone=0.3,
    perceptual_strength=0.8,
    colorimetric_gamma=1.8,
    softclip_knee=0.7,
    softclip_desat=0.35,
)


@pytest.mark.parametrize("kw", [dict(dynamic_peak_detection=0), {}], ids=["static", "peak"])
def test_libplacebo_params_at_defaults_change_nothing(make_hdr, kw):
    src = make_hdr(vs.YUV420P16)
    assert_same_clip(src.vszip.Tonemap(**kw), src.vszip.Tonemap(**kw, **LIBPLACEBO_DEFAULTS))


def test_lut3d_size_zero_keeps_defaults(make_hdr):
    src = make_hdr(vs.YUV420P16)
    kw = dict(dynamic_peak_detection=0)
    assert_same_clip(src.vszip.Tonemap(**kw), src.vszip.Tonemap(**kw, lut3d_size=[0, 0, 0]))


@pytest.mark.parametrize(
    ("base", "change"),
    [
        ({}, dict(film_strength=0.0)),
        ({}, dict(film_strength=1.5)),
        ({}, dict(lut_size=64)),
        ({}, dict(spline_contrast=1.0)),
        ({}, dict(slope_tuning=4.0)),
        ({}, dict(slope_offset=0.6)),
        ({}, dict(knee_minimum=0.3, knee_maximum=0.6, knee_default=0.45)),
        (dict(tone_mapping_function_s="st2094-40"), dict(knee_adaptation=0.9)),
        (dict(tone_mapping_function_s="bt2390"), dict(knee_offset=0.5)),
        (dict(tone_mapping_function_s="reinhard"), dict(reinhard_contrast=0.2)),
        (dict(tone_mapping_function_s="mobius"), dict(linear_knee=0.6)),
        (dict(tone_mapping_function_s="linear"), dict(exposure=2.0)),
        ({}, dict(perceptual_deadzone=0.0)),
        ({}, dict(perceptual_strength=0.2)),
        (dict(gamut_mapping=2), dict(softclip_knee=0.4)),
        (dict(gamut_mapping=2), dict(softclip_desat=0.0)),
        ({}, dict(lut3d_size=[17, 17, 65])),
        ({}, dict(lut3d_tricubic=1)),
        (dict(dst_csp=1), dict(gamut_expansion=1)),
    ],
    ids=str,
)
def test_libplacebo_param_changes_output(make_hdr, base, change):
    src = make_hdr(vs.YUV444P16)
    kw = dict(dynamic_peak_detection=0, **base)
    assert max_abs_diff_all_planes(src.vszip.Tonemap(**kw), src.vszip.Tonemap(**kw, **change)) > 0


def test_mastering_clip_needs_detection(make_hdr):
    """The mastering-peak soft clip is gated by the measured peak, so it only
    acts with peak detection on."""
    src = make_hdr(vs.YUV444P16)
    static = dict(dynamic_peak_detection=0)
    assert_same_clip(src.vszip.Tonemap(**static), src.vszip.Tonemap(**static, mastering_clip=0.0))
    assert max_abs_diff_all_planes(src.vszip.Tonemap(), src.vszip.Tonemap(mastering_clip=0.0)) > 0


@pytest.mark.parametrize(
    ("fn", "param", "constant"),
    [
        ("spline", 0.8, dict(spline_contrast=0.8)),
        ("st2094-40", 0.6, dict(knee_adaptation=0.6)),
        ("bt2390", 1.5, dict(knee_offset=1.5)),
        ("reinhard", 0.3, dict(reinhard_contrast=0.3)),
        ("gamma", 0.5, dict(linear_knee=0.5)),
        ("linear", 2.0, dict(exposure=2.0)),
    ],
)
def test_tone_mapping_param_sets_its_constant(make_hdr, fn, param, constant):
    """tone_mapping_param (vs-placebo's) is the function's tone constant, as
    in pl_tone_map_params_infer; it overrides an explicit constant too."""
    src = make_hdr(vs.YUV444P16)
    kw = dict(dynamic_peak_detection=0, tone_mapping_function_s=fn)
    via_param = src.vszip.Tonemap(**kw, tone_mapping_param=param)
    assert_same_clip(via_param, src.vszip.Tonemap(**kw, **constant))
    other = {k: (v + 0.1 if v < 1 else v - 0.1) for k, v in constant.items()}
    assert_same_clip(via_param, src.vszip.Tonemap(**kw, **other, tone_mapping_param=param))


def test_st2094_default_knee_adaptation_is_param_def(make_hdr):
    """vs-placebo passes param_def (0.7) when tone_mapping_param is absent,
    not the constant's own default (0.4)."""
    src = make_hdr(vs.YUV444P16)
    kw = dict(dynamic_peak_detection=0, tone_mapping_function_s="st2094-40")
    assert_same_clip(src.vszip.Tonemap(**kw), src.vszip.Tonemap(**kw, knee_adaptation=0.7))
    assert max_abs_diff_all_planes(src.vszip.Tonemap(**kw), src.vszip.Tonemap(**kw, knee_adaptation=0.4)) > 0


def _mean_y(clip):
    with clip.get_frame(0) as f:
        return float(np.asarray(f[0], dtype=np.float64).mean())


def test_inverse_tone_mapping_expands(make_hdr):
    """Without it a brighter target never brightens the source (the curve is
    clamped to the source range); with it, an invertible curve expands it."""
    src = make_hdr(vs.YUV444P16)
    kw = dict(dynamic_peak_detection=0, dst_csp=1, dst_max=4000.0, gamut_mapping=0)
    plain = src.vszip.Tonemap(**kw)
    inv = src.vszip.Tonemap(**kw, inverse_tone_mapping=1)
    assert max_abs_diff_all_planes(plain, src) <= 2
    assert _mean_y(inv) > _mean_y(plain) + 100
    # functions without an inverse stay as they are
    kw_h = dict(kw, tone_mapping_function_s="hable")
    assert_same_clip(src.vszip.Tonemap(**kw_h), src.vszip.Tonemap(**kw_h, inverse_tone_mapping=1))


def test_lut3d_tricubic_stays_close(make_hdr):
    """Tricubic sampling is a cubic B-spline (approximating, not
    interpolating), so it smooths the gamut LUT's corners: on this saturated
    image the mean change from trilinear is ~48 codes. A coordinate or
    weight bug would move it by thousands. (The kernel test "tricubic gamut
    lookup reproduces an affine LUT" checks the formula itself.)"""
    src = make_hdr(vs.YUV444P16)
    kw = dict(dynamic_peak_detection=0)
    with src.vszip.Tonemap(**kw).get_frame(0) as fa, src.vszip.Tonemap(**kw, lut3d_tricubic=1).get_frame(0) as fb:
        a = np.stack([np.asarray(fa[p]) for p in range(3)]).astype(np.int64)
        b = np.stack([np.asarray(fb[p]) for p in range(3)]).astype(np.int64)
    d = np.abs(a - b)
    assert 0 < d.mean() <= 128, d.mean()


def test_lut3d_tricubic_fused_matches_split(make_hdr):
    """show_clipping runs the fused loop; away from clipped pixels it must
    compute what the split passes do. The exact check is the kernel test
    "fused color map matches the split passes" (nothing clipped, both
    lookups); here, end to end, only highlighted pixels may differ, a
    minority on this saturated image."""
    src = make_hdr(vs.YUV444PS)
    kw = dict(dynamic_peak_detection=0, lut3d_tricubic=1)
    with src.vszip.Tonemap(**kw).get_frame(0) as fa, src.vszip.Tonemap(**kw, show_clipping=1).get_frame(0) as fb:
        a = np.stack([np.asarray(fa[p]) for p in range(3)])
        b = np.stack([np.asarray(fb[p]) for p in range(3)])
    same = (a == b).all(axis=0)
    assert same.mean() > 0.5


# --- vs-placebo parity (optional) -------------------------------------------


def _placebo(core):
    # VSPLACEBO_LIB: a vs-placebo DLL to load when none is autoloaded (e.g. a
    # local build with a newer libplacebo)
    lib = os.environ.get("VSPLACEBO_LIB")
    if not hasattr(core, "placebo") and lib:
        core.std.LoadPlugin(lib)
    if not hasattr(core, "placebo"):
        pytest.skip("vs-placebo not installed (or set VSPLACEBO_LIB)")
    return core.placebo


@pytest.mark.parametrize(
    "kw",
    [
        dict(dynamic_peak_detection=0),
        dict(dynamic_peak_detection=0, tone_mapping_function=3),
        dict(dynamic_peak_detection=0, gamut_mapping=2),
        dict(dynamic_peak_detection=0, dst_csp=1),
    ],
    ids=str,
)
def test_matches_vs_placebo(core, src_rgb, kw):
    """Parity with libplacebo v7.387 on the GPU (vs-placebo 2.0.4; older
    builds differ in the algorithm). Without peak detection the remaining
    error is LUT/approximation noise, largest on this image's saturated
    colors (mean 2-7 codes, p99.9 below 300). With detection the GPU stores
    the decoded frame in a half-float FBO that truncates, which darkens
    bright frames by up to ~0.1%, so detection is left out here."""
    pl = _placebo(core)
    src = _hdr(src_rgb, vs.RGB48).std.SetFrameProps(_Range=1)
    try:
        ref = pl.Tonemap(src, src_csp=1, **kw)
        ref.get_frame(0)
    except vs.Error as e:
        pytest.skip(f"vs-placebo unusable here: {e}")
    out = src.vszip.Tonemap(src_csp=1, **kw)
    with out.get_frame(0) as fa, ref.get_frame(0) as fb:
        a = np.stack([np.asarray(fa[p]) for p in range(3)]).astype(np.int64)
        b = np.stack([np.asarray(fb[p]) for p in range(3)]).astype(np.int64)
    d = np.abs(a - b)
    assert d.mean() <= 8, d.mean()
    assert np.percentile(d, 99.9) <= 400, np.percentile(d, 99.9)


# --- validation --------------------------------------------------------------


@pytest.mark.parametrize(
    ("fmt", "kw", "message"),
    [
        (vs.YUV420P10, {}, "16-bit integer or 32-bit float"),
        (vs.YUV420P8, {}, "16-bit integer or 32-bit float"),
        (vs.GRAY16, {}, "16-bit integer or 32-bit float"),
        (vs.YUV420PH, {}, "16-bit integer or 32-bit float"),
        (vs.YUV410P16, {}, "16-bit integer or 32-bit float"),
        (vs.YUV420P16, dict(src_csp=0), "src_csp"),
        (vs.YUV420P16, dict(src_csp=3), "Dolby Vision"),
        (vs.YUV420P16, dict(use_dovi=1), "Dolby Vision"),
        (vs.YUV420P16, dict(dst_csp=3), "Dolby Vision"),
        (vs.YUV420P16, dict(dst_csp=4), "dst_csp"),
        (vs.YUV420P16, dict(src_csp=-1), "src_csp"),
        (vs.YUV420P16, dict(dst_prim=18), "dst_prim"),
        (vs.YUV420P16, dict(gamut_mapping=10), "gamut_mapping"),
        (vs.YUV420P16, dict(tone_mapping_function=12), "tone_mapping_function"),
        (vs.YUV420P16, dict(tone_mapping_function_s="nope"), "tone_mapping_function_s"),
        (vs.YUV420P16, dict(metadata=5), "metadata"),
        (vs.YUV420P16, dict(percentile=101.0), "percentile"),
        (vs.YUV420P16, dict(smoothing_period=61.0), "smoothing_period"),
        (vs.YUV420P16, dict(src_max=-1.0), "src_max"),
        (vs.YUV420P16, dict(dst_max=20000.0), "dst_max"),
        (vs.YUV420P16, dict(contrast_smoothness=0.5), "contrast_smoothness"),
        (vs.YUV420P16, dict(tone_mapping_param=11.0), "tone_mapping_param"),
        (vs.YUV420P16, dict(dynamic_peak_detection=2), "dynamic_peak_detection"),
        (vs.YUV420P16, dict(lut_size=1), "lut_size"),
        (vs.YUV420P16, dict(lut_size=4097), "lut_size"),
        (vs.YUV420P16, dict(mastering_clip=1.5), "mastering_clip"),
        (vs.YUV420P16, dict(film_strength=2.5), "film_strength"),
        (vs.YUV420P16, dict(inverse_tone_mapping=2), "inverse_tone_mapping"),
        (vs.YUV420P16, dict(knee_minimum=0.6), "knee_minimum"),
        (vs.YUV420P16, dict(knee_maximum=0.4), "knee_maximum"),
        (vs.YUV420P16, dict(knee_offset=0.4), "knee_offset"),
        (vs.YUV420P16, dict(spline_contrast=1.6), "spline_contrast"),
        (vs.YUV420P16, dict(exposure=11.0), "exposure"),
        (vs.YUV420P16, dict(perceptual_strength=-0.1), "perceptual_strength"),
        (vs.YUV420P16, dict(colorimetric_gamma=11.0), "colorimetric_gamma"),
        (vs.YUV420P16, dict(gamut_expansion=2), "gamut_expansion"),
        (vs.YUV420P16, dict(lut3d_tricubic=2), "lut3d_tricubic"),
        (vs.YUV420P16, dict(lut3d_size=[48, 32]), "lut3d_size must have 3 values"),
        (vs.YUV420P16, dict(lut3d_size=[48, 32, 256, 4]), "lut3d_size must have 3 values"),
        (vs.YUV420P16, dict(lut3d_size=[1, 32, 256]), "lut3d_size values"),
        (vs.YUV420P16, dict(lut3d_size=[48, 32, 1025]), "lut3d_size values"),
        (vs.YUV420P16, dict(lut3d_size=[1024, 1024, 17]), "lut3d_size is too large"),
    ],
)
def test_rejects(core, fmt, kw, message):
    clip = core.std.BlankClip(width=96, height=96, format=fmt, length=1)
    with pytest.raises(vs.Error, match=message):
        clip.vszip.Tonemap(**kw).get_frame(0)


@pytest.mark.parametrize(("w", "h"), [(62, 64), (64, 62)])
def test_rejects_small_input(core, w, h):
    clip = core.std.BlankClip(width=w, height=h, format=vs.YUV420P16, length=1)
    with pytest.raises(vs.Error, match="at least 64x64"):
        clip.vszip.Tonemap()


def test_minimum_size_works(core):
    clip = core.std.BlankClip(width=64, height=64, format=vs.YUV420P16, length=2, color=[30000, 32768, 32768])
    out = clip.vszip.Tonemap()
    for n in range(2):
        with out.get_frame(n) as f:
            assert np.ptp(np.asarray(f[0])) == 0
