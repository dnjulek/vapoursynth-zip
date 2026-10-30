"""Contract and oracles for Tonemap.

Tonemap ports vs-placebo's Tonemap: libplacebo's renderer path for one frame
(YCbCr BT.2020 NCL decode with lanczos chroma upsampling, PQ/HLG linearize,
pl_shader_color_map_ex in IPT with a 1D tone LUT and a 3D gamut LUT,
delinearize, encode), following libplacebo v7.387. It is not 1:1
with the GPU: PQ transfers are LUT-interpolated, pow/exp/log approximated,
the gamut LUT stays f32 and 16-bit output is rounded instead of dithered.

Observable contract (src/vapoursynth/tonemap.zig, src/filters/tonemap.zig):

- input: 16-bit integer or 32-bit float, RGB or YUV with 2x or no
  subsampling per axis. Output: RGB48 / YUV444P16 (float:
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


def _rgb2xyz(prim, white=(0.3127, 0.3290)):
    m = np.array([[x / y, 1.0, (1 - x - y) / y] for x, y in prim]).T
    w = np.array([white[0] / white[1], 1.0, (1 - white[0] - white[1]) / white[1]])
    return m * np.linalg.solve(m, w)


_BT709_TO_BT2020 = np.linalg.solve(_rgb2xyz(((0.708, 0.292), (0.170, 0.797), (0.131, 0.046))),
                                   _rgb2xyz(((0.64, 0.33), (0.30, 0.60), (0.15, 0.06))))


def _hlg_oetf(e):
    a = 0.17883277
    b, c = 1 - 4 * a, 0.5 - a * np.log(4 * a)
    return np.where(e <= 1 / 12, np.sqrt(3 * e), a * np.log(np.maximum(12 * e - b, 1e-12)) + c)


def _subsample(c, f):
    """Chroma to the format's subsampling: rows averaged in pairs (centered),
    columns [1 2 1] / 4 around the even ones (left siting, _ChromaLocation 0)."""
    if f.subsampling_h:
        c = 0.5 * (c[0::2] + c[1::2])
    if f.subsampling_w:
        p = np.pad(c, ((0, 0), (1, 1)), mode="edge")
        c = 0.25 * p[:, 0:-2:2] + 0.5 * p[:, 1:-1:2] + 0.25 * p[:, 2::2]
    return c


def _hdr(src_rgb, fmt, trc="st2084", gain=12.0, meta=True):
    """The test image as HDR: linear BT.2020 light expanded to ~1200 nits
    (x^2 * 12, 1.0 = 100 nits), encoded with `trc` (PQ, or HLG of the scene
    light gain / 10, clipped at 1). The image's blue is nearly constant
    (252..255), so blue takes the mirrored green instead.

    Computed in numpy float64, not with resize's transfer conversions: zimg
    picks its SIMD path by CPU, and its AVX-512 path rounds differently from
    AVX2 (CI runners have AVX-512), which moved every golden average."""
    f = vs.core.get_video_format(fmt)
    with src_rgb.get_frame(0) as fr:
        rgb = np.stack([np.asarray(fr[p], dtype=np.float64) for p in range(3)]) / 255
    # BT.709 inverse OETF, then BT.2020 primaries
    lin = np.where(rgb < 0.081, rgb / 4.5, ((rgb + 0.099) / 1.099) ** (1 / 0.45))
    lin = np.einsum("ij,jhw->ihw", _BT709_TO_BT2020, lin)
    lin = np.stack([lin[0], lin[1], lin[1][:, ::-1]])
    lin = np.maximum(lin, 0) ** 2 * gain
    e = _pq_oetf(lin * 100 / 10000) if trc == "st2084" else _hlg_oetf(np.minimum(lin / 10, 1))

    peak = (1 << f.bits_per_sample) - 1
    integer = f.sample_type == vs.INTEGER
    if f.color_family == vs.RGB:
        planes = [np.clip(np.rint(v * peak), 0, peak) if integer else v for v in e]
        props = dict(_Matrix=0, _Range=1)
    else:
        y = 0.2627 * e[0] + 0.6780 * e[1] + 0.0593 * e[2]
        cb, cr = _subsample((e[2] - y) / 1.8814, f), _subsample((e[0] - y) / 1.4746, f)
        if integer:
            s = 1 << (f.bits_per_sample - 8)
            planes = [np.clip(np.rint(16 * s + 219 * s * y), 0, peak)]
            planes += [np.clip(np.rint(128 * s + 224 * s * c), 0, peak) for c in (cb, cr)]
        else:
            planes = [y, cb, cr]
        props = dict(_Matrix=9, _Range=0)
        if f.subsampling_w or f.subsampling_h:
            props["_ChromaLocation"] = 0
    dtype = np.uint16 if integer else np.float32
    planes = [p.astype(dtype) for p in planes]

    blank = vs.core.std.BlankClip(width=src_rgb.width, height=src_rgb.height, format=fmt, length=1)

    def fill(n, f):
        out = f.copy()
        for p in range(3):
            np.asarray(out[p])[:] = planes[p]
        return out

    out = blank.std.ModifyFrame(blank, fill).std.SetFrameProps(_Primaries=9, _Transfer=16 if trc == "st2084" else 18, **props)
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
        # DolbyVisionRPU attached (RPU_P8 / RPU_P5 below)
        Case(vs.YUV420P16, args=dict(src_csp=3, dynamic_peak_detection=0), variant="dovi8"),
        Case(vs.YUV420P16, args=dict(src_csp=3), variant="dovi8"),
        Case(vs.YUV444P16, args=dict(src_csp=3, dst_csp=1, dynamic_peak_detection=0), variant="dovi8"),
        Case(vs.YUV420P16, "odd", dict(src_csp=3, dynamic_peak_detection=0), variant="dovi5"),
        Case(vs.YUV420P16, args=dict(src_csp=1, dynamic_peak_detection=0), variant="dovi8"),
    ]
)


@pytest.mark.parametrize("case", CASES, ids=str)
def test_golden_cases(golden, make_hdr, case):
    src = make_hdr(case.fmt, case.geometry, "std-b67" if case.variant == "hlg" else "st2084")
    if case.variant in ("dovi8", "dovi5"):
        src = _dv(src, RPU_P8 if case.variant == "dovi8" else RPU_P5)
    args = {"src_csp": 1, **case.args}
    out = src.vszip.Tonemap(**args)
    # Builds without FMA (the baseline x86_64 wheel, x86_64_v3-fma) round
    # differently, as the project allows for non-bit-exact filters: single
    # extreme pixels then move by up to 5 16-bit codes (measured over every
    # case here) or ~1.2e-5 in float, and the (0-1 normalized) averages by at
    # most 7e-8, which near 0 (centered float chroma) exceeds the relative
    # 1e-6. A material matrix, curve or decode change moves the averages by
    # far more than the 1e-6 absolute floor (~0.07 of a 16-bit code).
    is_float = vs.core.get_video_format(case.fmt).sample_type == vs.FLOAT
    golden.check("tonemap", case, out, extrema_abs=1e-4 if is_float else 8, avg_abs=1e-6)


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


@pytest.mark.parametrize("bits", [10, 12])
@pytest.mark.parametrize(
    ("fmt16", "kw", "rpu"),
    [
        (vs.YUV420P16, dict(), None),
        (vs.YUV444P16, dict(dst_csp=1, dynamic_peak_detection=0), None),
        (vs.YUV420P16, dict(src_csp=3), "p8"),
        (vs.YUV420P16, dict(src_csp=3, dynamic_peak_detection=0), "p5"),
    ],
    ids=str,
)
def test_low_bit_depth_matches_shifted_16bit(make_hdr, bits, fmt16, kw, rpu):
    """10/12-bit limited-range input decodes exactly like the same codes
    shifted to 16 bit (zimg's resize.Point, what vs-placebo needs): the
    2^(16 - bits) is a power of two folded into the decode matrix, and the
    Dolby Vision code scale is code / 2^bits either way."""
    f16 = vs.core.get_video_format(fmt16)
    low = make_hdr(fmt16).resize.Point(format=f16.replace(bits_per_sample=bits))
    if rpu:
        low = _dv(low, RPU_P8 if rpu == "p8" else RPU_P5)
    shifted = low.resize.Point(format=fmt16)
    assert_same_clip(low.vszip.Tonemap(**kw), shifted.vszip.Tonemap(**kw))


def test_full_range_10bit_matches_16bit(make_hdr):
    """Full range takes 10-bit codes to 16 bit with BT.2100's scaling (luma by
    65535/1023, chroma by the same around 512 -> 32768), like zimg's
    resize.Point but unrounded, so the two paths differ only by zimg's input
    rounding (half a 16-bit code)."""
    src = make_hdr(vs.YUV444P16).std.SetFrameProps(_Range=1)
    low = src.resize.Point(format=vs.YUV444P10, range_in=1, range=1)
    a = low.vszip.Tonemap(dst_csp=1, dynamic_peak_detection=0)
    b = low.resize.Point(format=vs.YUV444P16, range_in=1, range=1).vszip.Tonemap(dst_csp=1, dynamic_peak_detection=0)
    with a.get_frame(0) as fa, b.get_frame(0) as fb:
        d = np.abs(np.stack([np.asarray(fa[p]) for p in range(3)]).astype(np.int64)
                   - np.stack([np.asarray(fb[p]) for p in range(3)]).astype(np.int64))
    assert d.mean() <= 0.6, d.mean()
    assert d.max() <= 4, d.max()


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


# --- Dolby Vision -------------------------------------------------------------

# sample RPUs (HEVC unspec62 NAL units) from dovi_tool's assets/tests
RPU_P8 = bytes.fromhex(
    "0000000119080908406136506f003ff801ffc00fffd0000008000006800000400000340000030200000301a2566000035ea2566f9fceb1c25664"
    "4ca00000100000030080000003008000000301c36224301860a5e308e0514000001a63e5affff000000300000300000300060207dce01518300"
    "80169b99980c028218008008005b6854100020200000300080500000300000300000300120c1f4000640000030000d2d3f72280")
RPU_P5 = bytes.fromhex(
    "00000001190809004061b6506f003ff801ffc00fffd000001000001b00000302000003036000004000006880000c7c1a448003f16c110c80000"
    "42fa95c0000030000200000030020000003010ae7fa8ffa8ffa8d0ae7fa8ffa8ffa8d0ae7fffc00000300000300000300019081f73805463008"
    "0169b99980c028218008008005b6854100020200000300080500000300000300000300120c1f40006400000300007b68c48480")
RPU_FEL = bytes.fromhex(
    "000000011908090840613650ae20002008020080200802007f801ffc00ffc001fffa0000030100000300d0000008000006800000400000"
    "340000030200000301a000001000000d0000030080000068000004000003034000002000000a86d9e67b2f863efc189e6029e98c06b8c56"
    "6a7cd5f8dffcb6d8c39e38a9981b86ca6790f405f434cb7d80e9e04c349fd3bb77ec2cd5de3a147d44c0a2e8062164a029b5d59500ed38f5"
    "8c46c4ec13a14582cff3023536d57bbb6e411483b92966413698fce6bc31bd620fadd405bbef4643160b1f01bf2eefbf2a5e2f3ccb4abf3c"
    "c3983779e7568311b004800004004004000004012000010010010000010048000040040040000072566000035ea2566f9fceb1c256644ca0"
    "000030100000300080000030008000003001c36224301860a5e308e051400000301a63e5affff000003000003000003000060200f80e151"
    "8030080059ca1200c028218df82580080061400002022a512088050000030002281150120c07d000020d60015eb8f29d8280")
RPU_MEL = bytes.fromhex(
    "000000011908090840613650af003ff801ffc00ffc001fffa000001000000d0000030080000068000004000003000040000020000020000"
    "003000400000302000003020000030000400000200000200000392b300001af512b37cfe758e12b3226500000080000030040000003004000"
    "000300e1b112180c3052f1847028a000000d31f2d7fff8000003000003000003000301007c070a8c300800410a9e00c028215858004af800"
    "6cb00002021a8959880500000300000300000300120c07d0000205d2005e258dd4a680")
DV_SAMPLE = r"D:\dv_sample.mkv"
LG_DEMOS = (r"D:\dovi\others\LG Demo DolbyVision Trailer.mkv", r"D:\dovi\el\LG Demo DolbyVision Trailer EL.hevc")


def _crc32_mpeg2(data):
    crc = 0xFFFFFFFF
    for b in data:
        crc ^= b << 24
        for _ in range(8):
            crc = ((crc << 1) ^ 0x04C11DB7) if crc & 0x80000000 else crc << 1
            crc &= 0xFFFFFFFF
    return crc


class _Bits:
    def __init__(self, b):
        self.b, self.p = b, 0

    def u(self, n):
        v = 0
        for _ in range(n):
            v = (v << 1) | ((self.b[self.p >> 3] >> (7 - (self.p & 7))) & 1)
            self.p += 1
        return v

    def ue(self):
        z = 0
        while self.u(1) == 0:
            z += 1
        return (1 << z) - 1 + self.u(z)


def _rpu_fields(nalu):
    """Independent bit-level reader of a profile 8 RPU (dovi_tool's syntax):
    the unescaped payload, the bit offsets of vdr_dm_metadata_present_flag
    and of the DM data, and the DM matrices."""
    esc = nalu[4:]
    data = bytearray(esc[:2])
    for i in range(2, len(esc)):
        if not (esc[i - 2] == 0 and esc[i - 1] == 0 and esc[i] == 3):
            data.append(esc[i])
    r = _Bits(data)
    assert r.u(8) == 25 and r.u(6) == 2
    fmt = r.u(11)
    r.u(8)
    assert r.u(1)  # vdr_seq_info_present_flag
    r.u(1)
    assert r.u(2) == 0  # coefficient_data_type
    denom = r.ue()
    r.u(3)
    assert fmt & 0x700 == 0
    r.ue(), r.ue(), r.ue()
    r.u(5)
    disable_residual = r.u(1)
    dm_pos = r.p
    assert r.u(1) == 1 and r.u(1) == 0  # DM present, no use_prev
    r.ue(), r.ue(), r.ue()
    pivots = []
    for _ in range(3):
        n = r.ue() + 2
        pivots.append(n)
        r.u(10 * n)
    assert disable_residual  # no NLQ
    r.ue(), r.ue()
    for n in pivots:
        for _ in range(n - 1):
            if r.ue() == 0:  # polynomial
                order = r.ue() + 1
                if order == 1:
                    assert r.u(1) == 0
                for _ in range(order + 1):
                    r.ue(), r.u(denom)
            else:
                order = r.u(2) + 1
                r.ue(), r.u(denom)
                for _ in range(order * 7):
                    r.ue(), r.u(denom)
    dm_start = r.p
    r.ue(), r.ue(), r.ue()

    def s16():
        v = r.u(16)
        return v - 65536 if v & 0x8000 else v

    ycc = np.array([s16() for _ in range(9)], dtype=np.float64).reshape(3, 3) / 8192
    off = np.array([r.u(32) for _ in range(3)], dtype=np.float64) / (1 << 28)
    lms = np.array([s16() for _ in range(9)], dtype=np.float64).reshape(3, 3) / 16384
    return data, dm_pos, dm_start, ycc, off, lms


def _rpu_strip_dm(nalu):
    """The RPU with vdr_dm_metadata_present_flag cleared and its DM data
    dropped (CRC recomputed, emulation prevention redone): its frame keeps
    the previous frame's matrices."""
    data, dm_pos, dm_start, *_ = _rpu_fields(nalu)
    bits = [(data[i >> 3] >> (7 - (i & 7))) & 1 for i in range(dm_start)]
    bits[dm_pos] = 0
    bits += [0] * (-len(bits) % 8)
    out = bytearray(int("".join(map(str, bits[i:i + 8])), 2) for i in range(0, len(bits), 8))
    out += _crc32_mpeg2(out[1:]).to_bytes(4, "big") + b"\x80"
    escaped = bytearray()
    for b in out:
        if len(escaped) >= 2 and escaped[-2] == 0 and escaped[-1] == 0 and b <= 3:
            escaped.append(3)
        escaped.append(b)
    return bytes(escaped)


def _rpu_pivot_count(comp, count):
    """A CRC-correct profile 8.1 RPU (no DM data, identity curves) whose
    component `comp` signals num_pivots_minus2 = `count` (the others 0);
    an oversized count is followed by two pivots. Mirrors dovi.zig's
    TestRpu."""
    return _rpu_build(pivot_comp=comp, pivot_count=count)


# BT.2020 limited-range YCbCr -> RGB and an RGB -> LMS, in the RPU's fixed
# point (/8192, offsets /2^28, /16384); and a second, different set
_DM_A = ([9575, 0, 13806, 9575, -1541, -5349, 9575, 17617, 0], [16777216, 134217728, 134217728],
         [7222, 8771, 390, 2654, 12430, 1300, 0, 422, 15962])
_DM_B = ([9000, 300, 12000, 9800, -1200, -6000, 9300, 16000, 400], [20000000, 134217728, 134217728],
         [7000, 9000, 384, 2800, 12000, 1584, 100, 500, 15784])


def _rpu_build(dm=None, use_prev=False, compressed=False, pivot_comp=-1, pivot_count=0):
    """A CRC-correct profile 8.1 RPU with identity curves. dm: None (no DM
    data) or (ycc, offsets, lms) as in _DM_A; use_prev: the mapping is the
    previous frame's (use_prev_vdr_rpu_flag); compressed: compressed DM data
    (reserved_zero_3bits = 1, no matrices); pivot_comp/pivot_count: one
    component's num_pivots_minus2, possibly out of range."""
    bits = []

    def put(v, n):
        bits.extend((v >> (n - 1 - i)) & 1 for i in range(n))

    def ue(v):
        x = v + 1
        put(0, x.bit_length() - 1)
        put(x, x.bit_length())

    def se(v):
        ue(2 * v - 1 if v > 0 else -2 * v)

    has_dm = dm is not None or compressed
    put(25, 8)
    put(2, 6), put(18, 11), put(1, 4), put(0, 4), put(1, 1), put(0, 1), put(0, 2)
    ue(23)
    put(1, 2), put(0, 1)
    ue(2), ue(2), ue(4)
    put(0, 1), put(1 if compressed else 0, 3), put(0, 1), put(1, 1)  # reserved_zero_3bits, disable_residual_flag
    put(int(has_dm), 1), put(int(use_prev), 1)
    if use_prev:
        ue(0)  # prev_vdr_rpu_id
    else:
        ue(0), ue(0), ue(0)
        pieces = []
        for c in range(3):
            k = pivot_count if c == pivot_comp else 0
            ue(k)
            n = k + 2 if k <= 7 else 2
            pieces.append(n - 1)
            for p in range(n):
                put(0 if p == 0 else 100, 10)
        ue(0), ue(0)
        for n in pieces:
            for _ in range(n):
                ue(0), ue(0), put(0, 1)
                se(0), put(0, 23), se(1), put(0, 23)
    if has_dm:
        ue(0), ue(0), ue(0)  # affected/current_dm_metadata_id, scene_refresh_flag
        if not compressed:
            ycc, off, lms = dm
            for v in ycc:
                put(v & 0xFFFF, 16)
            for v in off:
                put(v, 32)
            for v in lms:
                put(v & 0xFFFF, 16)
            put(65535, 16), put(0, 16), put(0, 16), put(0, 32)  # signal_eotf and params
            put(10, 5), put(0, 2), put(0, 2), put(0, 2)  # bit depth, color space, chroma, range
            put(62, 12), put(3696, 12), put(42, 10)  # source min/max PQ, diagonal
        ue(2)  # CM v2.9 extension blocks: L1 and L6
        bits += [0] * (-len(bits) % 8)  # dm_alignment_zero_bit
        ue(5), put(1, 8), put(7, 12), put(2900, 12), put(1500, 12), put(0, 4)  # L1 min/max/avg PQ
        ue(8), put(6, 8), put(1000, 16), put(1, 16), put(1000, 16), put(400, 16)  # L6
    bits += [0] * (-len(bits) % 8)
    raw = bytearray(int("".join(map(str, bits[i:i + 8])), 2) for i in range(0, len(bits), 8))
    raw += _crc32_mpeg2(raw[1:]).to_bytes(4, "big") + b"\x80"
    out = bytearray()
    for i, b in enumerate(raw):
        if i > 2 and len(out) >= 2 and out[-2] == 0 and out[-1] == 0 and b <= 3:
            out.append(3)
        out.append(b)
    return bytes(out)


def test_malformed_pivot_counts_are_catchable():
    """CRC-correct RPUs whose pivot count is out of range (one past the
    limit, and the Exp-Golomb extremes that once overflowed `count + 2`)
    must fail as a catchable filter error with src_csp=3 and be ignored with
    src_csp=1, never abort the host; valid counts 0 and 7 decode. Run in a
    child process, so a regression can't take the test run down."""
    cases = [(c, k) for c in range(3) for k in (0, 7, 8, 2**64 - 3, 2**64 - 2)]
    body = "".join(f"""
s = base.std.SetFrameProps(DolbyVisionRPU={_rpu_pivot_count(c, k)!r})
for mode in (3, 1):
    try:
        with s.vszip.Tonemap(src_csp=mode, **kw).get_frame(0) as f:
            out = 'same' if all(bytes(f[p]) == bytes(plain[p]) for p in range(3)) else 'differs'
    except vs.Error as e:
        out = 'error: ' + str(e)
    print({c}, {k}, mode, out, flush=True)
""" for c, k in cases)
    script = f"""
import sys
sys.path.insert(0, {str(os.path.dirname(os.path.abspath(__file__)))!r})
import vapoursynth as vs
from conftest import _load_vszip
core = _load_vszip()
base = core.std.BlankClip(width=64, height=64, format=vs.YUV444P16, length=1, color=[30000, 32768, 32768])
kw = dict(dynamic_peak_detection=0, gamut_mapping=0)
plain = base.vszip.Tonemap(src_csp=1, **kw).get_frame(0)
""" + body
    import subprocess
    import sys
    proc = subprocess.run([sys.executable, "-"], input=script, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    got = {}
    for line in proc.stdout.splitlines():
        c, k, mode, out = line.split(" ", 3)
        got[(int(c), int(k), int(mode))] = out
    for c, k in cases:
        valid = k <= 7
        dv, hdr = got[(c, k, 3)], got[(c, k, 1)]
        if valid:
            assert not dv.startswith("error"), (c, k, dv)
        else:
            assert "invalid `DolbyVisionRPU`" in dv, (c, k, dv)
            # no DM data in these packets, so a bad one changes nothing
            assert hdr == "same", (c, k, hdr)


def _dv(src, rpu):
    return src.std.SetFrameProps(DolbyVisionRPU=rpu)


def _pq_eotf(x):
    m1, m2, c1, c2, c3 = 2610 / 16384, 2523 / 4096 * 128, 3424 / 4096, 2413 / 4096 * 32, 2392 / 4096 * 32
    p = np.clip(x, 0, 1) ** (1 / m2)
    return (np.maximum(p - c1, 0) / (c2 - c3 * p)) ** (1 / m1)


def _pq_oetf(y):
    m1, m2, c1, c2, c3 = 2610 / 16384, 2523 / 4096 * 128, 3424 / 4096, 2413 / 4096 * 32, 2392 / 4096 * 32
    yp = np.maximum(y, 0) ** m1
    return ((c1 + c2 * yp) / (1 + c3 * yp)) ** m2


def test_dovi_decode_matches_numpy_model(make_hdr):
    """Independent float64 model of libplacebo's Dolby Vision decode for
    this profile 8.1 RPU (identity reshaping, checked by the kernel tests):
    codes / 2^16, the RPU's YCbCr matrix and offsets to PQ LMS, PQ EOTF, its
    RGB->LMS inverted through libplacebo's BT.2020 LMS->RGB, then (DoVi to
    a 10000-nit PQ target with clip gamut mapping: no tone or gamut change)
    PQ OETF and BT.2020 limited-range encode. Measured: identical codes."""
    src = _dv(make_hdr(vs.YUV444P16), RPU_P8)
    out = src.vszip.Tonemap(src_csp=3, dst_csp=1, dynamic_peak_detection=0, gamut_mapping=0)
    _, _, _, ycc, off, lms = _rpu_fields(RPU_P8)
    lms2rgb = np.array([[3.06441879, -2.16597676, 0.10155818],
                        [-0.65612108, 1.78554118, -0.12943749],
                        [0.01736321, -0.04725154, 1.03004253]]) @ lms
    with src.get_frame(0) as f:
        v = np.stack([np.asarray(f[p], dtype=np.float64) for p in range(3)])
    s = np.minimum(v / 65536, 1023 / 1024)  # clamped to the pivot range
    lin = np.einsum("ij,jhw->ihw", lms2rgb, _pq_eotf(np.einsum("ij,jhw->ihw", ycc, s - off[:, None, None])))
    rgb = _pq_oetf(np.maximum(lin, 0))
    y = 0.2627 * rgb[0] + 0.6780 * rgb[1] + 0.0593 * rgb[2]
    want = np.stack([4096 + 56064 * y, 32768 + 57344 * (rgb[2] - y) / 1.8814, 32768 + 57344 * (rgb[0] - y) / 1.4746])
    with out.get_frame(0) as f:
        got = np.stack([np.asarray(f[p], dtype=np.float64) for p in range(3)])
    d = np.abs(got - np.clip(np.round(want), 4096, [[[60160]], [[61440]], [[61440]]]))
    assert d.mean() <= 1.5, d.mean()
    assert d.max() <= 48, d.max()


@pytest.mark.parametrize("detect", [0, 1])
def test_dovi_mixed_rpu_chain_any_request_order(make_hdr, detect):
    """Frames whose RPU is only partial (use_prev_vdr_rpu_flag, no DM data,
    compressed DM data) continue the last complete one before them, in frame
    order, so concurrent out-of-order requests (with peak detection, whose
    window frames are resolved too) give the sequential result; and a chain
    after anchor A differs from the same chain after anchor B."""
    base = make_hdr(vs.YUV420P16)
    # use_prev RPUs carry DM data in practice (without it they are shorter
    # than the 25 bytes libdovi accepts); their matrices are ignored, as in
    # vs-placebo's create_dovi_meta
    seq = [dict(dm=_DM_A), dict(use_prev=True, dm=_DM_B), dict(), dict(compressed=True),
           dict(use_prev=True, compressed=True), dict(dm=_DM_B), dict(), dict(use_prev=True, dm=_DM_A),
           dict(compressed=True), dict(dm=_DM_A), dict(use_prev=True, compressed=True), dict()]
    clip = vs.core.std.Splice([_dv(base, _rpu_build(**f)) for f in seq])
    kw = dict(src_csp=3, dynamic_peak_detection=detect, smoothing_period=2.0)
    n = clip.num_frames
    want = _frame_arrays(clip.vszip.Tonemap(**kw), range(n))
    got = _frame_arrays(clip.vszip.Tonemap(**kw), list(reversed(range(n))) * 2, threads=8)
    for k in range(n):
        for p in range(3):
            assert np.array_equal(want[k][p], got[k][p]), (k, p)
    # the partial frames after B (6..8) decode with B's matrices, not A's
    alt = vs.core.std.Splice([_dv(base, _rpu_build(**f)) for f in seq[:5] + [dict(dm=_DM_A)] + seq[6:]])
    alt_out = _frame_arrays(alt.vszip.Tonemap(**kw), [6])
    assert any(not np.array_equal(want[6][p], alt_out[6][p]) for p in range(3))
    # a use_prev RPU's own matrices are ignored: frame 1 (carrying B's, after
    # A) decodes exactly as if it carried A's (same DM metadata otherwise)
    same = vs.core.std.Splice([_dv(base, _rpu_build(**f)) for f in [seq[0], dict(use_prev=True, dm=_DM_A)] + seq[2:]])
    same_out = _frame_arrays(same.vszip.Tonemap(**kw), [1])
    for p in range(3):
        assert np.array_equal(want[1][p], same_out[1][p]), p


def test_dovi_rpu_chain_cap():
    """A partial RPU resolves up to 300 frames after its complete one; one
    frame further is a catchable error, not a hang."""
    blank = vs.core.std.BlankClip(width=64, height=64, format=vs.YUV420P16, length=1, color=[30000, 32768, 32768])
    clip = _dv(blank, _rpu_build(dm=_DM_A)) + _dv(blank, _rpu_build()) * 301
    out = clip.vszip.Tonemap(src_csp=3, dynamic_peak_detection=0)
    out.get_frame(300)
    with pytest.raises(vs.Error, match="no complete DolbyVisionRPU within 300 frames"):
        out.get_frame(301)


def _fel_pair(make_hdr, rpu, el_fmt=vs.YUV420P10):
    """A 640x320 BL and a half-size EL (made from the picture's own
    luma/chroma, so its codes vary around mid-scale), with the RPU on the
    EL clip, where vs-placebo reads it when an EL is given."""
    bl = make_hdr(vs.YUV420P16)
    el = bl.resize.Bicubic(bl.width // 2, bl.height // 2, format=el_fmt)
    return bl, _dv(el, rpu)


def test_dovi_fel_composes(make_hdr):
    """A FEL RPU composes the EL (and is read from the EL clip: the BL has
    none); a MEL RPU signals no residual, so the EL changes nothing."""
    kw = dict(src_csp=3, dynamic_peak_detection=0)
    bl, el = _fel_pair(make_hdr, RPU_FEL)
    fel = bl.vszip.Tonemap(dovi_el=el, **kw)
    base = _dv(bl, RPU_FEL).vszip.Tonemap(**kw)
    assert max_abs_diff_all_planes(fel, base) > 100
    bl, el = _fel_pair(make_hdr, RPU_MEL)
    assert_same_clip(bl.vszip.Tonemap(dovi_el=el, **kw), _dv(bl, RPU_MEL).vszip.Tonemap(**kw))


def test_dovi_fel_with_detection_any_request_order(make_hdr):
    """With detection, the window frames are measured with their own EL
    frames too (requested with them): concurrent out-of-order requests give
    the sequential result."""
    bl, el = _fel_pair(make_hdr, RPU_FEL)
    # frames that differ, so the window matters
    bl = vs.core.std.Splice([bl.std.Expr(f"x {k * 900} +") for k in range(6)])
    el = vs.core.std.Splice([el.std.Expr(f"x {k * 9} +") for k in range(6)])
    kw = dict(src_csp=3, smoothing_period=2.0)
    n = bl.num_frames
    want = _frame_arrays(bl.vszip.Tonemap(dovi_el=el, **kw), range(n))
    got = _frame_arrays(bl.vszip.Tonemap(dovi_el=el, **kw), list(reversed(range(n))) * 2, threads=8)
    for k in range(n):
        for p in range(3):
            assert np.array_equal(want[k][p], got[k][p]), (k, p)


@pytest.mark.parametrize("detect", [0, 1])
def test_dovi_el_rpu_chain_any_request_order(make_hdr, detect):
    """With an EL, the RPU chain is the EL clip's (the BL has none): FEL, MEL
    and profile 8 anchors, each followed by use_prev RPUs, which keep the
    anchor's NLQ state. Concurrent out-of-order requests give the sequential
    result; without detection, exactly the frames after a FEL anchor depend
    on the EL's samples."""
    bl, el = _fel_pair(make_hdr, RPU_FEL)
    prev, p8 = _rpu_build(use_prev=True, dm=_DM_A), _rpu_build(dm=_DM_A)
    rpus = [RPU_FEL, prev, prev, RPU_MEL, prev, RPU_FEL, prev, p8, prev, RPU_FEL, prev, prev]
    fel = [1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1]
    n = len(rpus)
    bl = vs.core.std.Splice([bl.std.Expr(f"x {k * 900} +") for k in range(n)])
    el = vs.core.std.Splice([_dv(el.std.Expr(f"x {k * 9} +"), r) for k, r in enumerate(rpus)])
    kw = dict(src_csp=3, dynamic_peak_detection=detect, smoothing_period=2.0)
    want = _frame_arrays(bl.vszip.Tonemap(dovi_el=el, **kw), range(n))
    got = _frame_arrays(bl.vszip.Tonemap(dovi_el=el, **kw), list(reversed(range(n))) * 2, threads=8)
    for k in range(n):
        for p in range(3):
            assert np.array_equal(want[k][p], got[k][p]), (k, p)
    if detect:
        return
    # the same RPUs on other EL samples
    other = _frame_arrays(bl.vszip.Tonemap(dovi_el=el.std.Expr("1023 x -"), **kw), range(n))
    for k in range(n):
        same = all(np.array_equal(want[k][p], other[k][p]) for p in range(3))
        assert same != bool(fel[k]), k


def test_dovi_el_source_error_is_catchable():
    """An EL frame that fails, needed by a later frame's use_prev walk (and,
    with detection, by every window that reaches it), is a catchable error
    for each such request, never an abort or a hang; a frame that needs
    neither still decodes. Run in a child process."""
    script = f"""
import sys
sys.path.insert(0, {str(os.path.dirname(os.path.abspath(__file__)))!r})
import vapoursynth as vs
from conftest import _load_vszip
core = _load_vszip()
bl = core.std.BlankClip(width=128, height=64, format=vs.YUV420P16, length=4, color=[30000, 32768, 32768])
el = core.std.BlankClip(width=64, height=32, format=vs.YUV420P10, length=1, color=[512, 500, 520])
rpus = [{RPU_FEL!r}, {_rpu_build(use_prev=True, dm=_DM_A)!r}, {_rpu_build(use_prev=True, dm=_DM_A)!r}, {RPU_FEL!r}]
el = core.std.Splice([el.std.SetFrameProps(DolbyVisionRPU=r) for r in rpus])

def sel(n):
    if n == 0:
        raise ValueError("EL frame unavailable")
    return el

bad = el.std.FrameEval(sel)
for detect in (0, 1):
    out = bl.vszip.Tonemap(dovi_el=bad, src_csp=3, dynamic_peak_detection=detect, smoothing_period=2.0)
    for n in (2, 2, 0, 1, 3):
        try:
            out.get_frame(n)
            res = 'ok'
        except vs.Error as e:
            res = 'error' if 'EL frame unavailable' in str(e) else 'other: ' + str(e)
        print(detect, n, res, flush=True)
"""
    import subprocess
    import sys
    proc = subprocess.run([sys.executable, "-"], input=script, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    got = [tuple(line.split(" ", 2)) for line in proc.stdout.splitlines()]
    want = [(d, n, "ok" if (d, n) == ("0", "3") else "error") for d in "01" for n in "22013"]
    assert got == want, got


@pytest.mark.parametrize("kw", [dict(), dict(contrast_recovery=0.3), dict(dynamic_peak_detection=0, contrast_recovery=0.3)], ids=str)
def test_cache_frame_keeps_the_output(make_hdr, kw):
    """cache_frame (default 1) keeps a frame's linear RGB from the detection
    pass (or contrast recovery's feature pass) for the output pass; 0
    decodes it again. Identical output either way: HDR10 and Dolby Vision
    with a FEL, sequential against concurrent out-of-order requests (a
    request reuses only its own frame's measurement)."""
    bl, el = _fel_pair(make_hdr, RPU_FEL)
    n = 6
    bl = vs.core.std.Splice([bl.std.Expr(f"x {k * 900} +") for k in range(n)])
    el = vs.core.std.Splice([el.std.Expr(f"x {k * 9} +") for k in range(n)])
    for src in (dict(), dict(src_csp=3, dovi_el=el)):
        args = dict(smoothing_period=2.0, **src, **kw)
        want = _frame_arrays(bl.vszip.Tonemap(cache_frame=0, **args), range(n))
        got = _frame_arrays(bl.vszip.Tonemap(**args), list(reversed(range(n))) * 2, threads=8)
        for k in range(n):
            for p in range(3):
                assert np.array_equal(want[k][p], got[k][p]), (src.keys(), k, p)
    with pytest.raises(vs.Error):
        bl.vszip.Tonemap(cache_frame=2)


def test_dovi_fel_native_10bit_matches_16bit(make_hdr):
    """The EL code scale is 2^(EL depth - bits): a 10-bit EL clip and its
    codes shifted to 16 bit compose identically."""
    kw = dict(src_csp=3, dynamic_peak_detection=0)
    bl, el = _fel_pair(make_hdr, RPU_FEL)
    assert_same_clip(bl.vszip.Tonemap(dovi_el=el, **kw), bl.vszip.Tonemap(dovi_el=el.resize.Point(format=vs.YUV420P16), **kw))


@pytest.mark.parametrize("kw", [dict(dst_csp=1), dict(dst_csp=0), dict(dst_csp=0, gamut_mapping=0)], ids=str)
def test_dovi_fel_matches_vs_placebo(core, make_hdr, kw):
    """FEL composition parity on 4:4:4 BL (the EL itself is 4:2:0). The GPU
    stores each EL plane's vertical lanczos pass in half float, round toward
    zero: at these 16-bit EL samples that is up to half a 10-bit code, which
    flips the EL code's rounding on many pixels of this synthetic EL (its
    codes are spread over the whole range, unlike a real FEL around its
    offset), each flip a full deadzone step; vszip keeps f32. Measured:
    mean 4.3 / 14.4 / 12.3, p99.9 29 / 162 / 317; with that store emulated
    in a test build all three are within 10. Without the EL vszip is
    600-1300 off (p99.9 3800-12300), so the bounds still catch a missing or
    misplaced EL."""
    pl = _placebo(core)
    bl, el = _fel_pair(make_hdr, RPU_FEL, vs.YUV420P16)
    bl = bl.resize.Point(format=vs.YUV444P16)
    kw = dict(src_csp=3, dynamic_peak_detection=0, **kw)
    with bl.vszip.Tonemap(dovi_el=el, **kw).get_frame(0) as fa, pl.Tonemap(bl, dovi_el=el, **kw).get_frame(0) as fb:
        d = np.abs(np.stack([np.asarray(fa[p]) for p in range(3)]).astype(np.int64)
                   - np.stack([np.asarray(fb[p]) for p in range(3)]).astype(np.int64))
    assert d.mean() <= 25, d.mean()
    assert np.percentile(d, 99.9) <= 600, np.percentile(d, 99.9)


_EL_SIZES = {"same": (1, 1), "half_w": (2, 1), "half_h": (1, 2), "half": (2, 2)}
_EL_FORMATS = {"444": vs.YUV444P16, "422": vs.YUV422P16, "440": vs.YUV440P16, "420": vs.YUV420P16}


@pytest.mark.parametrize("fmt", list(_EL_FORMATS))
@pytest.mark.parametrize("size", list(_EL_SIZES))
def test_dovi_fel_geometry_matches_vs_placebo(core, size, fmt):
    """Every accepted EL layer size and subsampling, against vs-placebo, on a
    flat BL (so the output follows the EL alone) and an EL pattern whose
    neighbours differ by odd codes. A plane enlarged on one axis only is
    read bilinearly on the other (pl_shader_sample_ortho2), which matters
    where the top-left chroma shift starts that axis half a texel in (4:2:2
    and 4:4:0 chroma): nearest sampling there was 73-105 off (max 864).
    Measured: mean <= 5, max 17 (one deadzone step: exact half-code ties
    the GPU's float noise breaks either way, and its half-float store
    between two lanczos passes)."""
    pl = _placebo(core)
    w, h = 132, 68
    fw, fh = _EL_SIZES[size]
    bl = core.std.BlankClip(width=w, height=h, format=vs.YUV444P16, length=1, color=[24000, 32768, 32768])
    bl = bl.std.SetFrameProps(_Transfer=16, _Primaries=9, _Matrix=9, _Range=0)

    def fill(n, f):
        out = f.copy()
        for p in range(3):
            a = np.asarray(out[p])
            y, x = np.indices(a.shape)
            a[:] = (512 + (7 * x + 11 * y + p * 13) % 97 - 48) * 64
        return out

    el = core.std.BlankClip(width=w // fw, height=h // fh, format=_EL_FORMATS[fmt], length=1)
    el = _dv(el.std.ModifyFrame(el, fill), RPU_FEL)
    kw = dict(src_csp=3, dst_csp=1, dynamic_peak_detection=0, gamut_mapping=0)
    with bl.vszip.Tonemap(dovi_el=el, **kw).get_frame(0) as fa, pl.Tonemap(bl, dovi_el=el, **kw).get_frame(0) as fb:
        d = np.abs(np.stack([np.asarray(fa[p]) for p in range(3)]).astype(np.int64)
                   - np.stack([np.asarray(fb[p]) for p in range(3)]).astype(np.int64))
    assert d.mean() <= 8, d.mean()
    assert d.max() <= 34, d.max()


@pytest.mark.parametrize("kw", [dict(dst_csp=1), dict(dst_csp=0)], ids=str)
def test_dovi_fel_real_matches_vs_placebo(core, kw):
    """The LG profile 4 demo with its FEL (extracted by
    bench/tonemap/dovi_el_demux.py) when present: 4:4:4 BL, center crop
    (BL by multiples of 8, EL by half, keeping the 4x chroma aligned)."""
    pl = _placebo(core)
    if not all(os.path.exists(p) for p in LG_DEMOS) or not hasattr(core, "ffms2"):
        pytest.skip("LG Dolby Vision demo and its EL not available")
    bl0, el0 = core.ffms2.Source(LG_DEMOS[0]), core.ffms2.Source(LG_DEMOS[1])
    frames = [bl0.num_frames * k // 6 for k in (1, 3, 5)]
    bl = core.std.Splice([bl0[k] for k in frames]).resize.Point(format=vs.YUV444P16)
    bl = bl.std.Crop(left=1440, right=1440, top=808, bottom=808)
    el = core.std.Splice([el0[k] for k in frames]).resize.Point(format=vs.YUV420P16)
    el = el.std.Crop(left=720, right=720, top=404, bottom=404)
    kw = dict(src_csp=3, dynamic_peak_detection=0, **kw)
    a, b = bl.vszip.Tonemap(dovi_el=el, **kw), pl.Tonemap(bl, dovi_el=el, **kw)
    d = []
    for n in range(bl.num_frames):
        with a.get_frame(n) as fa, b.get_frame(n) as fb:
            d.append(np.abs(np.stack([np.asarray(fa[p]) for p in range(3)]).astype(np.int64)
                            - np.stack([np.asarray(fb[p]) for p in range(3)]).astype(np.int64)))
    d = np.stack(d)
    # measured on the full frames: -> PQ mean 1.5, p99.9 22; -> SDR mean
    # 6.5, p99.9 76 (the perceptual gamut LUT on the FEL's saturated detail)
    assert d.mean() <= 10, d.mean()
    assert np.percentile(d, 99.9) <= 150, np.percentile(d, 99.9)


def test_dovi_profile5_is_full_range_and_differs(make_hdr):
    """Profile 5 (IPTPQc2, full range): a different decode, full-range YUV out."""
    src = make_hdr(vs.YUV420P16)
    p5 = _dv(src, RPU_P5).vszip.Tonemap(src_csp=3, dynamic_peak_detection=0)
    p8 = _dv(src, RPU_P8).vszip.Tonemap(src_csp=3, dynamic_peak_detection=0)
    assert props(p5)["_Range"] == 1
    assert props(p8)["_Range"] == 0
    assert max_abs_diff_all_planes(p5, p8) > 1000


def test_rpu_metadata_applies_to_hdr10_too(make_hdr):
    """As in vs-placebo, the RPU's source range and L1/L6 metadata replace
    the props whenever DolbyVisionRPU is present, whatever src_csp is; a
    bad RPU is ignored there (it only errors with src_csp=3)."""
    src = make_hdr(vs.YUV420P16)
    kw = dict(dynamic_peak_detection=0)
    plain = src.vszip.Tonemap(**kw)
    assert max_abs_diff_all_planes(_dv(src, RPU_P8).vszip.Tonemap(**kw), plain) > 0
    bad = bytearray(RPU_P8)
    bad[60] ^= 0x10
    assert_same_clip(_dv(src, bytes(bad)).vszip.Tonemap(**kw), plain)
    with pytest.raises(vs.Error, match="invalid `DolbyVisionRPU`"):
        _dv(src, bytes(bad)).vszip.Tonemap(src_csp=3, **kw).get_frame(0)
    with pytest.raises(vs.Error, match="missing `DolbyVisionRPU`"):
        src.vszip.Tonemap(src_csp=3, **kw).get_frame(0)


def test_dovi_rpu_chain_follows_frame_order(make_hdr):
    """An RPU without DM data keeps the previous frame's matrices; frame n
    folds the RPUs back to the last complete one, in frame order, whatever
    order frames are requested in, and fetches those frames itself."""
    base = make_hdr(vs.YUV420P16)
    partial = _rpu_strip_dm(RPU_P8)
    kw = dict(dynamic_peak_detection=0)
    # frames: complete P8, then two partial ones; and the same after a
    # complete P5 (whose matrices differ)
    after_p8 = _dv(base, RPU_P8) + _dv(base, partial) * 2
    after_p5 = _dv(base, RPU_P5) + _dv(base, partial) * 2
    seq = _frame_arrays(after_p8.vszip.Tonemap(**kw, src_csp=3), [0, 1, 2])
    # frame 2 first, in a fresh filter: it walks back to frame 0 itself
    alone = _frame_arrays(after_p8.vszip.Tonemap(**kw, src_csp=3), [2, 1, 0])
    for k in range(3):
        for p in range(3):
            assert np.array_equal(seq[k][p], alone[k][p]), (k, p)
    # the partial frames take the matrices of the complete frame before them
    assert max_abs_diff_all_planes(after_p5.vszip.Tonemap(**kw, src_csp=3), after_p8.vszip.Tonemap(**kw, src_csp=3), n=2) > 1000
    assert_same_clip(after_p8.vszip.Tonemap(**kw, src_csp=3)[1], after_p8.vszip.Tonemap(**kw, src_csp=3)[2])
    # with no complete RPU before it, a chain starts from the zeroed state
    # (vs-placebo's static on its first call): no reshaping, zero matrices
    lone = _dv(base, partial).vszip.Tonemap(**kw, src_csp=3)
    with lone.get_frame(0) as f:
        assert np.isfinite(np.asarray(f[0], dtype=np.float64)).all()


@pytest.mark.parametrize(
    "kw",
    [
        dict(dynamic_peak_detection=0),
        dict(dynamic_peak_detection=0, dst_csp=1),
        dict(dynamic_peak_detection=0, gamut_mapping=2, tone_mapping_function=3),
    ],
    ids=str,
)
@pytest.mark.parametrize("rpu", [RPU_P8, RPU_P5], ids=["p8", "p5"])
def test_dovi_matches_vs_placebo(core, make_hdr, rpu, kw):
    """Parity of the Dolby Vision decode on 4:4:4 input (on 4:2:0 the GPU
    truncates the vertically upsampled chroma to half float, round toward
    zero, which the RPU's chroma mapping amplifies; see tonemap-port.md).
    Detection is left out for the half-float FBO reason given in
    test_matches_vs_placebo."""
    pl = _placebo(core)
    src = _dv(make_hdr(vs.YUV444P16), rpu)
    try:
        ref = pl.Tonemap(src, src_csp=3, **kw)
        ref.get_frame(0)
    except vs.Error as e:
        pytest.skip(f"vs-placebo unusable here: {e}")
    out = src.vszip.Tonemap(src_csp=3, **kw)
    with out.get_frame(0) as fa, ref.get_frame(0) as fb:
        a = np.stack([np.asarray(fa[p]) for p in range(3)]).astype(np.int64)
        b = np.stack([np.asarray(fb[p]) for p in range(3)]).astype(np.int64)
    d = np.abs(a - b)
    # the saturation gamut map with reinhard measured 8.8 on this image
    assert d.mean() <= 10, d.mean()
    assert np.percentile(d, 99.9) <= 400, np.percentile(d, 99.9)


DV_CLIPS = {
    # profile 8.1, identity reshaping
    "p81": (DV_SAMPLE, [2000, 9000, 33000]),
    # Dolby's Sol Levante test content: profile 5 (IPTPQc2, full range,
    # 9-pivot luma polynomials) and 8.4 (HLG base, chroma MMR)
    "p5": (r"D:\dovi\contents2\BL_RPU_dvhe-05_1920x1080@24fps_0_6313.mp4", [631, 3157, 5682]),
    "p84": (r"D:\dovi\contents2\BL_RPU_dvhe-08-84_1920x1080@24fps_0_6313.mp4", [631, 3157, 5682]),
}


@pytest.mark.parametrize("clip", list(DV_CLIPS))
@pytest.mark.parametrize("fmt", [vs.YUV444P16, vs.YUV420P16])
def test_dovi_sample_matches_vs_placebo(core, fmt, clip):
    """Real Dolby Vision clips on this machine, when present, through ffms2.
    On the Sol Levante MP4s ffms2 5.2.1 gives 1 frame in 48 the previous
    frame's RPU (BestSource R22 is exact, R18 dropped the prop there); both
    plugins get the same props, so the parity is unaffected."""
    pl = _placebo(core)
    path, frames = DV_CLIPS[clip]
    if not os.path.exists(path) or not hasattr(core, "ffms2"):
        pytest.skip("Dolby Vision sample or ffms2 not available")
    src = core.ffms2.Source(path).resize.Point(format=fmt)
    # center 960x544
    x, y = (src.width - 960) // 4 * 2, (src.height - 544) // 4 * 2
    src = src.std.Crop(left=x, right=src.width - 960 - x, top=y, bottom=src.height - 544 - y)
    src = core.std.Splice([src[n] for n in frames])
    kw = dict(src_csp=3, dynamic_peak_detection=0)
    a, b = src.vszip.Tonemap(**kw), pl.Tonemap(src, **kw)
    d = []
    for n in range(src.num_frames):
        with a.get_frame(n) as fa, b.get_frame(n) as fb:
            d.append(np.abs(np.stack([np.asarray(fa[p]) for p in range(3)]).astype(np.int64)
                            - np.stack([np.asarray(fb[p]) for p in range(3)]).astype(np.int64)))
    d = np.stack(d)
    # measured, worst clip: 4:4:4 mean 2.3, p99.9 24; 4:2:0 mean 19, p99.9
    # 171: the GPU's round-toward-zero half-float store of the vertically
    # upsampled chroma (emulating it gave mean 1.9; see tonemap-port.md)
    limit = (4, 40) if fmt == vs.YUV444P16 else (25, 220)
    assert d.mean() <= limit[0], d.mean()
    assert np.percentile(d, 99.9) <= limit[1], np.percentile(d, 99.9)


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
        (vs.YUV420P8, {}, "16-bit integer or 32-bit float"),
        (vs.GRAY16, {}, "16-bit integer or 32-bit float"),
        (vs.YUV420PH, {}, "16-bit integer or 32-bit float"),
        (vs.YUV410P16, {}, "16-bit integer or 32-bit float"),
        (vs.YUV420P16, dict(src_csp=0), "src_csp"),
        (vs.YUV420P16, dict(src_csp=3), "missing `DolbyVisionRPU`"),
        (vs.YUV420PS, dict(src_csp=3), "integer YUV"),
        (vs.RGB48, dict(src_csp=3), "integer YUV"),
        (vs.YUV420P16, dict(dst_csp=3), "Dolby Vision output"),
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


def test_dovi_el_arguments(core):
    """dovi_el must be integer YUV of the BL's size or half of it; like
    vs-placebo, it is ignored without use_dovi."""
    bl = core.std.BlankClip(width=128, height=96, format=vs.YUV420P16, length=1)
    for el in (core.std.BlankClip(width=96, height=64, format=vs.YUV420P10, length=1),
               core.std.BlankClip(width=64, height=48, format=vs.YUV420PS, length=1),
               core.std.BlankClip(width=64, height=48, format=vs.RGB48, length=1)):
        with pytest.raises(vs.Error, match="dovi_el must be"):
            bl.vszip.Tonemap(src_csp=3, dovi_el=el)
    odd = core.std.BlankClip(width=96, height=64, format=vs.YUV420P10, length=1)
    bl.vszip.Tonemap(use_dovi=0, dovi_el=odd).get_frame(0)
    bl.vszip.Tonemap(use_dovi=1).get_frame(0)


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
