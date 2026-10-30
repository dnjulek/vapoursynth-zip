import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import vapoursynth as vs

from golden import Case, sweep
from helpers import props


@pytest.fixture(scope="module")
def src16(src_rgb):
    return src_rgb.resize.Bicubic(format=vs.YUV420P16, matrix=1)


def score(a: vs.VideoNode, b: vs.VideoNode, n: int = 0) -> float:
    return props(vs.core.vszip.SSIMULACRA2(a, b), n)["SSIMULACRA2"]


def distort(clip: vs.VideoNode, kind: str) -> vs.VideoNode:
    """Build a distorted copy of `clip`. Each kind degrades a different way so
    the resulting SSIMULACRA2 scores are mutually distinguishable."""
    if kind == "resize":  # bicubic 2x up then back down (ringing/blur)
        return clip.resize.Bicubic(clip.width * 2, clip.height * 2).resize.Bicubic(clip.width, clip.height)
    if kind == "blur1":
        return clip.std.BoxBlur(hradius=1, vradius=1)
    if kind == "blur3":
        return clip.std.BoxBlur(hradius=3, vradius=3)
    raise ValueError(f"unknown distortion {kind!r}")


# SSIMULACRA2 has no scalar args; the swept `dist` kwarg selects how the
# distorted clip is built (see distort()). The Create callback accepts any
# resize-convertible format: RGB (8-16 bit int via LUT, f32) is read natively,
# YUV (matrix) and GRAY (matrix gray->RGB) go through toRGBS first, and the
# sRGB decode always runs in the plugin. Sweep target formats spanning each
# path and the three distortions; GRAY is a distinct conversion path, so
# include GRAY8/GRAY16 below.
CASES = (
    sweep(
        base_fmt=vs.YUV420P16,
        base_args=dict(dist="blur1"),
        formats=[vs.YUV420P8, vs.YUV420P16, vs.RGB24, vs.RGBS, vs.GRAY8, vs.GRAY16],
        args=[dict(dist="resize"), dict(dist="blur1"), dict(dist="blur3")],
        geometries=["odd", "tiny"],
    )
    + [
        # hand-picked format x distortion interactions
        Case(vs.RGBS, args=dict(dist="resize")),
        Case(vs.RGB24, args=dict(dist="blur3")),
        Case(vs.YUV420P8, args=dict(dist="resize")),
        Case(vs.YUV420P16, args=dict(dist="blur3")),
        # GRAY color-family conversion path (gray->RGB via matrix)
        Case(vs.GRAY16, args=dict(dist="resize")),
        Case(vs.GRAY8, args=dict(dist="blur3")),
    ]
)


@pytest.mark.parametrize("case", CASES, ids=str)
def test_golden_cases(golden, make_clip, case):
    ref = make_clip(case.fmt, case.geometry)
    dist = distort(ref, case.args["dist"])
    s = props(vs.core.vszip.SSIMULACRA2(ref, dist))["SSIMULACRA2"]
    golden.check_value("ssimulacra2", case.id, s, rel=1e-3)


def test_identical_constant_clip(core):
    src = core.std.BlankClip(None, 64, 64, vs.YUV420P16, length=1, color=[30000, 20000, 40000])
    assert score(src, src) == 100.0


def test_identical_real_image(src16):
    assert score(src16, src16) > 99.9


def test_golden(src16):
    dist = src16.resize.Bicubic(src16.width * 2, src16.height * 2).resize.Bicubic(src16.width, src16.height)
    assert score(src16, dist) == pytest.approx(69.6831769659849, rel=1e-3)


def test_more_distortion_scores_lower(src16):
    blur1 = src16.std.BoxBlur(hradius=1, vradius=1)
    blur3 = src16.std.BoxBlur(hradius=3, vradius=3)
    s_ident, s1, s3 = score(src16, src16), score(src16, blur1), score(src16, blur3)
    assert s_ident > s1 > s3


def _rgb_pair(src_rgb, fmt):
    ref = src_rgb.resize.Point(format=fmt)
    return ref, src_rgb.std.BoxBlur(hradius=2, vradius=2).resize.Point(format=fmt)


def test_yuv_matches_manual_rgbs(src16):
    """YUV is converted to RGB inside the filter (zimg's steps: range, Bicubic
    chroma upsampling, matrix; values within ~1e-7 of resize), then the
    in-plugin sRGB decode: the score of converting to (still sRGB-encoded)
    RGBS by hand. Not bit-exact: 1e-7 of input noise alone moves a score by
    ~1e-4."""
    dist = src16.resize.Bicubic(src16.width * 2, src16.height * 2).resize.Bicubic(src16.width, src16.height)
    manual = score(src16.resize.Bicubic(format=vs.RGBS), dist.resize.Bicubic(format=vs.RGBS))
    assert score(src16, dist) == pytest.approx(manual, abs=2e-3)


@pytest.mark.parametrize("fmt", [vs.YUV420P8, vs.YUV422P10, vs.YUV444PS, vs.YUV440P8])
@pytest.mark.parametrize("loc", [None, 1, 2, 5])
@pytest.mark.parametrize("matrix", [None, 6, 9])
def test_yuv_conversion_matches_resize(src_rgb, fmt, loc, matrix):
    """The in-filter YUV -> RGB follows resize's frame props (_Matrix over the
    height rule, _ChromaLocation, _Range): scoring a YUV clip against its own
    resize.Bicubic RGBS conversion lands at the ~1e-7 noise floor (a 1e-7
    perturbation of an RGBS clip scores ~99.5)."""
    yuv = src_rgb.resize.Bicubic(format=fmt, matrix=1)
    props = {}
    if loc is not None:
        props["_ChromaLocation"] = loc
    if matrix is not None:
        props["_Matrix"] = matrix
    if props:
        yuv = yuv.std.SetFrameProps(**props)
    default = 1 if yuv.height > 650 else 6
    rgb = yuv.resize.Bicubic(format=vs.RGBS, matrix_in=default)
    assert score(yuv, rgb) > 99.4


@pytest.mark.parametrize("fmt", [vs.YUV420P16, vs.YUV422P10], ids=lambda f: vs.PresetVideoFormat(f).name)
@pytest.mark.parametrize("width", [64, 66, 68, 70])
def test_chroma_upsampling_right_edge(src_rgb, fmt, width):
    """The horizontal chroma upsampler's vector interior must stop at the
    chroma width. Bounded by the padded scratch row instead, it read unwritten
    samples past the right edge when the chroma width is 2 or 3 above a
    multiple of the vector length: chroma widths 34 and 35 (68, 70) hit that
    for 4, 8 and 16 lanes. Scored against its own resize conversion, such a
    clip got 66-96; a correct one scores above 99.4, with or without FMA."""
    src = src_rgb.resize.Bicubic(format=fmt, matrix=1).std.CropAbs(width=width, height=128, left=200, top=100)
    assert score(src, src.resize.Bicubic(format=vs.RGBS)) > 99.2


# --- float64 oracle -------------------------------------------------------
# libjxl's tools/ssimulacra2.cc in float64: exact sRGB decode, exact cube
# root, separate blur(a^2)/blur(b^2) variances, scales stopped below 8 px. Its
# blur, CreateRecursiveGaussian(1.5) run with zero padding, is computed here
# from the recursion itself (_jxl_kernel), independently of vszip's taps.


def _jxl_kernel():
    """Impulse response of libjxl's recursive Gaussian (sigma 1.5): a
    symmetric 9-tap kernel, returned as its 9 taps."""
    import math

    import numpy as np

    sigma = 1.5
    radius = round(3.2795 * sigma + 0.2546)
    om = [math.pi / (2 * radius) * m for m in (1, 3, 5)]
    p = [1 / math.tan(0.5 * om[0]), -1 / math.tan(0.5 * om[1]), 1 / math.tan(0.5 * om[2])]
    r = [p[0] ** 2 / math.sin(om[0]), -p[1] ** 2 / math.sin(om[1]), p[2] ** 2 / math.sin(om[2])]
    rho = [math.exp(-0.5 * sigma * sigma * w * w) / radius for w in om]
    d13, d35, d51 = p[0] * r[1] - r[0] * p[1], p[1] * r[2] - r[1] * p[2], p[2] * r[0] - r[2] * p[0]
    z15, z35 = d35 / d13, d51 / d13
    a = np.array([p, r, [z15, z35, 1.0]])
    beta = np.linalg.solve(a, [1.0, radius * radius - sigma * sigma, z15 * rho[0] + z35 * rho[1] + rho[2]])
    n2 = [-beta[i] * math.cos(om[i] * (radius + 1)) for i in range(3)]
    d1 = [-2 * math.cos(om[i]) for i in range(3)]
    # FastGaussian1D on an impulse at 20 of a zero signal of 41
    x = np.zeros(41)
    x[20] = 1.0
    out, prev, prev2 = np.zeros(41), [0.0] * 3, [0.0] * 3
    for n in range(-radius + 1, 41):
        lo, hi = n - radius - 1, n + radius - 1
        s = (x[lo] if 0 <= lo < 41 else 0.0) + (x[hi] if 0 <= hi < 41 else 0.0)
        o = [n2[k] * s - d1[k] * prev[k] - prev2[k] for k in range(3)]
        prev2, prev = prev, o
        if n >= 0:
            out[n] = sum(o)
    assert np.abs(out[:16]).max() < 1e-12 and np.abs(out[25:]).max() < 1e-12
    return out[16:25]


_WEIGHTS_FROM_SOURCE = Path(__file__).resolve().parent.parent / "src" / "filters" / "ssimulacra2.zig"


def _oracle(ref, dist):
    """Score of two HxWx3 uint8 sRGB arrays."""
    import numpy as np

    k9 = _jxl_kernel()

    def blur(p):
        h, w = p.shape
        z = np.pad(p, 4)  # zero padded, as libjxl
        t = sum(k9[k] * z[k:k + h, :] for k in range(9))
        return sum(k9[k] * t[:, k:k + w] for k in range(9))

    def linear(a):
        v = a / 255.0
        return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)

    def xyb(lin):
        m = np.array([[0.30, 0.622, 0.078], [0.23, 0.692, 0.078], [0.24342269, 0.20476745, 1 - 0.24342269 - 0.20476745]])
        bias = 0.0037930734
        c = np.cbrt(np.maximum(lin @ m.T + bias, 0)) - np.cbrt(bias)
        y = 0.5 * (c[..., 0] + c[..., 1])
        return [0.5 * (c[..., 0] - c[..., 1]) * 14 + 0.42, y + 0.01, (c[..., 2] - y) + 0.55]

    def down(lin):
        h, w, _ = lin.shape
        a = lin[np.minimum(np.arange(2 * ((h + 1) // 2)), h - 1)][:, np.minimum(np.arange(2 * ((w + 1) // 2)), w - 1)]
        return (a[0::2, 0::2] + a[0::2, 1::2] + a[1::2, 0::2] + a[1::2, 1::2]) / 4

    src = _WEIGHTS_FROM_SOURCE.read_text(encoding="latin-1")
    body = src[src.index("const weight = [108]f64{"):]
    weight = [float(t) for t in body[body.index("{") + 1:body.index("};")].split(",") if t.strip()]

    la, lb = linear(ref.astype(np.float64)), linear(dist.astype(np.float64))
    total, i = 0.0, 0
    per = []  # [scale][plane] -> 6 norms
    for s in range(6):
        if s and (la.shape[0] < 8 or la.shape[1] < 8):
            break
        if s:
            la, lb = down(la), down(lb)
        xa, xb = xyb(la), xyb(lb)
        row = []
        for p, q in zip(xa, xb):
            mu1, mu2 = blur(p), blur(q)
            num = (1 - (mu1 - mu2) ** 2) * (2 * (blur(p * q) - mu1 * mu2) + 0.0009)
            den = (blur(p * p) - mu1 ** 2) + (blur(q * q) - mu2 ** 2) + 0.0009
            d = np.maximum(1 - num / den, 0)
            e = (1 + np.abs(q - mu2)) / (1 + np.abs(p - mu1)) - 1
            art, det = np.maximum(e, 0), np.maximum(-e, 0)
            row.append([d.mean(), (d ** 4).mean() ** 0.25, art.mean(), (art ** 4).mean() ** 0.25, det.mean(), (det ** 4).mean() ** 0.25])
        per.append(row)
    # libjxl's Msssim::Score: the weight index runs on over the computed scales
    # only, so a clip with fewer than 6 takes shifted weights for planes 1, 2
    for c in range(3):
        for s in range(len(per)):
            for n in range(2):
                for m in (0, 2, 4):
                    total += weight[i] * per[s][c][m + n]
                    i += 1
    x = total * 0.9562382616834844
    x = 2.326765642916932 * x - 0.020884521182843837 * x ** 2 + 6.248496625763138e-05 * x ** 3
    return 100 - 10 * x ** 0.6276336467831387 if x > 0 else 100.0


def _rgb24(clip):
    import numpy as np

    with clip.get_frame(0) as f:
        return np.stack([np.asarray(f[p]) for p in range(3)], -1).copy()


@pytest.mark.parametrize("dist", ["blur1", "resize", "quant"])
@pytest.mark.parametrize("size", [(96, 64, 300, 120), (160, 96, 40, 200), (122, 70, 500, 10)], ids=lambda s: f"{s[0]}x{s[1]}")
def test_matches_float64_oracle(src_rgb, size, dist):
    """libjxl's algorithm in float64. The former blur((a+b)^2) denominator
    was up to 6.6 points off (p95 0.26) over 528 real-frame pairs, and the
    former sampled-Gaussian kernel with mirrored edges several points; libjxl's
    own float32 math is up to ~1.8 off. vszip now lands within ~1e-3."""
    pytest.importorskip("numpy")
    w, h, left, top = size
    ref = src_rgb.std.CropAbs(width=w, height=h, left=left, top=top)
    if dist == "quant":
        d = ref.std.Expr("x 12 /").std.Expr("x 12 *")  # integer output rounds
    else:
        d = distort(ref, dist)
    a, b = _rgb24(ref), _rgb24(d)
    assert score(ref, d) == pytest.approx(_oracle(a, b), abs=5e-3)


def test_flat_pair_matches_float64_oracle(core):
    """Flat 254 vs 253 white: the exact SSIM term is (mu1 - mu2)^2 ~ 9e-6 per
    pixel (more at the zero-padded borders). f32 cancellation in the
    variances used to swamp it (7 points low); libjxl's own f32 math rounds it
    away."""
    np = pytest.importorskip("numpy")
    a = core.std.BlankClip(None, 200, 120, vs.RGB24, length=1, color=[254, 254, 254])
    b = core.std.BlankClip(None, 200, 120, vs.RGB24, length=1, color=[253, 253, 253])
    expected = _oracle(np.full((120, 200, 3), 254, np.uint8), np.full((120, 200, 3), 253, np.uint8))
    assert score(a, b) == pytest.approx(expected, abs=0.1)


_UPSTREAM_SCRIPT = """
import sys
sys.path.insert(0, {tests!r})
import vapoursynth as vs
from conftest import _load_vszip
core = _load_vszip()
good = core.std.BlankClip(None, 64, 64, {fmt}, length=2)
def fail(n, f):
    raise RuntimeError("upstream failure")
bad = good.std.ModifyFrame(good, fail)
for a, b in ((bad, good), (good, bad)):
    try:
        core.vszip.SSIMULACRA2(a, b)
    except vs.Error as e:
        assert "frame 0 failed" in str(e) and "upstream failure" in str(e), str(e)
    else:
        raise AssertionError("no error")
print("ok")
"""


@pytest.mark.parametrize("fmt", ["vs.RGB24", "vs.YUV420P8", "vs.GRAY8"])
def test_frame0_upstream_error_is_reported(fmt):
    """Creation reads frame 0 for its props (_Transfer, and _Matrix for YUV).
    If that frame fails upstream, the call must raise; it used to dereference
    the null frame and take the process down, hence the child process."""
    script = _UPSTREAM_SCRIPT.format(tests=str(Path(__file__).resolve().parent), fmt=fmt)
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stdout + r.stderr


_NARROW_SCRIPT = """
import sys
sys.path.insert(0, {tests!r})
import vapoursynth as vs
from conftest import _load_vszip
core = _load_vszip()
for fmt, w in {cases!r}:
    src = core.std.BlankClip(None, w, 64, getattr(vs, fmt), length=1, color=[100, 90, 160])
    with core.vszip.SSIMULACRA2(src, src).get_frame(0) as f:
        assert f.props["SSIMULACRA2"] == 100.0, (fmt, w, f.props["SSIMULACRA2"])
print("ok")
"""


def test_narrow_subsampled_widths():
    """Rows narrower than the upsampler's first interior pair (6 output
    columns) go through the tap table alone; they used to index past the
    table and abort the process, hence the child process."""
    cases = [("YUV420P8", 2), ("YUV420P8", 4), ("YUV422P8", 2), ("YUV422P8", 4), ("YUV422P8", 6), ("YUV411P8", 4)]
    script = _NARROW_SCRIPT.format(tests=str(Path(__file__).resolve().parent), cases=cases)
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stdout + r.stderr


@pytest.mark.parametrize("fmt", [vs.RGB24, vs.RGB30, vs.RGB48])
def test_native_int_rgb_matches_float_path(src_rgb, fmt):
    """Integer RGB is decoded through a sample -> linear LUT built from the
    float decode of v * (1 / peak) -- the exact f32 values zimg produces for
    the int -> RGBS conversion, so both paths give the identical score."""
    ref, dist = _rgb_pair(src_rgb, fmt)
    assert score(ref, dist) == score(ref.resize.Point(format=vs.RGBS), dist.resize.Point(format=vs.RGBS))


def test_out_of_range_float_is_clamped(src_rgb):
    """Encoded float values outside [0, 1] (YUV -> RGB overshoot) clamp
    before the decode, as libjxl's PNG input would, instead of taking its odd
    extension / out-of-fit polynomial."""
    ref, dist = _rgb_pair(src_rgb, vs.RGBS)
    ref, dist = ref.std.Expr("x 2 * 0.4 -"), dist.std.Expr("x 2 * 0.4 -")
    clamp = "x 0 max 1 min"
    # the input really leaves [0, 1] at both ends, so equality pins both bounds
    st = ref.std.PlaneStats().get_frame(0).props
    assert st["PlaneStatsMin"] < 0 and st["PlaneStatsMax"] > 1
    assert score(ref, dist) == score(ref.std.Expr(clamp), dist.std.Expr(clamp))


def test_linear_tagged_input_skips_decode(src_rgb):
    """_Transfer=8 (linear) input is used as is: zimg-linearized input lands
    within the zimg-vs-libjxl decode difference of the internal path, while
    decoding it a second time would be far off."""
    ref, dist = _rgb_pair(src_rgb, vs.RGBS)
    lin = [c.std.SetFrameProps(_Transfer=13).resize.Point(transfer=8) for c in (ref, dist)]
    internal = score(ref, dist)
    assert score(*lin) == pytest.approx(internal, abs=0.02)
    redecoded = [c.std.SetFrameProps(_Transfer=13) for c in lin]
    assert abs(score(*redecoded) - internal) > 0.5


def test_output_passes_input_through(src16, src_rgb):
    """Only the frame prop matters: the output is the reference frame -- RGB
    and YUV (converted inside the filter) as is, other families (GRAY) as the
    toRGBS clip."""
    rgb = src_rgb.resize.Point(format=vs.RGB24)
    assert vs.core.vszip.SSIMULACRA2(rgb, rgb).format.id == vs.RGB24
    out = vs.core.vszip.SSIMULACRA2(src16, src16)
    assert out.format.id == vs.YUV420P16
    assert (out.width, out.height, out.num_frames) == (src16.width, src16.height, src16.num_frames)
    gray = src16.std.ShufflePlanes(0, vs.GRAY)
    assert vs.core.vszip.SSIMULACRA2(gray, gray).format.id == vs.RGBS


def test_concurrent_requests_match_sequential(temporal_rgb):
    """Guard the filter's fmParallel contract with real concurrent requests."""
    ref = temporal_rgb.resize.Bicubic(format=vs.YUV420P16, matrix=1)
    distorted = ref.std.BoxBlur(hradius=2, vradius=2)
    sequential = vs.core.vszip.SSIMULACRA2(ref, distorted)
    concurrent = vs.core.vszip.SSIMULACRA2(ref, distorted)
    expected = [props(sequential, n)["SSIMULACRA2"] for n in range(ref.num_frames)]

    def get_score(n):
        return props(concurrent, n)["SSIMULACRA2"]

    with ThreadPoolExecutor(max_workers=ref.num_frames) as pool:
        actual = list(pool.map(get_score, reversed(range(ref.num_frames))))
    assert actual == pytest.approx(list(reversed(expected)), rel=1e-12)


def test_dimension_error(core):
    a = core.std.BlankClip(None, 64, 32, vs.YUV420P16, length=1)
    with pytest.raises(vs.Error, match="clips must have the same dimensions"):
        core.vszip.SSIMULACRA2(a, a.std.Crop(left=8))


def test_length_error(core):
    a = core.std.BlankClip(None, 64, 32, vs.YUV420P16, length=5)
    with pytest.raises(vs.Error, match="clips must have the same length"):
        core.vszip.SSIMULACRA2(a, a.std.Trim(0, 2))


# Every validation branch in ssimulacraCreate: mismatched dimensions and
# mismatched length are the only two checks (the filter accepts any
# resize-convertible format/bit-depth).
@pytest.mark.parametrize(
    ("distorted", "msg"),
    [
        (lambda a: a.std.Crop(left=8), "clips must have the same dimensions"),
        (lambda a: a.std.AddBorders(top=2), "clips must have the same dimensions"),
        (lambda a: a.std.Trim(0, 2), "clips must have the same length"),
    ],
)
def test_validation_errors(core, distorted, msg):
    a = core.std.BlankClip(None, 64, 32, vs.YUV420P16, length=5)
    with pytest.raises(vs.Error, match=msg):
        core.vszip.SSIMULACRA2(a, distorted(a))
