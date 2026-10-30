"""Contract and oracles for Butteraugli (libjxl master's butteraugli_main).

The always-run oracle is `_oracle`, a float64 NumPy model written from
libjxl's lib/jxl/butteraugli/butteraugli.cc (ButteraugliComparator path: one
2x-subsampled level, score = diffmap maximum, 3-norm from
lib/extras/metrics.cc). It shares no code with the Zig port. Against
libjxl's own butteraugli_main, measured outside the suite on 48 real frames
(up to 1080p, JPEG/blur/rescale/noise, 8 and 16 bit), vszip agrees to one
float32 ulp in the distance and ~1e-6 per diffmap pixel.
"""

import subprocess
import sys
from pathlib import Path

import pytest
import vapoursynth as vs

from golden import Case
from helpers import props

np = pytest.importorskip("numpy")

TESTS = Path(__file__).resolve().parent


def score(a: vs.VideoNode, b: vs.VideoNode, **kw) -> tuple[float, float]:
    p = props(vs.core.vszip.Butteraugli(a, b, **kw))
    return p["Butteraugli"], p["Butteraugli3Norm"]


def rgb24(clip):
    with clip.get_frame(0) as f:
        return np.stack([np.asarray(f[p]) for p in range(3)], -1).copy()


# --- float64 oracle ---------------------------------------------------------

WMUL = [400.0, 1.50815703118, 0, 2150.0, 10.6195433239, 16.2176043152, 29.2353797994, 0.844626970982, 0.703646627719]
GLOBAL_SCALE = 1.0 / (17.83 * 0.79079917404)


def _kernel(sigma):
    diff = max(1, int(2.25 * abs(sigma)))
    i = np.arange(-diff, diff + 1, dtype=np.float64)
    return np.exp(-i * i / (2.0 * sigma * sigma))


def _blur1d(p, k, axis):
    """libjxl ConvolutionWithTranspose along `axis`: taps clipped at the
    edges and renormalized by the sum of the taps used."""
    o = len(k) // 2
    p = np.moveaxis(p, axis, 1)
    h, n = p.shape
    out = np.zeros_like(p)
    for x in range(n):
        lo, hi = max(0, x - o), min(n - 1, x + o)
        kk = k[lo - x + o:hi - x + o + 1]
        out[:, x] = p[:, lo:hi + 1] @ kk / kk.sum()
    return np.moveaxis(out, 1, axis)


def _blur(p, sigma):
    k = _kernel(sigma)
    if len(k) == 5:  # Separable5: normalized taps, mirrored edges (-1 -> 0)
        k = k / k.sum()
        z = np.pad(p, 2, mode="symmetric")
        t = sum(k[j] * z[:, j:j + p.shape[1]] for j in range(5))
        return sum(k[j] * t[j:j + p.shape[0], :] for j in range(5))
    return _blur1d(_blur1d(p, k, 1), k, 0)


MIX = np.array([[0.29956550340058319, 0.63373087833825936, 0.077705617820981968, 1.7557483643287353],
                [0.22158691104574774, 0.69391388044116142, 0.0987313588422, 1.7557483643287353],
                [0.02, 0.02, 0.20480129041026129, 12.226454707163354]])


def _opsin(rgb, intensity):
    blurred = [_blur(c, 1.2) for c in rgb]
    out = []
    pre = [MIX[i, 0] * blurred[0] * intensity + MIX[i, 1] * blurred[1] * intensity + MIX[i, 2] * blurred[2] * intensity + MIX[i, 3]
           for i in range(3)]
    cur = [MIX[i, 0] * rgb[0] * intensity + MIX[i, 1] * rgb[1] * intensity + MIX[i, 2] * rgb[2] * intensity + MIX[i, 3]
           for i in range(3)]
    for i in range(3):
        pm = np.maximum(np.maximum(pre[i], MIX[i, 3]), 1e-4)
        gamma = 19.245013259874995 * np.log(np.maximum(pm, 0) + 9.9710635769299145) - 23.16046239805755
        sens = np.maximum(gamma / pm, 1e-4)
        out.append(cur[i] * sens)
    c0 = np.maximum(out[0], 1.7557483643287353)
    c1 = np.maximum(out[1], 1.7557483643287353)
    c2 = np.maximum(out[2], 12.226454707163354)
    return [c0 - c1, c0 + c1, c2]


def _remove(x, w):
    return np.where(x > w, x - w, np.where(x < -w, x + w, 0.0))


def _amplify(x, w):
    return np.where(x > w, x + w, np.where(x < -w, x - w, x + x))


def _maxclamp(v, m):
    k = 0.724216145665
    return np.where(v < -m, (v + m) * k - m, np.where(v >= m, (v - m) * k + m, v))


def _psycho(xyb):
    lf = [_blur(c, 7.15593339443) for c in xyb]
    mf = [a - b for a, b in zip(xyb, lf)]
    lf = [lf[0] * 33.832837186260, lf[1] * 14.458268100570, (lf[2] - 0.362267051518 * lf[1]) * 49.87984651440]
    hf = []
    for i in range(2):
        b = _blur(mf[i], 3.22489901262)
        hf.append(mf[i] - b)
        mf[i] = _remove(b, 0.29) if i == 0 else _amplify(b, 0.1)
    mf[2] = _blur(mf[2], 3.22489901262)
    hf[0] = hf[0] * ((46.0 / (hf[1] * hf[1] + 46.0)) * (1 - 0.653020556257) + 0.653020556257)
    uhf = []
    for i in range(2):
        b = _blur(hf[i], 1.56416327805)
        if i == 0:
            uhf.append(_remove(hf[i] - b, 0.04))
            hf[i] = _remove(b, 1.5)
        else:
            c = _maxclamp(b, 28.4691806922)
            uhf.append(_maxclamp(hf[i] - c, 5.19175294647) * 2.69313763794)
            hf[i] = _amplify(c * 2.155, 0.132)
    return lf, mf, hf, uhf


_LF_LINES = [
    [(0, -4), (0, -2), (0, 0), (0, 2), (0, 4)], [(-4, 0), (-2, 0), (0, 0), (2, 0), (4, 0)],
    [(-3, -3), (-2, -2), (0, 0), (2, 2), (3, 3)], [(-3, 3), (-2, 2), (0, 0), (2, -2), (3, -3)],
    [(-4, 1), (-2, 1), (0, 0), (2, -1), (4, -1)], [(-4, -1), (-2, -1), (0, 0), (2, 1), (4, 1)],
    [(-1, -4), (-1, -2), (0, 0), (1, 2), (1, 4)], [(1, -4), (1, -2), (0, 0), (-1, 2), (-1, 4)],
    [(-3, -2), (-2, -1), (0, 0), (2, 1), (3, 2)], [(-3, 2), (-2, 1), (0, 0), (2, -1), (3, -2)],
    [(-2, -3), (-1, -2), (0, 0), (1, 2), (2, 3)], [(-2, 3), (-1, 2), (0, 0), (1, -2), (2, -3)],
    [(2, -4), (1, -2), (0, 0), (-1, 2), (-2, 4)], [(-2, -4), (-1, -2), (0, 0), (1, 2), (2, 4)],
    [(-4, -2), (-2, -1), (0, 0), (2, 1), (4, 2)], [(-4, 2), (-2, 1), (0, 0), (2, -1), (4, -2)],
]
_H9 = [
    [(0, x) for x in range(-4, 5)], [(y, 0) for y in range(-4, 5)],
    [(-4, 1), (-3, 1), (-2, 1), (-1, 0), (0, 0), (1, 0), (2, -1), (3, -1), (4, -1)],
    [(-4, -1), (-3, -1), (-2, -1), (-1, 0), (0, 0), (1, 0), (2, 1), (3, 1), (4, 1)],
    [(-1, -4), (-1, -3), (-1, -2), (0, -1), (0, 0), (0, 1), (1, 2), (1, 3), (1, 4)],
    [(1, -4), (1, -3), (1, -2), (0, -1), (0, 0), (0, 1), (-1, 2), (-1, 3), (-1, 4)],
]
_H7 = [
    [(i, i) for i in range(-3, 4)], [(i, -i) for i in range(-3, 4)],
    [(-3, -2), (-2, -1), (-1, -1), (0, 0), (1, 1), (2, 1), (3, 2)],
    [(-3, 2), (-2, 1), (-1, 1), (0, 0), (1, -1), (2, -1), (3, -2)],
    [(-2, -3), (-1, -2), (-1, -1), (0, 0), (1, 1), (1, 2), (2, 3)],
    [(-2, 3), (-1, 2), (-1, 1), (0, 0), (1, -1), (1, -2), (2, -3)],
]
_HF_LINES = [_H9[0], _H9[1], _H7[0], _H7[1], _H9[2], _H9[3], _H9[4], _H9[5],
             _H7[2], _H7[3], _H7[4], _H7[5], _H9[5], _H9[4], _H9[3], _H9[2]]


def _malta(lf, l0, l1, w_gt, w_lt, norm1):
    mulli = 0.611612573796 if lf else 0.39905817637
    w_pre_gt = mulli * np.sqrt(0.5 * w_gt) / 8.5
    w_pre_lt = mulli * np.sqrt(0.33 * w_lt) / 8.5
    absval = 0.5 * (np.abs(l0) + np.abs(l1))
    d = (w_pre_gt * norm1) / (norm1 + absval) * (l0 - l1)
    s2 = (w_pre_lt * norm1) / (norm1 + absval)
    small, big = 0.55 * np.abs(l0), 1.05 * np.abs(l0)
    neg = np.where(l1 > -small, -s2 * (l1 + small), np.where(l1 < -big, s2 * (-l1 - big), 0.0))
    pos = np.where(l1 < small, s2 * (small - l1), np.where(l1 > big, -s2 * (l1 - big), 0.0))
    d = d + np.where(l0 < 0, neg, pos)
    h, w = d.shape
    z = np.pad(d, 4)
    out = np.zeros_like(d)
    for line in (_LF_LINES if lf else _HF_LINES):
        s = sum(z[4 + dy:4 + dy + h, 4 + dx:4 + dx + w] for dy, dx in line)
        out += s * s
    return out


def _l2_asym(a, b, w_gt, w_lt):
    d = a - b
    small, big = 0.4 * np.abs(a), np.abs(a)
    neg = np.where(b > -small, b + small, np.where(b < -big, -b - big, 0.0))
    pos = np.where(b < small, small - b, np.where(b > big, b - big, 0.0))
    v = np.where(a < 0, neg, pos)
    return 0.8 * w_gt * d * d + 0.8 * w_lt * v * v


def _fuzzy_erosion(p):
    h, w = p.shape
    big = np.full((h + 6, w + 6), np.inf)
    big[3:3 + h, 3:3 + w] = p
    cand = [p, 2 * p, 2 * p] + [big[3 + dy:3 + dy + h, 3 + dx:3 + dx + w]
                                for dy in (-3, 0, 3) for dx in (-3, 0, 3) if (dy, dx) != (0, 0)]
    m = np.sort(np.stack(cand), axis=0)
    return 0.45 * m[0] + 0.3 * m[1] + 0.25 * m[2]


def _mask(hf, uhf):
    x = (uhf[0] + hf[0]) * 2.5
    y = uhf[1] * 0.4 + hf[1] * 0.4
    m = np.sqrt(x * x + y * y)
    mul, bias = 6.19424080439, 6.19424080439 * 12.61050594197
    return np.sqrt(mul * np.abs(m) + bias) - np.sqrt(bias)


def _mask_y(v, offset, scaler, mul):
    r = GLOBAL_SCALE * (1.0 + mul / (scaler * v + offset))
    return r * r


def _diffmap(rgb0, rgb1, intensity, hfa, xmul):
    lf0, mf0, hf0, uhf0 = _psycho(_opsin(rgb0, intensity))
    lf1, mf1, hf1, uhf1 = _psycho(_opsin(rgb1, intensity))
    ac = [np.zeros_like(rgb0[0]) for _ in range(3)]
    sq = np.sqrt(hfa)
    ac[1] += _malta(False, uhf0[1], uhf1[1], 1.10039032555 * hfa, 1.10039032555 / hfa, 71.7800275169)
    ac[0] += _malta(False, uhf0[0], uhf1[0], 173.5 * hfa, 173.5 / hfa, 5.0)
    ac[1] += _malta(True, hf0[1], hf1[1], 18.7237414387 * sq, 18.7237414387 / sq, 4498534.45232)
    ac[0] += _malta(True, hf0[0], hf1[0], 6923.99476109 * sq, 6923.99476109 / sq, 8051.15833247)
    ac[1] += _malta(True, mf0[1], mf1[1], 37.0819870399, 37.0819870399, 130262059.556)
    ac[0] += _malta(True, mf0[0], mf1[0], 8246.75321353, 8246.75321353, 1009002.70582)
    dc = []
    for c in range(3):
        if c < 2:
            ac[c] += _l2_asym(hf0[c], hf1[c], WMUL[c] * hfa, WMUL[c] / hfa)
        ac[c] += WMUL[3 + c] * (mf0[c] - mf1[c]) ** 2
        dc.append(WMUL[6 + c] * (lf0[c] - lf1[c]) ** 2)
    b0 = _blur(_mask(hf0, uhf0), 2.7)
    b1 = _blur(_mask(hf1, uhf1), 2.7)
    mask = _fuzzy_erosion(b0)
    ac[1] += 10.0 * (b0 - b1) ** 2
    m = _mask_y(mask, 0.829591754942, 0.451936922203, 2.5485944793)
    dm = _mask_y(mask, 0.20025578522, 3.87449418804, 0.505054525019)
    return np.sqrt((xmul * dc[0] + dc[1] + dc[2]) * dm + (xmul * ac[0] + ac[1] + ac[2]) * m)


def _subsample(p):
    h, w = p.shape
    out = np.zeros(((h + 1) // 2, (w + 1) // 2))
    for dy in (0, 1):
        for dx in (0, 1):
            q = p[dy::2, dx::2]
            out[:q.shape[0], :q.shape[1]] += 0.25 * q
    if w & 1:
        out[:, -1] *= 2
    if h & 1:
        out[-1, :] *= 2
    return out


def _oracle(ref, dist, intensity=80.0, hfa=1.0, xmul=1.0):
    """(distance, 3-norm) of two HxWx3 uint8 sRGB arrays."""
    def lin(a):
        v = a.astype(np.float64) / 255.0
        return [c for c in np.moveaxis(np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4), -1, 0)]

    r0, r1 = lin(ref), lin(dist)
    full = _diffmap(r0, r1, intensity, hfa, xmul)
    sub = _diffmap([_subsample(c) for c in r0], [_subsample(c) for c in r1], intensity, hfa, xmul)
    h, w = full.shape
    d = full * 0.85 + 0.5 * np.repeat(np.repeat(sub, 2, 0), 2, 1)[:h, :w]
    p3 = (np.mean(d ** 3) ** (1 / 3) + np.mean(d ** 6) ** (1 / 6) + np.mean(d ** 12) ** (1 / 12)) / 3
    return d.max(), p3


# --- tests -------------------------------------------------------------------


def _distort(clip, kind):
    if kind == "blur":
        return clip.std.BoxBlur(hradius=1, vradius=1)
    if kind == "rescale":
        return clip.resize.Bicubic(clip.width // 2, clip.height // 2).resize.Bicubic(clip.width, clip.height)
    if kind == "quant":
        return clip.std.Expr("x 12 /").std.Expr("x 12 *")  # integer output rounds
    raise ValueError(kind)


@pytest.mark.parametrize("kind", ["blur", "rescale", "quant"])
@pytest.mark.parametrize("size", [(64, 64, 300, 120), (65, 71, 41, 200), (72, 80, 500, 10), (97, 66, 200, 240)],
                         ids=lambda s: f"{s[0]}x{s[1]}")
def test_matches_float64_oracle(src_rgb, size, kind):
    """Widths around the 8/16-lane vector boundaries, odd sizes (odd
    subsampling), and three distortions. The oracle is exact math; libjxl's
    own float32 result sits up to 6.3e-5 relative from it here (72x80
    quant), and vszip matches libjxl's to ~1e-7 there. 2e-4 still catches a
    1% change of hf_asymmetry (5.4e-4)."""
    w, h, left, top = size
    ref = src_rgb.std.CropAbs(width=w, height=h, left=left, top=top)
    dist = _distort(ref, kind)
    d, p3 = score(ref, dist)
    od, op3 = _oracle(rgb24(ref), rgb24(dist))
    assert d == pytest.approx(od, rel=2e-4)
    assert p3 == pytest.approx(op3, rel=2e-4)


@pytest.mark.parametrize("kw", [dict(intensity_target=250.0), dict(hf_asymmetry=0.8), dict(xmul=0.5)], ids=str)
def test_parameters_match_oracle(src_rgb, kw):
    ref = src_rgb.std.CropAbs(width=80, height=72, left=100, top=50)
    dist = _distort(ref, "blur")
    d, p3 = score(ref, dist, **kw)
    od, op3 = _oracle(rgb24(ref), rgb24(dist), kw.get("intensity_target", 80.0), kw.get("hf_asymmetry", 1.0), kw.get("xmul", 1.0))
    assert d == pytest.approx(od, rel=2e-4)
    assert p3 == pytest.approx(op3, rel=2e-4)


def test_matches_libjxl_raw_maps(src_rgb):
    """Pinned upstream raw maps (tests/fixtures/butteraugli_libjxl.npz:
    libjxl master 863caac `butteraugli_main --rawdistmap`, both inputs saved
    as sRGB PNGs). "noise1" is a low-amplitude case (integer noise in
    [-1, 1]), where the float64 oracle's exact log/sRGB decode differ from
    libjxl's float32 approximations per pixel; this checks the map itself.
    Measured: FMA builds 1.1e-6 relative per pixel with the distance
    identical, the no-FMA baseline build 3.7e-4 relative per pixel (low
    amplitude) and 5.7e-6 on the distance."""
    fx = np.load(TESTS / "fixtures" / "butteraugli_libjxl.npz")
    ref = src_rgb.std.CropAbs(width=96, height=80, left=120, top=60)
    a = rgb24(ref)
    noise = np.random.default_rng(1234).integers(-1, 2, a.shape)
    noisy = np.clip(a.astype(np.int32) + noise, 0, 255).astype(np.uint8)
    blank = vs.core.std.BlankClip(ref, keep=True)

    def put(n, f):
        fo = f.copy()
        for p in range(3):
            np.asarray(fo[p])[:] = noisy[..., p]
        return fo

    dists = {"noise1": blank.std.ModifyFrame(blank, put), "blur": ref.std.BoxBlur(hradius=1, vradius=1)}
    for name, dist in dists.items():
        want = fx[name].astype(np.float64)
        with vs.core.vszip.Butteraugli(ref, dist, distmap=2).get_frame(0) as f:
            got = np.asarray(f[0]).astype(np.float64)
            assert f.props["Butteraugli"] == pytest.approx(want.max(), rel=2e-5), name
        np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-6, err_msg=name)


def test_identical_scores_zero(src_rgb):
    assert score(src_rgb, src_rgb) == (0.0, 0.0)


def test_more_distortion_scores_higher(src_rgb):
    d1, _ = score(src_rgb, src_rgb.std.BoxBlur(hradius=1, vradius=1))
    d3, _ = score(src_rgb, src_rgb.std.BoxBlur(hradius=3, vradius=3))
    assert 0 < d1 < d3


def test_formats_agree(src_rgb):
    """RGBS of the same samples takes the f32 sRGB decode instead of the LUT:
    the same linear values, so the same score. RGB48 of v * 257 is v / 255
    up to f32 rounding of the sample scale."""
    dist = src_rgb.std.BoxBlur(hradius=1, vradius=1)
    s8 = score(src_rgb, dist)
    s32 = score(src_rgb.resize.Point(format=vs.RGBS), dist.resize.Point(format=vs.RGBS))
    s16 = score(src_rgb.resize.Point(format=vs.RGB48), dist.resize.Point(format=vs.RGB48))
    assert s32 == s8
    assert s16 == pytest.approx(s8, rel=1e-5)


def test_linear_input_skips_decode(src_rgb):
    """_Transfer=8 (linear) float input is taken as linear light."""
    dist = src_rgb.std.BoxBlur(hradius=1, vradius=1)
    lin = [c.resize.Point(format=vs.RGBS, transfer_in=13, transfer=8).std.SetFrameProps(_Transfer=8)
           for c in (src_rgb, dist)]
    assert score(*lin) == pytest.approx(score(src_rgb, dist), rel=1e-4)


def test_crop_and_stride_independent(src_rgb):
    """A crop (frame stride wider than its width) scores like the same
    content copied into a fresh clip."""
    crop = src_rgb.std.CropAbs(width=97, height=70, left=33, top=21)
    dist = crop.std.BoxBlur(hradius=1, vradius=1)

    def fresh(c):
        a = rgb24(c)
        blank = vs.core.std.BlankClip(c, keep=True)

        def mod(n, f):
            fo = f.copy()
            for p in range(3):
                np.asarray(fo[p])[:] = a[..., p]
            return fo

        return blank.std.ModifyFrame(blank, mod)

    assert score(crop, dist) == score(fresh(crop), fresh(dist))


def test_distmap_outputs(src_rgb):
    dist = src_rgb.std.BoxBlur(hradius=1, vradius=1)
    raw = vs.core.vszip.Butteraugli(src_rgb, dist, distmap=2)
    heat = vs.core.vszip.Butteraugli(src_rgb, dist, distmap=1)
    passthrough = vs.core.vszip.Butteraugli(src_rgb, dist)
    assert raw.format.id == vs.GRAYS and heat.format.id == src_rgb.format.id
    with raw.get_frame(0) as f:
        m = np.asarray(f[0])
        assert float(m.max()) == f.props["Butteraugli"]
        assert (m >= 0).all()
    with heat.get_frame(0) as f, passthrough.get_frame(0) as p, src_rgb.get_frame(0) as s:
        assert f.props["Butteraugli"] == p.props["Butteraugli"]
        for c in range(3):
            assert np.array_equal(np.asarray(p[c]), np.asarray(s[c]))
        assert np.asarray(f[0]).max() > 0


def test_distmap_color_properties():
    """Heatmap colors are what butteraugli_main saves as sRGB: tagged sRGB
    whatever the input's tags (a linear input used to pass its _Transfer=8
    on). The raw map drops the color tags; passthrough keeps the input's."""
    a = vs.core.std.BlankClip(None, 64, 64, vs.RGBS, length=1, color=[0.2, 0.2, 0.2])
    a = a.std.SetFrameProps(_Transfer=8, _Primaries=9, _Matrix=0)
    b = vs.core.std.BlankClip(a, color=[0.25, 0.2, 0.2]).std.SetFrameProps(_Transfer=8, _Primaries=9, _Matrix=0)
    heat = props(vs.core.vszip.Butteraugli(a, b, distmap=1))
    assert (heat["_Transfer"], heat["_Primaries"], heat["_Matrix"], heat["_Range"]) == (13, 1, 0, 1)
    raw = props(vs.core.vszip.Butteraugli(a, b, distmap=2))
    assert not {"_Transfer", "_Primaries", "_Matrix"} & set(raw)
    passthrough = props(vs.core.vszip.Butteraugli(a, b))
    assert (passthrough["_Transfer"], passthrough["_Primaries"]) == (8, 9)


def test_concurrent_requests_match_sequential(temporal_rgb):
    """Two separate output nodes (so the second can't be served from the
    first one's cache): one requested in order, one by 8 threads in shuffled
    order. Both scores and every raw-map sample must match."""
    import random
    from concurrent.futures import ThreadPoolExecutor

    clip = temporal_rgb
    dist = clip.std.BoxBlur(hradius=1, vradius=1)

    def grab(node, n):
        with node.get_frame(n) as f:
            return f.props["Butteraugli"], f.props["Butteraugli3Norm"], np.asarray(f[0]).copy()

    seq_node = vs.core.vszip.Butteraugli(clip, dist, distmap=2)
    par_node = vs.core.vszip.Butteraugli(clip, dist, distmap=2)
    seq = [grab(seq_node, n) for n in range(clip.num_frames)]
    order = list(range(clip.num_frames))
    random.Random(7).shuffle(order)
    with ThreadPoolExecutor(8) as ex:
        got = dict(zip(order, ex.map(lambda n: grab(par_node, n), order)))
    for n, (d, p3, m) in enumerate(seq):
        gd, gp3, gm = got[n]
        assert (gd, gp3) == (d, p3)
        assert np.array_equal(gm, m)


GOLDEN =[Case(vs.RGB24), Case(vs.RGB48), Case(vs.RGBS), Case(vs.RGB24, geometry="odd")]


@pytest.mark.parametrize("case", GOLDEN, ids=str)
def test_golden_cases(golden, make_clip, case):
    ref = make_clip(case.fmt, case.geometry)
    d, p3 = score(ref, ref.std.BoxBlur(hradius=1, vradius=1))
    golden.check_value("butteraugli", case.id, d, rel=1e-5)
    golden.check_value("butteraugli", case.id + "|3norm", p3, rel=1e-5)


@pytest.mark.parametrize(
    ("make", "msg"),
    [
        (lambda c: (c.resize.Point(format=vs.YUV444P8, matrix_s="709"),) * 2, "only RGB"),
        (lambda c: (c.std.CropAbs(width=63, height=64),) * 2, "at least 64x64"),
        (lambda c: (c, c.resize.Point(format=vs.RGB48)), "same constant format"),
        (lambda c: (c, c.std.Crop(right=2)), "same constant format"),
        (lambda c: (c, c + c), "same length"),
        (lambda c: (c.resize.Point(format=vs.RGBH),) * 2, "half-float"),
    ],
)
def test_validation_errors(src_rgb, make, msg):
    with pytest.raises(vs.Error, match=msg):
        vs.core.vszip.Butteraugli(*make(src_rgb))


@pytest.mark.parametrize("kw", [dict(distmap=3), dict(distmap=-1), dict(intensity_target=0.0), dict(hf_asymmetry=0.0), dict(xmul=-1.0)], ids=str)
def test_argument_errors(src_rgb, kw):
    with pytest.raises(vs.Error):
        vs.core.vszip.Butteraugli(src_rgb, src_rgb, **kw)


@pytest.mark.parametrize(("key", "lo", "hi"), [("intensity_target", 0.001, 10000.0), ("hf_asymmetry", 0.001, 1000.0), ("xmul", 0.0, 1000.0)])
def test_argument_range_endpoints(key, lo, hi):
    """The documented decimal endpoints are accepted (an f32-rounded bound,
    0.001f > 0.001, used to reject the minimum) and values just outside are
    not; the scores stay finite at both ends."""
    import math

    c = vs.core.std.BlankClip(None, 64, 64, vs.RGB24, length=1, color=[80, 120, 160])
    d = vs.core.std.BlankClip(c, color=[90, 120, 150])
    for v in (lo, hi):
        p = props(vs.core.vszip.Butteraugli(c, d, **{key: v}))
        assert math.isfinite(p["Butteraugli"]) and math.isfinite(p["Butteraugli3Norm"])
    for v in (math.nextafter(lo, -math.inf), math.nextafter(hi, math.inf)):
        with pytest.raises(vs.Error, match=key):
            vs.core.vszip.Butteraugli(c, d, **{key: v})


_WIDE_INT = """
import sys
sys.path.insert(0, {tests!r})
import vapoursynth as vs
from conftest import _load_vszip
core = _load_vszip()
fmt = core.query_video_format(vs.RGB, vs.INTEGER, {bits}, 0, 0)
a = core.std.BlankClip(None, 64, 64, fmt.id, length=1)
try:
    core.vszip.Butteraugli(a, a, distmap={distmap})
except vs.Error as e:
    assert "8-16 bit" in str(e), str(e)
else:
    raise AssertionError("accepted")
print("ok")
"""


@pytest.mark.parametrize("distmap", [0, 1, 2])
@pytest.mark.parametrize("bits", [17, 24, 32])
def test_wide_integer_rgb_rejected(bits, distmap):
    """Integer RGB above 16 bits (4-byte samples) is rejected at creation. It
    used to reach the u16 path: wrong scores at 17/24 bits, and debug panics
    (32-bit LUT setup, 17-bit heatmap), hence the child process."""
    script = _WIDE_INT.format(tests=str(TESTS), bits=bits, distmap=distmap)
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stdout + r.stderr


_UPSTREAM = """
import sys
sys.path.insert(0, {tests!r})
import vapoursynth as vs
from conftest import _load_vszip
core = _load_vszip()
good = core.std.BlankClip(None, 64, 64, vs.RGB24, length=2)
def fail(n, f):
    raise RuntimeError("upstream failure")
bad = good.std.ModifyFrame(good, fail)
for a, b in ((bad, good), (good, bad)):
    try:
        core.vszip.Butteraugli(a, b)
    except vs.Error as e:
        assert "frame 0 failed" in str(e) and "upstream failure" in str(e), str(e)
    else:
        raise AssertionError("no error")
print("ok")
"""


def test_frame0_upstream_error_is_reported():
    """Creation reads frame 0's _Transfer; a failing upstream frame must
    become this filter's error, not a crash (hence the child process)."""
    r = subprocess.run([sys.executable, "-c", _UPSTREAM.format(tests=str(TESTS))], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stdout + r.stderr
