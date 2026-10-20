"""Small independent models and deterministic data shared by both host suites.

These models specify visible pixels, not SIMD instructions. No goldens or
plugin output is used to generate expected values.
"""

import numpy as np
import vapoursynth as vs


WIDTHS = [1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65]
GRAY_FORMATS = [("Y8", vs.GRAY8, 8), ("Y10", vs.GRAY10, 10),
                ("Y12", vs.GRAY12, 12), ("Y14", vs.GRAY14, 14),
                ("Y16", vs.GRAY16, 16), ("Y32", vs.GRAYS, 32)]
RGB_CASES = [(name, fmt, bits, gray, m)
             for name, fmt, bits, gray in [
                 ("YV24", vs.YUV444P8, 8, False),
                 ("YUV444P10", vs.YUV444P10, 10, False),
                 ("YUV444P16", vs.YUV444P16, 16, False),
                 ("YUV444PS", vs.YUV444PS, 32, False),
                 ("Y8", vs.GRAY8, 8, True), ("Y32", vs.GRAYS, 32, True)]
             for m in ([2] if gray else [1, 2, 4, 5, 6, 7, 9])]


def pattern(bits, width, height=3, plane=0, frame=0):
    y, x = np.mgrid[:height, :width]
    codes = (x * 37 + y * 71 + plane * 53 + frame * 29) % 257
    if bits == 32:
        return (codes / 256).astype(np.float32)
    peak = (1 << bits) - 1
    return np.rint(codes * peak / 256).astype(np.uint8 if bits == 8 else np.uint16)


def source(core, frames, fmt):
    """Fresh frames filled with exact samples, including out-of-peak carriers."""
    height, width = frames[0][0].shape
    blank = core.std.BlankClip(width=width, height=height, format=fmt,
                               length=len(frames), keep=True)

    def fill(n, f):
        out = f.copy()
        for p, data in enumerate(frames[n]):
            np.asarray(out[p])[:] = data
        return out

    return core.std.ModifyFrame(blank, blank, fill)


def snapshot(node, n=0):
    with node.get_frame(n) as f:
        return ([np.array(f[p], copy=True) for p in range(f.format.num_planes)],
                {k: f.props[k] for k in f.props if k != "_ColorRange"})


def as_list(value):
    return list(value) if isinstance(value, (list, tuple)) else [value]


def average_model(src, ref, bits, exclude):
    keep = ~np.isin(src, exclude)
    scale = 1.0 if bits == 32 else float((1 << bits) - 1)
    avg = float(np.mean(src[keep], dtype=np.float64)) / scale if keep.any() else 0.0
    delta = np.abs(src - ref) if bits == 32 else np.abs(src.astype(np.int64) - ref.astype(np.int64))
    return avg, float(np.mean(delta, dtype=np.float64)) / scale


def quantile_model(src, bits, lower, upper):
    if lower == 0 and upper == 0:
        return (float(src.min()), float(src.max())) if bits == 32 else (int(src.min()), int(src.max()))
    if bits == 32:
        values = np.floor(np.clip(src, 0, 1).astype(np.float64) * 65535 + 0.5)
        peak = 65535
    else:
        values, peak = src, (1 << bits) - 1
    ordered = np.sort(values, axis=None)
    lo, hi = int(len(ordered) * lower), int(len(ordered) * upper)
    result = (ordered[lo] if lo < len(ordered) else peak,
              ordered[-1-hi] if hi < len(ordered) else 0)
    return tuple(float(x) / peak for x in result) if bits == 32 else tuple(int(x) for x in result)


def rgb_model(planes, bits, matrix=6, full=False, gray=False):
    """YCbCr normalization plus the matrix defined by its Kr/Kb weights."""
    if bits == 32:
        normalized = [p.astype(np.float64) for p in planes[:3]]
    else:
        peak, shift = (1 << bits) - 1, 1 << (bits - 8)
        normalized = []
        for p, data in enumerate(planes[:3]):
            lo = (0 if p == 0 else 1 << (bits - 1)) if full else ((16 if p == 0 else 128) * shift)
            span = peak if full else (219 if p == 0 else 224) * shift
            normalized.append((data.astype(np.float64) - lo) / span)
    if gray:
        return [normalized[0]] * 3
    kr, kb = {1: (0.2126, 0.0722), 4: (0.30, 0.11), 5: (0.299, 0.114),
              6: (0.299, 0.114), 7: (0.212, 0.087), 9: (0.2627, 0.0593)}[matrix]
    y, cb, cr = normalized
    r, b = y + 2 * (1-kr) * cr, y + 2 * (1-kb) * cb
    g = (y - kr * r - kb * b) / (1-kr-kb)
    return [r, g, b]


class VSHost:
    def __init__(self, core):
        self.core = core

    def source(self, frames, name, fmt):
        return source(self.core, frames, fmt)

    def apply(self, name, clip, **kwargs):
        return getattr(self.core.vszip, name)(clip, **kwargs)

    def frame(self, clip, n=0):
        return snapshot(clip, n)

    def tag(self, clip, **kwargs):
        return self.core.std.SetFrameProps(clip, **kwargs)

    def sequence(self, clip, tags):
        return self.core.std.Splice([self.tag(clip[n], **t) if t else clip[n]
                                     for n, t in enumerate(tags)])


def check_statistics(host, name, fmt, bits, width):
    a, b = pattern(bits, width), pattern(bits, width, frame=2)
    src, ref = host.source([[a]], name, fmt), host.source([[b]], name, fmt)
    src = host.tag(src, AuditMarker=731, auditAvg=999.0, auditDiff=999.0,
                   auditMin=999, auditMax=999)
    excludes = [0, 1, 0] if bits == 32 else [0, (1 << bits) // 2, 0]
    got = host.apply("PlaneAverage", src, exclude=excludes, clipb=ref, prop="audit")
    planes, props = host.frame(got)
    expected = average_model(a, b, bits, excludes)
    assert as_list(props["auditAvg"]) == pytest_approx([expected[0]], 2e-12)
    assert as_list(props["auditDiff"]) == pytest_approx([expected[1]], 2e-12)
    np.testing.assert_array_equal(planes[0], a)
    assert props["AuditMarker"] == 731
    for lower, upper in [(0.0, 0.0), (0.25, 0.5), (1.0, 1.0)]:
        out = host.apply("PlaneMinMax", src, minthr=lower, maxthr=upper,
                         clipb=ref, prop="audit")
        planes, props = host.frame(out)
        want = quantile_model(a, bits, lower, upper)
        assert as_list(props["auditMin"]) == pytest_approx([want[0]], 2e-7)
        assert as_list(props["auditMax"]) == pytest_approx([want[1]], 2e-7)
        assert as_list(props["auditDiff"]) == pytest_approx([expected[1]], 2e-12)
        np.testing.assert_array_equal(planes[0], a)
    original = host.frame(src)[1]
    assert original["auditAvg"] == 999.0 and original["auditMin"] == 999


def pytest_approx(value, tolerance):
    import pytest
    return pytest.approx(value, rel=0, abs=tolerance)


def check_all_excluded(host, name, fmt, bits):
    dtype = np.float32 if bits == 32 else np.uint8 if bits == 8 else np.uint16
    a = np.zeros((3, 17), dtype=dtype)
    b = np.full_like(a, 1 if bits == 32 else (1 << bits) - 1)
    src, ref = host.source([[a]], name, fmt), host.source([[b]], name, fmt)
    _, props = host.frame(host.apply("PlaneAverage", src, exclude=[0, 0], clipb=ref))
    assert as_list(props["psmAvg"]) == [0.0]
    assert as_list(props["psmDiff"]) == [1.0], "excludes affect Avg, not Diff"


def check_limiter(host, bits, width, default):
    fmt = {8: vs.RGB24, 10: vs.RGB30, 16: vs.RGB48, 32: vs.RGBS}[bits]
    name = {8: "RGBP", 10: "RGBP10", 16: "RGBP16", 32: "RGBPS"}[bits]
    frames = [[pattern(bits, width, plane=p) for p in range(3)]]
    if bits == 10:
        for p in frames[0]:
            p[0, 0] = 65535  # wider storage may contain out-of-peak samples
    if bits == 32:
        for p in frames[0]:
            p[0, 0], p[-1, -1] = -0.25, 1.5
    src = host.source(frames, name, fmt)
    src = host.tag(src, AuditMarker=37)
    if default:
        bounds = (0.0, 1.0) if bits == 32 else (0, (1 << bits) - 1)
        kwargs = {"planes": [1]}
    else:
        bounds = (0.2, 0.7) if bits == 32 else (37, 173)
        kwargs = {"planes": [1], "min": [bounds[0]] * 3, "max": [bounds[1]] * 3}
    out, props = host.frame(host.apply("Limiter", src, **kwargs))
    for p in range(3):
        expected = np.clip(frames[0][p], *bounds) if p == 1 else frames[0][p]
        np.testing.assert_array_equal(out[p], expected)
    assert props["AuditMarker"] == 37
    before, _ = host.frame(src)
    for a, b in zip(before, frames[0]):
        np.testing.assert_array_equal(a, b)


def check_rgb_metadata(host, name, fmt, bits, matrix, gray=False, height=9):
    frames = [[pattern(bits, 17, height, plane=p, frame=n)
               for p in range(1 if gray else 3)] for n in range(3)]
    if bits == 32 and not gray:
        for frame in frames:
            frame[1] -= 0.5
            frame[2] -= 0.5
    # Fallback and explicit metadata are interleaved; input pixels vary too.
    matrices, full = [matrix, 9, 6], [False, True, False]
    tags = [{"_Matrix": m, "_Range": int(f)} for m, f in zip(matrices, full)]
    if not isinstance(host, VSHost):
        tags = [{"_Matrix": m, "_ColorRange": int(not f)} for m, f in zip(matrices, full)]
    src = host.sequence(host.source(frames, name, fmt), tags)
    out = host.apply("SSIMULACRA2", src, distorted=src)
    for n in (2, 0, 1, 2):
        actual, props = host.frame(out, n)
        effective = matrices[n]
        if effective == 2:
            effective = 1 if height > 650 else 6
        expected = rgb_model(frames[n], bits, effective, full[n], gray)
        assert len(actual) == 3
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a, b, rtol=0, atol=2e-6)
        # The existing metric contract allows float cancellation for identical
        # nonconstant inputs (test_ssimulacra2 uses >99.9); the RGB conversion
        # itself has the much tighter independent assertion above.
        assert 99.9 < props["SSIMULACRA2"] <= 100.0


def check_selected_statistics(host, bits, family):
    fmt = ({8: vs.RGB24, 16: vs.RGB48, 32: vs.RGBS} if family == "rgb" else
           {8: vs.YUV444P8, 16: vs.YUV444P16, 32: vs.YUV444PS})[bits]
    name = ({8: "RGBP", 16: "RGBP16", 32: "RGBPS"} if family == "rgb" else
            {8: "YV24", 16: "YUV444P16", 32: "YUV444PS"})[bits]
    frames = [[pattern(bits, 33, 5, plane=p) for p in range(3)]]
    if bits == 32 and family == "yuv":
        frames[0][1] -= 0.5
        frames[0][2] -= 0.5
    src = host.source(frames, name, fmt)
    out, props = host.frame(host.apply("PlaneAverage", src, exclude=[-1], planes=[2, 0]))
    expected = [average_model(frames[0][p], frames[0][p], bits, [-1])[0] for p in [0, 2]]
    assert as_list(props["psmAvg"]) == pytest_approx(expected, 2e-12)
    for a, b in zip(out, frames[0]):
        np.testing.assert_array_equal(a, b)
