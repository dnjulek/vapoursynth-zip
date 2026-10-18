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
