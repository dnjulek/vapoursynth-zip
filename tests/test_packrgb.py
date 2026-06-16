import numpy as np
import pytest
import vapoursynth as vs

from golden import Case, sweep
from helpers import props, write_png

# PackRGB outputs a GRAY32 (32-bit-int) packed clip. std.PlaneStats only
# supports 8..16-bit int / 32-bit float, so golden_stats() can't measure
# GRAY32 directly. Reinterpret the packed frame's raw bytes as a 4x-wide
# GRAY8 stream (lossless: every packed byte is preserved and measured), which
# PlaneStats accepts. Distinct packings -> distinct byte streams -> distinct
# goldens, so this is a faithful snapshot fingerprint of the packed output.


def packed_to_bytes(core, packed: vs.VideoNode) -> vs.VideoNode:
    """View a GRAY32 packed clip as its raw little-endian byte stream, shaped
    as a 4x-wide GRAY8 clip (4 bytes per packed pixel)."""
    w, h = packed.width, packed.height
    blank = core.std.BlankClip(None, w * 4, h, vs.GRAY8, length=packed.num_frames)

    def sel(n, f):
        fout = f[1].copy()
        src = np.asarray(f[0][0]).view(np.uint8)  # H x (W*4)
        np.asarray(fout[0])[:] = src
        return fout

    return blank.std.ModifyFrame(clips=[packed, blank], selector=sel)


CASES = sweep(
    base_fmt=vs.RGB24,
    formats=[vs.RGB24, vs.RGB30],
    geometries=["odd", "tiny"],
) + [
    Case(vs.RGB30, geometry="odd"),
    Case(vs.RGB30, geometry="tiny"),
]


@pytest.mark.parametrize("case", CASES, ids=str)
def test_golden_cases(golden, core, make_clip, case):
    src = make_clip(case.fmt, case.geometry)
    out = src.vszip.PackRGB()
    assert out.format.id == vs.GRAY32
    golden.check("packrgb", case, packed_to_bytes(core, out))


@pytest.fixture(scope="module")
def rgb24(core, tmp_path_factory):
    """16x4 RGB24 clip with varied, known pixel values."""
    rows = [
        [((x * 16 + y * 3) % 256, (255 - x * 9) % 256, (x * x + y) % 256) for x in range(16)]
        for y in range(4)
    ]
    path = write_png(tmp_path_factory.mktemp("packrgb") / "grad.png", rows)
    return core.vszip.ImageRead(str(path))


def packed_value(clip: vs.VideoNode, x: int, y: int) -> int:
    with clip.get_frame(0) as f:
        return f[0][y, x]


def test_rgb24_packing(rgb24):
    out = rgb24.vszip.PackRGB()
    assert out.format.id == vs.GRAY32
    assert (out.width, out.height) == (rgb24.width, rgb24.height)
    with rgb24.get_frame(0) as f:
        for y in range(rgb24.height):
            for x in range(rgb24.width):
                r, g, b = f[0][y, x], f[1][y, x], f[2][y, x]
                expected = b | (g << 8) | (r << 16) | (255 << 24)  # BGRA bytes
                assert packed_value(out, x, y) == expected


def test_rgb30_packing(rgb24):
    src10 = rgb24.resize.Point(format=vs.RGB30)
    out = src10.vszip.PackRGB()
    assert out.format.id == vs.GRAY32
    with src10.get_frame(0) as f:
        for y in range(src10.height):
            for x in range(src10.width):
                r, g, b = f[0][y, x], f[1][y, x], f[2][y, x]
                expected = b | (g << 10) | (r << 20) | (0b11 << 30)
                assert packed_value(out, x, y) == expected


def test_props_preserved(rgb24):
    assert props(rgb24.vszip.PackRGB())["zigimg_format"] == "rgb24"


@pytest.mark.parametrize("fmt", [vs.RGB48, vs.YUV420P8, vs.GRAY8])
def test_format_error(core, fmt):
    src = core.std.BlankClip(None, 32, 16, fmt, length=1)
    with pytest.raises(vs.Error, match="only RGB24 and RGB30 inputs are supported"):
        src.vszip.PackRGB()


@pytest.mark.parametrize(
    "fmt",
    [
        # RGB family, wrong int bit depth: the accepted set is exactly
        # {RGB24 (8-bit), RGB30 (10-bit)}. Pin both boundary neighbours and the
        # interior so an extra `case` slipping into the switch can't go unnoticed.
        vs.RGB27,  # 9-bit (just above RGB24)
        vs.RGB36,  # 12-bit (just above RGB30)
        vs.RGB42,  # 14-bit
        vs.RGB48,  # 16-bit
        vs.RGBH,  # RGB but float16
        vs.RGBS,  # RGB but float32
        vs.YUV420P8,  # wrong color family (YUV 8-bit)
        vs.YUV444P10,  # wrong color family (YUV 10-bit)
        vs.GRAY8,  # wrong color family (gray)
        vs.GRAYS,  # wrong color family (gray float)
    ],
)
def test_validation_errors(core, fmt):
    """Create rejects every non-RGB24/RGB30 input format."""
    src = core.std.BlankClip(None, 32, 16, fmt, length=1)
    with pytest.raises(vs.Error, match="only RGB24 and RGB30 inputs are supported"):
        src.vszip.PackRGB()
