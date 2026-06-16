import pytest
import vapoursynth as vs

from golden import Case, sweep
from helpers import avg, props

# ColorMap accepts ONLY Gray8 and a single arg `color` (0..21, default 20).
# Output is always RGB24. Sweep every color index on GRAY8, plus geometry
# variants that exercise the SIMD tail ("odd") and scalar tail ("tiny").
CASES = (
    sweep(
        base_fmt=vs.GRAY8,
        base_args=dict(),
        # color is the only argument; each index selects a distinct LUT
        args=tuple(dict(color=c) for c in range(22)),
        geometries=["odd", "tiny"],
    )
    + [
        # hand-picked color x geometry interactions: a non-default color on the
        # tail geometries proves the LUT is applied independent of frame shape
        Case(vs.GRAY8, geometry="odd", args=dict(color=0)),
        Case(vs.GRAY8, geometry="tiny", args=dict(color=13)),
    ]
)


@pytest.mark.parametrize("case", CASES, ids=str)
def test_golden_cases(golden, make_clip, case):
    src = make_clip(case.fmt, case.geometry)
    golden.check("colormap", case, src.vszip.ColorMap(**case.args))

# Average of the colormapped output converted back to gray, one per map type
# (carried over from the old .vpy suite).
GOLDENS = [
    0.5453371821384804,
    0.4657149969362745,
    0.6759822495404412,
    0.4097088694852941,
    0.6413369715073529,
    0.2589842026654412,
    0.6345918734681373,
    0.5770620021446078,
    0.5281143918504903,
    0.548726619944853,
    0.6384453699448529,
    0.4189677351409314,
    0.5736758003982844,
    0.36308752680759804,
    0.3589300130208333,
    0.3995364200367647,
    0.46880407475490193,
    0.47296392463235293,
    0.29686014093137253,
    0.6083539560355392,
    0.6932635952818628,
    0.38608273973651963,
]


@pytest.mark.parametrize("color", range(22))
def test_golden(to_gray, color):
    out = to_gray(vs.GRAY8).vszip.ColorMap(color).resize.Bilinear(format=vs.GRAY8, matrix=1)
    assert avg(out) == pytest.approx(GOLDENS[color], rel=1e-6)


def test_output_format(to_gray):
    src = to_gray(vs.GRAY8)
    out = src.vszip.ColorMap()
    assert out.format.id == vs.RGB24
    assert (out.width, out.height) == (src.width, src.height)
    assert props(out)["_ColorRange"] == vs.Range.RANGE_FULL


def rgb_at(clip: vs.VideoNode, x: int = 0, y: int = 0) -> tuple[int, int, int]:
    with clip.get_frame(0) as f:
        return f[0][y, x], f[1][y, x], f[2][y, x]


@pytest.mark.parametrize(
    ("color", "value", "expected"),
    [
        (20, 0, (48, 18, 59)),  # turbo starts at dark blue
        (20, 255, (122, 4, 3)),  # turbo ends at dark red
        (0, 0, (255, 0, 0)),  # autumn: red -> yellow
        (0, 255, (255, 255, 0)),
    ],
)
def test_lut_endpoints(core, color, value, expected):
    src = core.std.BlankClip(None, 16, 16, vs.GRAY8, length=1, color=value)
    assert rgb_at(src.vszip.ColorMap(color)) == expected


def test_format_error(to_gray):
    with pytest.raises(vs.Error, match="only Gray8 format is supported"):
        to_gray(vs.GRAY16).vszip.ColorMap()


@pytest.mark.parametrize("color", [-100, -1, 22, 100])
def test_color_range_error(to_gray, color):
    with pytest.raises(vs.Error, match='"color" should be between 0 and 21'):
        to_gray(vs.GRAY8).vszip.ColorMap(color)


@pytest.mark.parametrize("fmt", [vs.GRAY16, vs.GRAYS, vs.YUV420P8, vs.RGB24])
def test_non_gray8_rejected(make_clip, fmt):
    with pytest.raises(vs.Error, match="only Gray8 format is supported"):
        make_clip(fmt).vszip.ColorMap()
