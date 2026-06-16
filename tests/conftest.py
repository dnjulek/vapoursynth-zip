"""Shared fixtures for the vszip test suite.

The suite drives the freshly built plugin in-process (no vspipe). Build it
first with `zig build`, or point VSZIP_LIB at an existing library.
"""

import os
import sys
from pathlib import Path

import pytest
import vapoursynth as vs

REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE = Path(__file__).resolve().parent / "image.png"


def pytest_addoption(parser):
    parser.addoption(
        "--update-goldens",
        action="store_true",
        default=False,
        help="regenerate tests/goldens/*.json from the current build instead of comparing",
    )


def pytest_configure(config):
    from golden import GoldenStore

    config._golden_store = GoldenStore(config.getoption("--update-goldens"))


def pytest_sessionfinish(session, exitstatus):
    store = getattr(session.config, "_golden_store", None)
    if store is not None:
        store.save()


@pytest.fixture(scope="session")
def golden(request):
    return request.config._golden_store


def _plugin_path() -> Path:
    env = os.environ.get("VSZIP_LIB")
    if env:
        return Path(env)
    if sys.platform == "win32":
        return REPO_ROOT / "zig-out" / "bin" / "vszip.dll"
    suffix = "dylib" if sys.platform == "darwin" else "so"
    return REPO_ROOT / "zig-out" / "lib" / f"libvszip.{suffix}"


def _load_vszip() -> vs.Core:
    core = vs.core
    if not hasattr(core, "vszip"):
        path = _plugin_path()
        if not path.is_file():
            pytest.exit(
                f"vszip plugin not found at {path}; run `zig build` first or set VSZIP_LIB",
                returncode=2,
            )
        core.std.LoadPlugin(str(path))
    return core


@pytest.fixture(scope="session")
def core() -> vs.Core:
    return _load_vszip()


@pytest.fixture(scope="session")
def src_rgb(core: vs.Core) -> vs.VideoNode:
    """Single-frame 640x320 RGB24 crop of the test image."""
    clip = core.vszip.ImageRead(str(IMAGE))
    return clip.std.Crop(left=clip.width - 640, bottom=clip.height - 320)


@pytest.fixture(scope="session")
def to_gray(src_rgb: vs.VideoNode):
    """Factory: the source as a GRAY clip in the given format."""

    def convert(fmt: int) -> vs.VideoNode:
        return src_rgb.resize.Bilinear(format=fmt, matrix=1).std.RemoveFrameProps("_Matrix")

    return convert


@pytest.fixture(scope="session")
def to_yuv(src_rgb: vs.VideoNode):
    """Factory: the source as a YUV clip in the given format."""

    def convert(fmt: int) -> vs.VideoNode:
        return src_rgb.resize.Bilinear(format=fmt, matrix=1)

    return convert


def _convert(src_rgb: vs.VideoNode, fmt: int) -> vs.VideoNode:
    f = vs.core.get_video_format(fmt)
    if f.color_family == vs.GRAY:
        return src_rgb.resize.Bilinear(format=fmt, matrix=1).std.RemoveFrameProps("_Matrix")
    if f.color_family == vs.YUV:
        return src_rgb.resize.Bilinear(format=fmt, matrix=1)
    return src_rgb if fmt == src_rgb.format.id else src_rgb.resize.Bilinear(format=fmt)


def _geometry(clip: vs.VideoNode, geometry: str) -> vs.VideoNode:
    """Geometry variants for golden cases. `odd` shaves the subsampling-mod
    minimum off each axis so width/height stop being multiples of the SIMD
    vector length; `tiny` is smaller than any vector register, forcing the
    scalar tail paths (cropped from an interior region, not a flat corner)."""
    f = clip.format
    wmod, hmod = 1 << f.subsampling_w, 1 << f.subsampling_h
    if geometry == "full":
        return clip
    if geometry == "odd":
        return clip.std.Crop(right=wmod, bottom=hmod)
    if geometry == "tiny":
        return clip.std.CropAbs(width=13 - 13 % wmod, height=7 - 7 % hmod, left=200, top=100)
    raise ValueError(f"unknown geometry {geometry!r}")


@pytest.fixture(scope="session")
def make_clip(src_rgb: vs.VideoNode):
    """Factory: the source image in any format/geometry, cached per session."""
    cache: dict[tuple, vs.VideoNode] = {}

    def make(fmt: int, geometry: str = "full") -> vs.VideoNode:
        key = (int(fmt), geometry)
        if key not in cache:
            cache[key] = _geometry(_convert(src_rgb, fmt), geometry)
        return cache[key]

    return make


@pytest.fixture(scope="session")
def temporal_rgb(core: vs.Core) -> vs.VideoNode:
    """3-frame 640x320 RGB24 clip; each frame is the crop shifted down one row,
    giving deterministic inter-frame motion for temporal filters."""
    img = core.vszip.ImageRead(str(IMAGE))
    frames = [
        img.std.Crop(left=img.width - 640, top=n, bottom=img.height - 320 - n)
        for n in range(3)
    ]
    return frames[0] + frames[1] + frames[2]


@pytest.fixture(scope="session")
def make_temporal_clip(temporal_rgb: vs.VideoNode):
    """Factory: the 3-frame shifted clip in any format/geometry, cached per
    session. Point resize preserves the dot-crawl-like detail the temporal
    filters (Checkmate, CombMask) react to."""
    cache: dict[tuple, vs.VideoNode] = {}

    def make(fmt: int, geometry: str = "full") -> vs.VideoNode:
        key = (int(fmt), geometry)
        if key not in cache:
            f = vs.core.get_video_format(fmt)
            clip = temporal_rgb.resize.Point(format=fmt, matrix=1)
            if f.color_family == vs.GRAY:
                clip = clip.std.RemoveFrameProps("_Matrix")
            cache[key] = _geometry(clip, geometry)
        return cache[key]

    return make
