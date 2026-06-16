"""Stdlib-only helpers for the vszip test suite: frame/prop access, clip
comparison, and minimal PNG/BMP encoders for ImageRead tests."""

import struct
import zlib
from binascii import crc32
from pathlib import Path

import vapoursynth as vs


# --- frame access -----------------------------------------------------------


def props(clip: vs.VideoNode, n: int = 0) -> dict:
    with clip.get_frame(n) as f:
        return dict(f.props)


def pix(clip: vs.VideoNode, x: int, y: int, plane: int = 0, n: int = 0):
    with clip.get_frame(n) as f:
        return f[plane][y, x]


def plane_stats(clip: vs.VideoNode, ref: vs.VideoNode | None = None, plane: int = 0, n: int = 0) -> dict:
    return props(clip.std.PlaneStats(ref, plane=plane), n)


def avg(clip: vs.VideoNode, plane: int = 0, n: int = 0) -> float:
    return plane_stats(clip, plane=plane, n=n)["PlaneStatsAverage"]


def diff(a: vs.VideoNode, b: vs.VideoNode, plane: int = 0, n: int = 0) -> float:
    return plane_stats(a, b, plane=plane, n=n)["PlaneStatsDiff"]


def max_abs_diff(a: vs.VideoNode, b: vs.VideoNode, plane: int = 0, n: int = 0) -> float:
    """Largest per-pixel absolute difference, in pixel-value units.
    std.Expr has no half-float support, so don't call this on F16 clips."""
    d = vs.core.std.Expr([a, b], "x y - abs")
    return plane_stats(d, plane=plane, n=n)["PlaneStatsMax"]


# --- clip assertions --------------------------------------------------------


def assert_same_clip(a: vs.VideoNode, b: vs.VideoNode, n: int | None = None) -> None:
    """a and b have identical format, dimensions and bit-identical pixels
    (frame props are deliberately not compared)."""
    assert a.format.id == b.format.id, f"format mismatch: {a.format.name} != {b.format.name}"
    assert (a.width, a.height) == (b.width, b.height)
    assert a.num_frames == b.num_frames
    frames = range(a.num_frames) if n is None else [n]
    for fn in frames:
        for plane in range(a.format.num_planes):
            d = diff(a, b, plane=plane, n=fn)
            assert d == 0.0, f"frame {fn} plane {plane}: PlaneStatsDiff={d}"


def assert_binary(clip: vs.VideoNode, n: int = 0) -> None:
    """Every pixel of an 8-bit clip is exactly 0 or 255."""
    mask = clip.std.Expr("x 0 = x 255 = or 255 0 ?")
    assert plane_stats(mask, n=n)["PlaneStatsMin"] == 255


def assert_has_gradient(clip: vs.VideoNode, n: int = 0) -> None:
    """At least one pixel is strictly between 0 and 255 (a grayscale ramp,
    not a binary 0/255 mask)."""
    inter = clip.std.Expr("x 0 > x 255 < and 255 0 ?")
    assert plane_stats(inter, n=n)["PlaneStatsMax"] == 255, "expected intermediate (gray) values, got a binary mask"


def repack(clip: vs.VideoNode) -> vs.VideoNode:
    """Bit-identical copy in freshly allocated, compactly strided frames.
    Two flips force real copies (a no-op resize may pass frames through), so
    `filter(cropped)` vs `filter(repack(cropped))` exercises the plugin's
    stride/offset handling."""
    return clip.std.FlipVertical().std.FlipVertical()


# --- minimal PNG encoder ----------------------------------------------------

_PNG_SIG = b"\x89PNG\r\n\x1a\n"
_COLOR_TYPE = {"gray": 0, "rgb": 2, "palette": 3, "graya": 4, "rgba": 6}
_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}


def _chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc32(tag + data))


def _pack_row(samples: list[int], bitdepth: int) -> bytes:
    if bitdepth == 16:
        return b"".join(struct.pack(">H", s) for s in samples)
    if bitdepth == 8:
        return bytes(samples)
    out = bytearray()
    acc = 0
    nbits = 0
    for s in samples:
        acc = (acc << bitdepth) | (s & ((1 << bitdepth) - 1))
        nbits += bitdepth
        if nbits == 8:
            out.append(acc)
            acc = 0
            nbits = 0
    if nbits:
        out.append(acc << (8 - nbits))
    return bytes(out)


def write_png(
    path: Path,
    rows: list[list],
    *,
    color: str = "rgb",
    bitdepth: int = 8,
    palette: list[tuple] | None = None,
    extra_chunks: tuple = (),
) -> Path:
    """Write a PNG. `rows` is height lists of width pixels; a pixel is an int
    (gray/palette) or a tuple of channel ints fitting `bitdepth`.
    `extra_chunks` are (tag, data) pairs inserted before IDAT."""
    ct = _COLOR_TYPE[color]
    nch = _CHANNELS[ct]
    width = len(rows[0])
    raw = bytearray()
    for row in rows:
        assert len(row) == width
        samples = []
        for px in row:
            px = px if isinstance(px, (tuple, list)) else (px,)
            assert len(px) == nch
            samples.extend(px)
        raw.append(0)  # filter type: None
        raw += _pack_row(samples, bitdepth)

    out = bytearray(_PNG_SIG)
    out += _chunk(b"IHDR", struct.pack(">IIBBBBB", width, len(rows), bitdepth, ct, 0, 0, 0))
    if ct == 3:
        out += _chunk(b"PLTE", b"".join(bytes(c) for c in palette))
    for tag, data in extra_chunks:
        out += _chunk(tag, data)
    out += _chunk(b"IDAT", zlib.compress(bytes(raw)))
    out += _chunk(b"IEND", b"")
    path.write_bytes(bytes(out))
    return path


def gama_chunk(gamma_times_100000: int) -> tuple[bytes, bytes]:
    return b"gAMA", struct.pack(">I", gamma_times_100000)


def srgb_chunk() -> tuple[bytes, bytes]:
    return b"sRGB", b"\x00"


def chrm_chunk(wx, wy, rx, ry, gx, gy, bx, by) -> tuple[bytes, bytes]:
    return b"cHRM", struct.pack(">8I", wx, wy, rx, ry, gx, gy, bx, by)


def cicp_chunk(primaries: int, transfer: int, matrix: int = 0, full_range: int = 1) -> tuple[bytes, bytes]:
    return b"cICP", bytes((primaries, transfer, matrix, full_range))


# --- minimal BMP encoder (24-bit BI_RGB, BITMAPV4HEADER) ---------------------


def write_bmp(path: Path, rows: list[list[tuple]]) -> Path:
    """Write an uncompressed 24-bit BMP. `rows` is top-down (r, g, b) tuples.
    Uses a 108-byte V4 info header: zigimg's reader rejects the classic
    40-byte BITMAPINFOHEADER. zigimg also ignores BMP row padding, so use a
    width divisible by 4 to keep the file unambiguous."""
    width = len(rows[0])
    row_size = (width * 3 + 3) & ~3
    pixel_data = bytearray()
    for row in reversed(rows):  # BMP stores bottom-up
        line = bytearray()
        for r, g, b in row:
            line += bytes((b, g, r))
        line += b"\x00" * (row_size - len(line))
        pixel_data += line
    header = struct.pack("<2sIHHI", b"BM", 14 + 108 + len(pixel_data), 0, 0, 14 + 108)
    dib = struct.pack("<IiiHHIIiiII", 108, width, len(rows), 1, 24, 0, len(pixel_data), 2835, 2835, 0, 0)
    dib += struct.pack("<4I", 0, 0, 0, 0)  # RGBA masks (unused for BI_RGB)
    dib += struct.pack("<I", 0) + b"\x00" * 36 + struct.pack("<3I", 0, 0, 0)  # cs type, endpoints, gamma
    path.write_bytes(header + dib + pixel_data)
    return path
