"""PNG/BMP fixture writers, vendored from vszip's tests/helpers.py (MIT)."""

import struct
import zlib
from binascii import crc32
from pathlib import Path

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
