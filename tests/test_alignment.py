"""Alignment contract test.

Several hot lookup tables are declared `align(64)` so their fixed-phase
vector loads never split a cache line (measured wins: +6% on zimg_random
dithering alone; see mt_mode.md §5c/§5e). Zig's default for comptime arrays
is only the element's natural alignment, so an accidentally dropped
annotation compiles fine and silently costs performance — this test pins the
contract by reading the built plugin's symbol addresses.

Runs against unstripped builds (Debug, or ReleaseFast with symbols); skips
when the library is stripped, since the annotations can only be verified
through the symbol table.
"""

import shutil
import subprocess

import pytest

from conftest import _plugin_path

# Globals declared align(64); symbol names as emitted by the Zig compiler.
ALIGNED_TABLES = [
    "filters.dither_pattern.zimg_blue_noise_f32",
    "filters.dither_pattern.zimg_bayer_rot",
    "filters.dither_pattern.fmtc_bayer_f32",
    "filters.dither_pattern.fmtc_void_f32",
    "filters.dither_pattern.fmtc_bayer_i16",
    "filters.dither_pattern.fmtc_void_i16",
    "filters.adaptive_grain_mask.PowConsts(8).rows",
]


def _symbol_addresses() -> dict[str, int]:
    nm = shutil.which("nm")
    if nm is None:
        pytest.skip("nm not available")
    lib = _plugin_path()
    if not lib.is_file():
        pytest.skip(f"plugin not found at {lib}")
    out = subprocess.run([nm, str(lib)], capture_output=True, text=True)
    if out.returncode != 0:
        pytest.skip(f"nm failed: {out.stderr.strip()[:100]}")
    addrs: dict[str, int] = {}
    for line in out.stdout.splitlines():
        parts = line.split(maxsplit=2)
        if len(parts) == 3 and parts[1] in ("r", "R", "d", "D"):
            addrs[parts[2]] = int(parts[0], 16)
    return addrs


def test_hot_tables_are_cache_line_aligned():
    addrs = _symbol_addresses()
    found = {name: addrs[name] for name in ALIGNED_TABLES if name in addrs}
    if not found:
        # PowConsts(8) only exists on 8-lane builds; the dither tables exist
        # on every build — none at all means a stripped library.
        pytest.skip("no annotated symbols in the library (stripped build)")
    misaligned = {n: hex(a) for n, a in found.items() if a % 64 != 0}
    assert not misaligned, (
        f"tables lost their align(64) (fixed-phase vector loads will split "
        f"cache lines): {misaligned}"
    )
    # The dither pattern tables must all be present together on any
    # unstripped build — catch a silent rename breaking this test's coverage.
    dither_tables = [n for n in ALIGNED_TABLES if "dither_pattern" in n]
    missing = [n for n in dither_tables if n not in addrs]
    assert not missing, f"expected symbols missing (renamed?): {missing}"
