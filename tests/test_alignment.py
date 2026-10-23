"""Binary-level contracts that protect hot-table alignment.

The tables are `align(common.vec_len)`: 16 bytes on baseline x86-64/aarch64,
32 on AVX2, 64 on AVX-512 builds. The binary does not report its `vec_len`, so
the test checks the 16-byte floor every target meets."""

import shutil
import subprocess

import pytest

from conftest import _plugin_path

MIN_VEC_LEN = 16


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
    result = subprocess.run([nm, str(lib)], capture_output=True, text=True)
    if result.returncode != 0:
        pytest.skip(f"nm failed: {result.stderr.strip()[:100]}")
    addresses = {}
    for line in result.stdout.splitlines():
        parts = line.split(maxsplit=2)
        if len(parts) == 3 and parts[1] in ("r", "R", "d", "D"):
            addresses[parts[2]] = int(parts[0], 16)
    return addresses


@pytest.mark.skipif(__import__("sys").platform == "win32", reason="nm symbol test is Unix-only")
def test_hot_tables_are_vector_aligned():
    addresses = _symbol_addresses()
    found = {name: addresses[name] for name in ALIGNED_TABLES if name in addresses}
    if not found:
        pytest.skip("annotated symbols unavailable in stripped binary")
    misaligned = {name: hex(address) for name, address in found.items() if address % MIN_VEC_LEN}
    assert not misaligned, f"tables lost align(vec_len): {misaligned}"

    dither_tables = [name for name in ALIGNED_TABLES if "dither_pattern" in name]
    missing = [name for name in dither_tables if name not in addresses]
    assert not missing, f"expected symbols missing or renamed: {missing}"
