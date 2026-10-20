"""AviSynth+ differential suite: avszip (AviSynth+) vs vszip (VapourSynth).

Opt-in: it needs `zig build -Davisynth=true` (zig-out/bin/avszip.dll) and an
installed AviSynth+ (Windows). Without them the whole directory is skipped at
collection, so the default `pytest` run (and CI) is unaffected.
"""

import ctypes
import os
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _unavailable():
    if sys.platform != "win32":
        return "AviSynth+ suite runs on Windows only"
    lib = pathlib.Path(os.environ.get("AVSZIP_LIB", HERE.parents[1] / "zig-out" / "bin" / "avszip.dll"))
    if not lib.is_file():
        return f"{lib} not found (build with `zig build -Davisynth=true` or set AVSZIP_LIB)"
    try:
        ctypes.CDLL("AviSynth.dll")
    except OSError:
        return "AviSynth.dll (AviSynth+) is not installed"
    return None


_REASON = _unavailable()
if _REASON:
    collect_ignore_glob = ["*.py"]


def pytest_report_header(config):
    return f"avisynth: {'skipped: ' + _REASON if _REASON else 'enabled'}"


if not _REASON:
    import avsrt  # noqa: E402
    import harness  # noqa: E402

    @pytest.fixture
    def env():
        e = avsrt.Env(plugins=[harness.AVSZIP])
        yield e
        e.close()
