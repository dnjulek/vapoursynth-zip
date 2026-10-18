"""Known validation gaps, isolated because one currently crashes the process."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


TESTS_DIR = Path(__file__).resolve().parent


CASES = {
    "bilateral-sigmaR": "clip.vszip.Bilateral(sigmaR=[float('nan')])",
    "limitfilter-dark-thr": "clip.vszip.LimitFilter(clip, dark_thr=[float('nan')])",
    "planeminmax-minthr": "clip.vszip.PlaneMinMax(minthr=float('nan'))",
    "deband-thr": "clip.vszip.Deband(thr=[float('nan')])",
    "wnnm-sigma": "clip.resize.Point(format=vs.GRAYS).vszip.WNNM(sigma=[float('nan')], block_size=4)",
}


@pytest.mark.xfail(
    strict=True,
    reason="known bug: public float arguments do not consistently reject NaN",
)
@pytest.mark.parametrize("expression", CASES.values(), ids=CASES.keys())
def test_nan_is_rejected_without_crashing(expression):
    """The desired API contract is a catchable vs.Error, never UB or a crash.

    strict xfail keeps today's known failures visible without taking down the
    pytest worker. Once a filter is fixed, XPASS deliberately fails so its case
    must be moved out of the known-bug set.
    """
    script = f"""
import sys
sys.path.insert(0, {str(TESTS_DIR)!r})
import vapoursynth as vs
from conftest import _load_vszip
core = _load_vszip()
clip = core.std.BlankClip(None, 16, 16, vs.GRAY8, length=1, color=80)
try:
    out = {expression}
    out.get_frame(0)
except vs.Error:
    print('REJECTED')
    raise SystemExit(0)
print('ACCEPTED')
raise SystemExit(3)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=TESTS_DIR,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0 and "REJECTED" in result.stdout, (
        f"returncode={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"
    )
