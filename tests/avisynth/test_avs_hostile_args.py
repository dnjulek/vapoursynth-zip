"""AviSynth's own parsers must enforce the same hostile-float contract as VS.

Reuse the VS manifest and crash-isolating runner, but pass real V11 doubles
and longs through ctypes. Script parsing must not sanitize NaN/inf or narrow
an i64 before the plugin sees it. No native failure can kill the pytest worker.
"""

import pathlib
import re

import pytest

import avsrt
import harness as H
from test_nonfinite import CASES, MANIFEST, _run_function_cases


def _avs_float_parameters():
    """{function: float parameters} declared by the AVS adapters in src/avisynth."""
    out = {}
    for path in (H.REPO_ROOT / "src" / "avisynth").glob("*.zig"):
        text = path.read_text(encoding="utf-8")
        name = re.search(r'pub const filter_name = "avszip_(\w+)"', text)
        names = {"eedi3": ["EEDI3", "EEDI3H"], "wnnm": ["WNNM"], "bm3d": ["BM3D", "BM3Dv2"]}.get(
            path.stem, [name[1]] if name else [])
        declarations = re.findall(r'(?:pub const params|const shared_params) = ([^;]+);', text)
        signature = "".join(s for d in declarations for s in re.findall(r'"([^"]*)"', d))
        for fn in names:
            out[fn] = set(re.findall(r'\[([^\]]+)\]f', signature))
    return out


# not every vszip function has an AviSynth port (PackRGB, ...)
AVS_FLOAT_PARAMETERS = _avs_float_parameters()
CASES_AVS = [c for c in CASES if c[0] != "Limiter@f16" and c[0].split("@")[0] in AVS_FLOAT_PARAMETERS]
SETUP_AVS = f"""
import sys, types
sys.path.insert(0, {str(pathlib.Path(__file__).resolve().parent)!r})
import avsrt, harness
import numpy as np
env = avsrt.Env(plugins=[harness.AVSZIP])
vs = types.SimpleNamespace(Error=avsrt.AvsError)
class Proxy:
    def __init__(self, clip):
        self.clip = clip
        self.vszip = self
    def __getattr__(self, name):
        def invoke(*args, **kwargs):
            args = [x.clip if isinstance(x, Proxy) else x for x in args]
            try:
                return env.invoke('avszip_' + name, self.clip, *args, **kwargs)
            except avsrt.AvsError as e:
                # Only the plugin's own validation counts as REJECTED: a call
                # the host refuses before the plugin runs (signature/type
                # mismatch) proves nothing about the plugin's checks, so it
                # escapes as a non-vs.Error and the case reports CRASH.
                if any(s in str(e) for s in HOST_SIGNATURE_ERRORS):
                    raise RuntimeError(f"host rejected the call before the plugin ran: {{e}}") from None
                raise
        return invoke
c8 = Proxy(env.eval('BlankClip(width=64,height=64,length=1,pixel_type="YV24")'))
c16 = Proxy(env.eval('BlankClip(width=64,height=64,length=1,pixel_type="YUV444P16")'))
cs = Proxy(env.eval('BlankClip(width=64,height=64,length=1,pixel_type="YUV444PS")'))
gs = Proxy(env.eval('BlankClip(width=64,height=64,length=1,pixel_type="Y32")'))
nan, inf = float('nan'), float('inf')
HOST_SIGNATURE_ERRORS = ("Invalid arguments to function", "had the wrong type", "does not have a named argument", "There is no function named")
def finite_output(clip):
    return all(bool(np.isfinite(p).all()) for p in clip.frame(0).planes)
"""


@pytest.fixture(scope="module")
def outcomes():
    grouped = {}
    for fn, cid, expr, kind in CASES_AVS:
        grouped.setdefault(fn, []).append((cid, expr))
    out = {}
    for cases in grouped.values():
        out.update(_run_function_cases(cases, setup=SETUP_AVS))
    return out


def test_manifest_covers_avs_float_parameters():
    """A new AVS float parameter must join the hostile-input matrix too."""
    want = {(fn, p) for fn, params in AVS_FLOAT_PARAMETERS.items() for p in params}
    have = {(fn.split("@")[0], p) for fn, (_, params, _) in MANIFEST.items() for p in params
            if fn.split("@")[0] in AVS_FLOAT_PARAMETERS}
    assert want == have, f"missing cases: {want-have}; stale cases: {have-want}"


@pytest.mark.parametrize("fn,cid,expr,kind", CASES_AVS, ids=[c[1] for c in CASES_AVS])
def test_hostile_float_argument(outcomes, fn, cid, expr, kind):
    result = outcomes[cid]
    assert not result.startswith("CRASH"), f"{cid}: {result}"
    if kind in ("nonfinite", "overlong"):
        assert result == "REJECTED", f"{cid}: {result}"
    else:
        assert result in ("REJECTED", "ACCEPTED"), f"{cid}: {result}"


@pytest.mark.parametrize("function,param,extra", [
    ("Checkmate", "tmax", {}),
    ("CombMask", "cthresh", {}),
    ("Dither", "bitdepth", {}),
    ("CLAHE", "limit", {}),
    ("EEDI3", "field", {}),
    ("WNNM", "radius", {"block_size": 4}),
    ("WNNM", "block_size", {}),
])
@pytest.mark.parametrize("value", [(1 << 32) + 1, -(1 << 32) + 1, (1 << 63) - 1, -(1 << 63)])
def test_integer_range_checked_before_narrowing(env, function, param, extra, value):
    fmt = "Y32" if function == "WNNM" else "Y8"
    clip = env.eval(f'BlankClip(width=64,height=64,length=3,pixel_type="{fmt}")')
    with pytest.raises(avsrt.AvsError, match=param):
        env.invoke("avszip_" + function, clip, **extra, **{param: value})
