"""Non-finite float arguments: never crash, and NaN must be rejected.

Every public float / float[] parameter of every vszip function is driven with
NaN, +inf and -inf (scalar position, and for arrays also the last accepted
position). Cases run in child processes so a crash (signal, abort, hang)
cannot take down the pytest worker: each child runs all cases of one function
in order and reports them one by one; if it dies mid-case, that case is
recorded as a crash and a fresh child continues after it.

Contract:
- no case may crash or hang, in Debug or ReleaseFast;
- NaN must be rejected with a catchable vs.Error;
- +-inf must not crash; whether it is rejected or has a defined meaning is per
  parameter (bounded parameters reject it through their range check).

Known violations are strict xfails: fixing one makes the test XPASS, which
fails, so the case must then be removed from the known-bug set.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import vapoursynth as vs

TESTS_DIR = Path(__file__).resolve().parent

# function -> (clip setup expression, {param: (is_array, valid_value, max_len)})
# `valid_value` fills the earlier array positions when the bad value is placed
# last; clips are 64x64 so every filter's minimum-size checks pass.
MANIFEST = {
    "AdaptiveGrainMask": ("c8.vszip.AdaptiveGrainMask({a})", {"luma_scaling": (False, 10.0, 1)}),
    "Bilateral": ("c16.vszip.Bilateral({a})", {"sigmaS": (True, 3.0, 3), "sigmaR": (True, 0.02, 3)}),
    "BilateralDither": (
        "c16.vszip.BilateralDither({a})",
        {"thr": (True, 2.5, 3), "flat": (True, 0.4, 3), "wmin": (True, 0.0, 3), "subspl": (True, 0.0, 3)},
    ),
    "Deband": (
        "c16.vszip.Deband({a})",
        {
            "thr": (True, 0.99, 3),
            "thr1": (True, 0.99, 3),
            "thr2": (True, 0.99, 3),
            "grain": (True, 0.0, 2),
            "random_param_ref": (False, 1.0, 1),
            "random_param_grain": (False, 1.0, 1),
            "angle_boost": (False, 1.5, 1),
            "max_angle": (False, 0.15, 1),
        },
    ),
    "EEDI3": (
        "gs.vszip.EEDI3(field=1, {a})",
        {p: (False, v, 1) for p, v in
         [("alpha", 0.2), ("beta", 0.25), ("gamma", 20.0), ("vthresh0", 32.0), ("vthresh1", 64.0), ("vthresh2", 4.0)]},
    ),
    "EEDI3H": (
        "gs.vszip.EEDI3H(field=1, {a})",
        {p: (False, v, 1) for p, v in
         [("alpha", 0.2), ("beta", 0.25), ("gamma", 20.0), ("vthresh0", 32.0), ("vthresh1", 64.0), ("vthresh2", 4.0)]},
    ),
    "LimitFilter": (
        "c16.vszip.LimitFilter(c16, {a})",
        {"dark_thr": (True, 1.0, 3), "bright_thr": (True, 1.0, 3), "elast": (True, 2.0, 3)},
    ),
    "Limiter": ("cs.vszip.Limiter({a})", {"min": (True, 0.0, 3), "max": (True, 0.5, 3)}),
    "PlaneMinMax": ("c8.vszip.PlaneMinMax({a})", {"minthr": (False, 0.0, 1), "maxthr": (False, 0.0, 1)}),
    "WNNM": ("gs.vszip.WNNM(block_size=4, {a})", {"sigma": (True, 3.0, 3)}),
}

SETUP = """
import sys
sys.path.insert(0, {tests!r})
import vapoursynth as vs
from conftest import _load_vszip
core = _load_vszip()
c8 = core.std.BlankClip(None, 64, 64, vs.YUV444P8, length=1, color=[80, 128, 128])
c16 = core.std.BlankClip(None, 64, 64, vs.YUV444P16, length=1, color=[20000, 32768, 32768])
cs = core.std.BlankClip(None, 64, 64, vs.YUV444PS, length=1, color=[0.3, 0.0, 0.0])
gs = core.std.BlankClip(None, 64, 64, vs.GRAYS, length=1, color=0.3)
nan, inf = float('nan'), float('inf')
"""

VALUES = {"nan": "nan", "+inf": "inf", "-inf": "-inf"}


def _cases():
    out = []
    for fn, (expr, params) in MANIFEST.items():
        for param, (is_array, valid, max_len) in params.items():
            positions = [0] + ([max_len - 1] if is_array and max_len > 1 else [])
            for pos in positions:
                for vname, vexpr in VALUES.items():
                    if is_array:
                        arg = "[" + ", ".join([repr(valid)] * pos + [vexpr]) + "]"
                    else:
                        arg = vexpr
                    cid = f"{fn}-{param}[{pos}]-{vname}" if is_array else f"{fn}-{param}-{vname}"
                    out.append((fn, cid, expr.format(a=f"{param}={arg}")))
    return out


CASES = _cases()
CASE_IDS = [c[1] for c in CASES]


def _run_function_cases(cases, timeout=60.0):
    """Run (id, expression) cases in child processes. Returns {id: result}
    with result REJECTED / ACCEPTED / CRASH(<detail>)."""
    results = {}
    todo = list(cases)
    while todo:
        body = "".join(
            f"""
print('START {cid}', flush=True)
try:
    _o = {expr}
    _o.get_frame(0)
    print('ACCEPTED {cid}', flush=True)
except vs.Error:
    print('REJECTED {cid}', flush=True)
"""
            for cid, expr in todo
        )
        script = SETUP.format(tests=str(TESTS_DIR)) + body
        try:
            proc = subprocess.run(
                [sys.executable, "-c", script], cwd=TESTS_DIR, env=os.environ.copy(),
                capture_output=True, text=True, timeout=timeout,
            )
            stdout, code = proc.stdout, proc.returncode
            detail = f"exit {code}: {proc.stderr.strip().splitlines()[-1][:120] if proc.stderr.strip() else ''}"
        except subprocess.TimeoutExpired as e:
            stdout = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            code, detail = None, "hang (timeout)"
        running = None
        for line in stdout.splitlines():
            tag, _, cid = line.partition(" ")
            if tag == "START":
                running = cid
            elif tag in ("ACCEPTED", "REJECTED"):
                results[cid] = tag
                running = None
        done = {cid for cid, _ in todo if cid in results}
        if running is not None:  # died inside this case
            results[running] = f"CRASH({detail})"
            done.add(running)
        elif code not in (0,) and not done:
            # died before the first case (setup failure): report all
            for cid, _ in todo:
                results[cid] = f"CRASH(setup {detail})"
            break
        todo = [(cid, expr) for cid, expr in todo if cid not in done]
    return results


@pytest.fixture(scope="module")
def results():
    by_fn: dict[str, list] = {}
    for fn, cid, expr in CASES:
        by_fn.setdefault(fn, []).append((cid, expr))
    out = {}
    for fn, cases in by_fn.items():
        out.update(_run_function_cases(cases))
    return out


# --- known bugs (strict: an XPASS means the case was fixed - remove it) ------

# Crash in at least one build mode (Debug panics where ReleaseFast may run on
# with UB). A listed case only excuses an actual crash; if it doesn't crash in
# the build under test, the test simply passes.
KNOWN_CRASHES = {
    # NaN reaches the PBFIC setup, `num - 2` underflows: Debug panic,
    # ReleaseFast segfault (pytest_review.md)
    "Bilateral-sigmaR[0]-nan",
    "Bilateral-sigmaR[2]-nan",
    # Debug: integer-overflow panic deriving the spatial radius from +inf
    "Bilateral-sigmaS[0]-+inf",
    "Bilateral-sigmaS[2]-+inf",
}

# NaN accepted without an error (validation only uses `< min` / `> max`, both
# false for NaN). Only Limiter rejects NaN today.
KNOWN_NAN_ACCEPTED = {
    "AdaptiveGrainMask-luma_scaling-nan",
    "PlaneMinMax-minthr-nan",
    "PlaneMinMax-maxthr-nan",
    *(f"{fn}-{p}-nan" for fn in ("EEDI3", "EEDI3H")
      for p in ("alpha", "beta", "gamma", "vthresh0", "vthresh1", "vthresh2")),
    *(f"Deband-{p}-nan" for p in ("random_param_ref", "random_param_grain", "angle_boost", "max_angle")),
    *(f"{fn}-{p}[{i}]-nan" for fn, p, n in [
        ("Bilateral", "sigmaS", 3), ("BilateralDither", "thr", 3), ("BilateralDither", "flat", 3),
        ("BilateralDither", "wmin", 3), ("BilateralDither", "subspl", 3), ("Deband", "thr", 3),
        ("Deband", "thr1", 3), ("Deband", "thr2", 3), ("Deband", "grain", 2),
        ("LimitFilter", "dark_thr", 3), ("LimitFilter", "bright_thr", 3), ("LimitFilter", "elast", 3),
        ("WNNM", "sigma", 3),
    ] for i in (0, n - 1)),
}


def test_manifest_covers_every_float_parameter(core):
    """A new float parameter must be added to MANIFEST (and so to the matrix)."""
    have = {(fn, p) for fn, (_, params) in MANIFEST.items() for p in params}
    want = set()
    for f in core.vszip.functions():
        for arg in f.signature.split(";"):
            parts = arg.split(":")
            if len(parts) >= 2 and parts[1] in ("float", "float[]"):
                want.add((f.name, parts[0]))
    assert want == have, f"missing: {sorted(want - have)}; stale: {sorted(have - want)}"


@pytest.mark.parametrize("cid", CASE_IDS)
def test_nonfinite_never_crashes(results, cid):
    r = results[cid]
    if r.startswith("CRASH") and cid in KNOWN_CRASHES:
        pytest.xfail(f"known crash: {r}")
    assert not r.startswith("CRASH"), f"{cid}: {r}"


def test_known_sets_are_real_cases():
    unknown = (KNOWN_CRASHES | KNOWN_NAN_ACCEPTED) - set(CASE_IDS)
    assert not unknown, f"known-bug entries that are not cases: {sorted(unknown)}"


NAN_IDS = [cid for cid in CASE_IDS if cid.endswith("-nan")]


@pytest.mark.parametrize("cid", NAN_IDS)
def test_nan_is_rejected(results, cid):
    r = results[cid]
    if cid in KNOWN_NAN_ACCEPTED or cid in KNOWN_CRASHES:
        if r == "REJECTED":
            pytest.fail(f"{cid} is fixed (now REJECTED): remove it from the known-bug sets")
        pytest.xfail(f"known bug: NaN not rejected ({r})")
    assert r == "REJECTED", f"{cid}: {r}"
