"""Hostile float arguments: never crash, reject what can't be represented.

Every public float / float[] parameter of every vszip function is driven with
three kinds of value:

- ``nonfinite``: NaN, +inf, -inf, at the scalar position and, for arrays,
  also the last accepted position;
- ``extreme``: finite values far outside any sane range (+-1e300, 1e-300);
- ``overlong``: arrays one element longer than the parameter accepts, both
  all-finite and with a NaN in the extra position.

Cases run in child processes so a crash (signal, abort, hang) cannot take down
the pytest worker: each child runs all cases of one function in order and
reports them one by one; if it dies mid-case, that case is recorded as a crash
and a fresh child continues after it.

Contract (enforced in Create by src/helper.zig checkFinite / checkArg /
checkArrayLen / getArray / getFloat / Maps):

- no case may crash or hang, in debug or fast builds;
- ``nonfinite`` and ``overlong`` cases are rejected with a catchable vs.Error;
- an ``extreme`` value is either rejected by a range check, or accepted and
  then the output frame must contain only finite samples (a finite argument
  must not be silently narrowed into an inf/NaN internal value).

Known violations go into the KNOWN_* sets below; a listed case that starts
behaving correctly fails the run so the entry gets removed.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import vapoursynth as vs

TESTS_DIR = Path(__file__).resolve().parent

# function -> (call template, {param: (is_array, valid_value, max_len)}, companions)
#
# `{a}` in the template receives the argument under test. `valid_value` fills
# the other array positions. `companions` are other arguments the call needs
# to get past unrelated validation (Limiter requires min and max together);
# the one under test is left out. Functions in EXACT_LEN take arrays of exactly
# max_len values, so their arrays are always built full-length. A
# "Name@variant" key tests the same function on a second clip type.
MANIFEST = {
    "AdaptiveGrainMask": ("c8.vszip.AdaptiveGrainMask({a})", {"luma_scaling": (False, 10.0, 1)}, {}),
    "Bilateral": ("c16.vszip.Bilateral({a})", {"sigmaS": (True, 3.0, 3), "sigmaR": (True, 0.02, 3)}, {}),
    "BM3D": ("gs.vszip.BM3D({a})", {"sigma": (True, 3.0, 3)}, {}),
    "BM3Dv2": ("gs.vszip.BM3Dv2({a})", {"sigma": (True, 3.0, 3)}, {}),
    "BM3DBasic": (
        "c8.vszip.BM3DBasic({a})",
        {"sigma": (True, 10.0, 3), "th_mse": (False, 400.0, 1), "hard_thr": (False, 2.7, 1)},
        {},
    ),
    "BM3DFinal": ("c8.vszip.BM3DFinal(c8, {a})", {"sigma": (True, 10.0, 3), "th_mse": (False, 200.0, 1)}, {}),
    "BM3DVBasic": (
        "c8.vszip.BM3DVBasic({a})",
        {"sigma": (True, 10.0, 3), "th_mse": (False, 400.0, 1), "hard_thr": (False, 2.7, 1)},
        {},
    ),
    "BM3DVFinal": ("c8.vszip.BM3DVFinal(c8, {a})", {"sigma": (True, 10.0, 3), "th_mse": (False, 200.0, 1)}, {}),
    "BilateralDither": (
        "c16.vszip.BilateralDither({a})",
        {"thr": (True, 2.5, 3), "flat": (True, 0.4, 3), "wmin": (True, 0.0, 3), "subspl": (True, 0.0, 3)},
        {},
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
        {},
    ),
    "EEDI3": (
        "gs.vszip.EEDI3(field=1, {a})",
        {p: (False, v, 1) for p, v in
         [("alpha", 0.2), ("beta", 0.25), ("gamma", 20.0), ("vthresh0", 32.0), ("vthresh1", 64.0), ("vthresh2", 4.0)]},
        {},
    ),
    "EEDI3H": (
        "gs.vszip.EEDI3H(field=1, {a})",
        {p: (False, v, 1) for p, v in
         [("alpha", 0.2), ("beta", 0.25), ("gamma", 20.0), ("vthresh0", 32.0), ("vthresh1", 64.0), ("vthresh2", 4.0)]},
        {},
    ),
    "LimitFilter": (
        "c16.vszip.LimitFilter(c16, {a})",
        {"dark_thr": (True, 1.0, 3), "bright_thr": (True, 1.0, 3), "elast": (True, 2.0, 3)},
        {},
    ),
    # float clip; bounds narrow to f32
    "Limiter": (
        "cs.vszip.Limiter({a})",
        {"min": (True, -0.5, 3), "max": (True, 0.5, 3)},
        {"min": "[-0.5, -0.5, -0.5]", "max": "[1.0, 0.5, 0.5]"},
    ),
    # half-float clip; bounds narrow to f16 (max 65504)
    "Limiter@f16": (
        "ch.vszip.Limiter({a})",
        {"min": (True, -0.5, 3), "max": (True, 0.5, 3)},
        {"min": "[-0.5, -0.5, -0.5]", "max": "[1.0, 0.5, 0.5]"},
    ),
    # integer clip; bounds truncate to integers: a separate code path
    "Limiter@int": (
        "c16.vszip.Limiter({a})",
        {"min": (True, 0.0, 3), "max": (True, 255.0, 3)},
        {"min": "[0, 0, 0]", "max": "[65535, 65535, 65535]"},
    ),
    "NLMeans": ("gs.vszip.NLMeans({a})", {"h": (False, 1.2, 1), "wref": (False, 1.0, 1)}, {}),
    "PlaneMinMax": ("c8.vszip.PlaneMinMax({a})", {"minthr": (False, 0.0, 1), "maxthr": (False, 0.0, 1)}, {}),
    "WNNM": ("gs.vszip.WNNM(block_size=4, {a})", {"sigma": (True, 3.0, 3)}, {}),
    "Tonemap": (
        "c16.vszip.Tonemap({a})",
        {
            "src_max": (False, 1000.0, 1),
            "src_min": (False, 0.005, 1),
            "dst_max": (False, 203.0, 1),
            "dst_min": (False, 0.2, 1),
            "smoothing_period": (False, 20.0, 1),
            "scene_threshold_low": (False, 1.0, 1),
            "scene_threshold_high": (False, 3.0, 1),
            "percentile": (False, 100.0, 1),
            "black_cutoff": (False, 1.0, 1),
            "tone_mapping_param": (False, 0.5, 1),
            "contrast_recovery": (False, 0.3, 1),
            "contrast_smoothness": (False, 3.5, 1),
        },
        {},
    ),
}
EXACT_LEN = {"Limiter", "Limiter@f16", "Limiter@int"}

SETUP = """
import sys
sys.path.insert(0, {tests!r})
import vapoursynth as vs
import numpy as np
from conftest import _load_vszip
core = _load_vszip()
c8 = core.std.BlankClip(None, 64, 64, vs.YUV444P8, length=1, color=[80, 128, 128])
c16 = core.std.BlankClip(None, 64, 64, vs.YUV444P16, length=1, color=[20000, 32768, 32768])
cs = core.std.BlankClip(None, 64, 64, vs.YUV444PS, length=1, color=[0.3, 0.0, 0.0])
ch = core.std.BlankClip(None, 64, 64, vs.YUV444PH, length=1, color=[0.3, 0.0, 0.0])
gs = core.std.BlankClip(None, 64, 64, vs.GRAYS, length=1, color=0.3)
nan, inf = float('nan'), float('inf')

def finite_output(clip):
    with clip.get_frame(0) as f:
        if f.format.sample_type != vs.FLOAT:
            return True
        return all(bool(np.isfinite(np.asarray(f[p])).all()) for p in range(f.format.num_planes))
"""

NONFINITE = {"nan": "nan", "+inf": "inf", "-inf": "-inf"}
EXTREME = {"+1e300": "1e300", "-1e300": "-1e300", "1e-300": "1e-300"}


def _cases():
    """[(function key, case id, expression, kind)]"""
    out = []
    for fn, (template, params, companions) in MANIFEST.items():
        exact = fn in EXACT_LEN
        for param, (is_array, valid, max_len) in params.items():
            others = ", ".join(f"{k}={v}" for k, v in companions.items() if k != param)

            def call(arg, param=param, others=others, template=template):
                a = f"{param}={arg}" + (f", {others}" if others else "")
                return template.format(a=a)

            positions = [0] + ([max_len - 1] if is_array and max_len > 1 else [])
            for kind, values in (("nonfinite", NONFINITE), ("extreme", EXTREME)):
                for pos in positions:
                    for vname, vexpr in values.items():
                        if is_array:
                            elems = [repr(valid)] * (max_len if exact else pos + 1)
                            elems[pos] = vexpr
                            arg = "[" + ", ".join(elems) + "]"
                            cid = f"{fn}-{param}[{pos}]-{vname}"
                        else:
                            arg = vexpr
                            cid = f"{fn}-{param}-{vname}"
                        out.append((fn, cid, call(arg), kind))
            if is_array:
                full = [repr(valid)] * (max_len + 1)
                out.append((fn, f"{fn}-{param}-overlong-finite", call("[" + ", ".join(full) + "]"), "overlong"))
                full[-1] = "nan"
                out.append((fn, f"{fn}-{param}-overlong-nan", call("[" + ", ".join(full) + "]"), "overlong"))
        if fn in EXACT_LEN:
            # paired bounds: with min = max = v nothing re-clamps the samples,
            # so an extreme bound narrowed to inf shows up in the output
            # (a single extreme bound next to a sane one stays hidden)
            n = next(iter(params.values()))[2]
            for vname, vexpr in EXTREME.items():
                arr = "[" + ", ".join([vexpr] * n) + "]"
                out.append((fn, f"{fn}-paired-{vname}", template.format(a=f"min={arr}, max={arr}"), "extreme"))
    return out


CASES = _cases()
CASE_IDS = [c[1] for c in CASES]
KIND = {c[1]: c[3] for c in CASES}
assert len(set(CASE_IDS)) == len(CASE_IDS), "duplicate case ids"


def _run_function_cases(cases, timeout=60.0, *, setup=None):
    """Run (id, expression) cases in child processes. Returns {id: result}
    with result REJECTED / ACCEPTED / ACCEPTED_NONFINITE_OUTPUT / CRASH(...)."""
    results = {}
    todo = list(cases)
    while todo:
        body = "".join(
            f"""
print('START {cid}', flush=True)
try:
    _o = {expr}
    _ok = finite_output(_o)
    print(('ACCEPTED' if _ok else 'ACCEPTED_NONFINITE_OUTPUT') + ' {cid}', flush=True)
except vs.Error:
    print('REJECTED {cid}', flush=True)
"""
            for cid, expr in todo
        )
        script = (SETUP.format(tests=str(TESTS_DIR)) if setup is None else setup) + body
        try:
            proc = subprocess.run(
                [sys.executable, "-c", script], cwd=TESTS_DIR, env=os.environ.copy(),
                capture_output=True, text=True, timeout=timeout,
            )
            stdout, code = proc.stdout, proc.returncode
            last = proc.stderr.strip().splitlines()[-1][:120] if proc.stderr.strip() else ""
            detail = f"exit {code}: {last}"
        except subprocess.TimeoutExpired as e:
            stdout = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            code, detail = None, "hang (timeout)"
        running = None
        for line in stdout.splitlines():
            tag, _, cid = line.partition(" ")
            if tag == "START":
                running = cid
            elif tag in ("ACCEPTED", "ACCEPTED_NONFINITE_OUTPUT", "REJECTED"):
                results[cid] = tag
                running = None
        done = {cid for cid, _ in todo if cid in results}
        if running is not None:  # died inside this case
            results[running] = f"CRASH({detail})"
            done.add(running)
        elif code != 0 and not done:
            # died before the first case (setup failure): report all
            for cid, _ in todo:
                results[cid] = f"CRASH(setup {detail})"
            break
        todo = [(cid, expr) for cid, expr in todo if cid not in done]
    return results


@pytest.fixture(scope="module")
def results():
    pytest.importorskip("numpy")
    by_fn: dict[str, list] = {}
    for fn, cid, expr, _ in CASES:
        by_fn.setdefault(fn, []).append((cid, expr))
    out = {}
    for cases in by_fn.values():
        out.update(_run_function_cases(cases))
    return out


# --- known bugs ---------------------------------------------------------------
# Empty since the shared validation helpers landed. A listed case only excuses
# the listed misbehaviour; if it behaves correctly the test fails so the entry
# is removed.

KNOWN_CRASHES: set[str] = set()  # crash in at least one build mode
KNOWN_NOT_REJECTED: set[str] = set()  # nonfinite/overlong accepted without vs.Error
KNOWN_NONFINITE_OUTPUT: set[str] = set()  # extreme finite arg accepted -> inf/NaN samples


def test_manifest_covers_every_float_parameter(core):
    """A new float parameter must be added to MANIFEST (and so to the matrix)."""
    have = {(fn.split("@")[0], p) for fn, (_, params, _) in MANIFEST.items() for p in params}
    want = set()
    for f in core.vszip.functions():
        for arg in f.signature.split(";"):
            parts = arg.split(":")
            if len(parts) >= 2 and parts[1] in ("float", "float[]"):
                want.add((f.name, parts[0]))
    assert want == have, f"missing: {sorted(want - have)}; stale: {sorted(have - want)}"


def test_case_kinds_are_complete():
    """Every nonfinite value is selected, including -inf (whose id ends in
    '--inf' and used to be lost by parsing the id)."""
    positions = sum(
        1 + (1 if is_array and max_len > 1 else 0)
        for _, params, _ in MANIFEST.values()
        for is_array, _, max_len in params.values()
    )
    assert sum(1 for k in KIND.values() if k == "nonfinite") == positions * len(NONFINITE)
    assert sum(1 for c in CASE_IDS if c.endswith("--inf") and KIND[c] == "nonfinite") == positions


def test_known_sets_are_real_cases():
    unknown = (KNOWN_CRASHES | KNOWN_NOT_REJECTED | KNOWN_NONFINITE_OUTPUT) - set(CASE_IDS)
    assert not unknown, f"known-bug entries that are not cases: {sorted(unknown)}"


@pytest.mark.parametrize("cid", CASE_IDS)
def test_never_crashes(results, cid):
    r = results[cid]
    if r.startswith("CRASH") and cid in KNOWN_CRASHES:
        pytest.xfail(f"known crash: {r}")
    assert not r.startswith("CRASH"), f"{cid}: {r}"


REJECT_IDS = [cid for cid in CASE_IDS if KIND[cid] in ("nonfinite", "overlong")]


@pytest.mark.parametrize("cid", REJECT_IDS)
def test_nonfinite_and_overlong_are_rejected(results, cid):
    r = results[cid]
    if cid in KNOWN_NOT_REJECTED or cid in KNOWN_CRASHES:
        if r == "REJECTED":
            pytest.fail(f"{cid} is fixed (now REJECTED): remove it from the known-bug sets")
        pytest.xfail(f"known bug: not rejected ({r})")
    assert r == "REJECTED", f"{cid}: {r}"


EXTREME_IDS = [cid for cid in CASE_IDS if KIND[cid] == "extreme"]


@pytest.mark.parametrize("cid", EXTREME_IDS)
def test_extreme_finite_never_yields_nonfinite_output(results, cid):
    r = results[cid]
    if r == "ACCEPTED_NONFINITE_OUTPUT" and cid in KNOWN_NONFINITE_OUTPUT:
        pytest.xfail("known bug: finite argument narrowed to inf/NaN")
    assert r != "ACCEPTED_NONFINITE_OUTPUT", f"{cid}: accepted, but the frame has inf/NaN samples"


def test_rejection_message_names_the_parameter(core):
    """The shared helper's message identifies the argument and the value."""
    clip = core.std.BlankClip(None, 64, 64, vs.YUV444P16, length=1)
    with pytest.raises(vs.Error, match=r"LimitFilter: dark_thr value nan is not a finite number"):
        clip.vszip.LimitFilter(clip, dark_thr=[float("nan")])
    with pytest.raises(vs.Error, match=r"Bilateral: sigmaS\[2\] value inf is not a finite number"):
        clip.vszip.Bilateral(sigmaS=[3, 3, float("inf")])
    with pytest.raises(vs.Error, match=r"Deband: max_angle value nan is not a finite number"):
        clip.vszip.Deband(max_angle=float("nan"))
    with pytest.raises(vs.Error, match=r"WNNM: sigma has too many elements \(got 4, max 3\)"):
        clip.resize.Point(format=vs.YUV444PS).vszip.WNNM(sigma=[3, 3, 3, 3])
    # extreme magnitudes print in scientific notation, not as 300 digits
    fclip = core.std.BlankClip(None, 8, 8, vs.GRAYS, length=1)
    with pytest.raises(vs.Error, match=r"Limiter: min value 1e300 is above maximum 3\.40282"):
        fclip.vszip.Limiter(min=[1e300], max=[1e300])
