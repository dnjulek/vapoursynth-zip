"""Registration contract: every function exists, is reachable under the
DLL-basename alias too, and dispatches to its own create callback.

The dispatch check guards the AviSynth+ SaveString-cache bug worked around in
avisynth-zig's `register` (NULL user_data records could alias each other).
"""

import pytest

import harness as H
from avsrt import AvsError

NAMES = [
    "Limiter", "PlaneAverage", "PlaneMinMax", "RFS", "BoxBlur", "AdaptiveBinarize", "AdaptiveGrainMask",
    "ColorMap", "CombMask", "CombMaskMT", "LimitFilter", "Compress", "Bilateral", "BilateralDither",
    "MosquitoNR", "Dither", "Deband", "Checkmate", "CLAHE", "XPSNR", "SSIMULACRA2", "EEDI3", "EEDI3H", "WNNM",
    "NLMeans",
]

# Registered too, but take no clip as first argument / are internal.
OTHER = ["ImageRead", "WNNMRaw"]


@pytest.mark.parametrize("name", NAMES + OTHER)
def test_function_exists_and_alias(env, name):
    env.eval(f'Assert(FunctionExists("avszip_{name}"), "missing avszip_{name}")\n'
             f'Assert(FunctionExists("avszip_avszip_{name}"), "missing alias avszip_avszip_{name}")', want_clip=False)


@pytest.mark.parametrize("name", NAMES)
def test_dispatches_to_own_callback(env, name):
    # A tiny YV411 clip that most filters refuse inside their own create;
    # whatever create raises must carry this function's own name.
    env.eval('tiny = BlankClip(pixel_type="YV411", width=4, height=2, length=1)', want_clip=False)
    two = name in ("RFS", "AdaptiveBinarize", "LimitFilter", "XPSNR", "SSIMULACRA2")
    args = "tiny, tiny" if two else "tiny"
    extra = {"Dither": ", 8", "RFS": ", frames=[0]", "PlaneAverage": ", exclude=[-1]", "EEDI3": ", 0", "EEDI3H": ", 0"}.get(name, "")
    try:
        env.eval(f"avszip_{name}({args}{extra})")
    except AvsError as e:
        assert str(e).startswith(f"avszip_{name}:"), f"avszip_{name} raised another function's error: {e}"
