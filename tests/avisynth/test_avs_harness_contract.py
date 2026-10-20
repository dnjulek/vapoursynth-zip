"""Sensitivity checks: a permissive comparator can hide real plugin defects."""

import numpy as np
import pytest

import avsrt
import harness as H


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("side", [0, 1])
def test_tolerant_comparison_rejects_nonfinite_sample(bad, side):
    frames = [avsrt.Frame([np.zeros((3, 17), np.float32)], {}) for _ in range(2)]
    frames[side].planes[0][1, 8] = bad
    with pytest.raises(AssertionError, match="nonfinite"):
        H.compare_planes(*frames, tol=0.01)


@pytest.mark.parametrize("side", [0, 1])
def test_comparison_requires_matching_plane_counts(side):
    frames = [avsrt.Frame([np.zeros((2, 3), np.uint8)], {}) for _ in range(2)]
    frames[side].planes.append(np.zeros((2, 3), np.uint8))
    with pytest.raises(AssertionError, match="plane count"):
        H.compare_planes(*frames)
    H.compare_planes(*frames, planes=[0])  # explicit alpha-free comparison


def test_equal_infinite_metric_is_valid_but_opposite_sign_is_not():
    a, b = (avsrt.Frame([], {"XPSNR_Y": float("inf")}) for _ in range(2))
    H.compare_props(a, b, ["XPSNR_Y"])
    b.props["XPSNR_Y"] = -float("inf")
    with pytest.raises(AssertionError):
        H.compare_props(a, b, ["XPSNR_Y"])


def test_typed_invoke_matches_script_and_keeps_inputs_owned(env):
    clip = env.eval('BlankClip(width=32,height=16,length=1,pixel_type="Y16",color_yuv=$808080)')
    env.set_clip("src", clip)
    direct = env.invoke("vszip_Limiter", clip, min=[20000.0], max=[30000.0])
    script = env.eval("src.vszip_Limiter(min=[20000.0],max=[30000.0])")
    H.compare_planes(direct.frame(0), script.frame(0))
    before = clip.frame(0)
    with pytest.raises(avsrt.AvsError, match="finite"):
        env.invoke("vszip_Limiter", clip, min=[float("nan")], max=[30000.0])
    H.compare_planes(clip.frame(0), before)


def test_typed_string_arguments_remain_valid_after_python_collection(env):
    """Smoke test: a string passed through the typed invoke reaches the host
    intact after GC and allocation churn. It cannot prove the strings are
    environment-owned (propSet copies its key, so a Python-owned buffer would
    pass too); that is guaranteed structurally by avsrt using avs_save_string
    with a c_void_p result."""
    import gc
    clip = env.eval('BlankClip(width=17,height=3,length=1,pixel_type="Y8")')
    key = "Audit" + "retained_argument_" * 17
    tagged = env.invoke("propSet", clip, key, 731)
    gc.collect()
    junk = [("reused allocation " * 23).encode() for _ in range(2000)]
    assert tagged.frame(0).props[key] == 731
    assert junk
