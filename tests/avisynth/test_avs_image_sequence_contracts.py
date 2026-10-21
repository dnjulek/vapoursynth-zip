"""Native alpha samples and invalid sequences beyond the format matrix."""

import numpy as np
import pytest

import harness as H
from helpers import write_png
from sequence_contracts import BAD_SEQUENCES, rejected_sequence


@pytest.mark.parametrize("kind", BAD_SEQUENCES)
@pytest.mark.parametrize("eager", [False, True], ids=["lazy", "validate"])
def test_incompatible_or_failed_decode_is_catchable(tmp_path, kind, eager):
    rejected_sequence(tmp_path, "avs", kind, eager)


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("width", [1, 17])
def test_gray_alpha_property_contains_exact_pixels(env, tmp_path, bits, width):
    peak = (1 << bits) - 1
    y, x = np.mgrid[:3, :width]
    gray, alpha = (x * 37 + y * 13) % (peak+1), (x * 173 + y * 113) % (peak+1)
    rows = [[(int(gray[j,i]), int(alpha[j,i])) for i in range(width)] for j in range(3)]
    path = write_png(tmp_path / "gray-alpha.png", rows, color="graya", bitdepth=bits)
    a = env.invoke("avszip_ImageRead", path=[str(path)])
    v = H.vszip().ImageRead(str(path))
    expected = alpha.astype(np.uint8 if bits == 8 else np.uint16)
    # The older test checked only '<v>', so a blank/wrong _Alpha passed it.
    for _ in range(3):
        np.testing.assert_array_equal(a.gray_frame_prop("_Alpha"), expected)
    np.testing.assert_array_equal(np.asarray(v.get_frame(0).props["_Alpha"][0]), expected)
    np.testing.assert_array_equal(a.frame(0).planes[0], gray.astype(expected.dtype))
