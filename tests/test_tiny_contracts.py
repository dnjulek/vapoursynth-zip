"""A legal minimum-sized constant input must remain valid and deterministic."""

import pytest

from tiny_contracts import TINY_PARAMETERS, check_tiny


@pytest.mark.parametrize("case", TINY_PARAMETERS)
def test_minimum_geometry_preserves_constant_or_zero_mask(case):
    check_tiny("vs", case)
