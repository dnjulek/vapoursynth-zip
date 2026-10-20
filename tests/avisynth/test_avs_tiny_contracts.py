"""Native crashes and scalar-tail mistakes must not terminate pytest itself."""

import pytest

from tiny_contracts import TINY_PARAMETERS, check_tiny


@pytest.mark.parametrize("case", TINY_PARAMETERS)
def test_minimum_geometry_preserves_constant_or_zero_mask(case):
    check_tiny("avs", case)
