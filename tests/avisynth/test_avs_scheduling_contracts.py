"""Changing frame properties guard dependencies as well as output pixels."""

import pytest

from scheduling_contracts import SCHEDULE_CASES, check_schedule
from test_avs_contract_oracles import AVSHost


@pytest.mark.parametrize("case", SCHEDULE_CASES, ids=[c[0] for c in SCHEDULE_CASES])
def test_cold_graph_prefetch_repeated_and_boundary_requests(env, case):
    check_schedule(AVSHost(env), case)
