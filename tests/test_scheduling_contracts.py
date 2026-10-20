"""Temporal and pooled-scratch filters must tolerate real concurrent requests."""

import pytest

from contract_oracles import VSHost
from scheduling_contracts import SCHEDULE_CASES, check_schedule


@pytest.mark.parametrize("case", SCHEDULE_CASES, ids=[c[0] for c in SCHEDULE_CASES])
def test_cold_graph_concurrent_repeated_and_boundary_requests(core, case):
    check_schedule(VSHost(core), case)
