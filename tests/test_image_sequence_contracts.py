"""Every decoded frame must match its clip's fixed layout, even after validate."""

import pytest

from sequence_contracts import BAD_SEQUENCES, rejected_sequence


@pytest.mark.parametrize("kind", BAD_SEQUENCES)
@pytest.mark.parametrize("eager", [False, True], ids=["lazy", "validate"])
def test_incompatible_or_failed_decode_is_catchable(tmp_path, kind, eager):
    rejected_sequence(tmp_path, "vs", kind, eager)
