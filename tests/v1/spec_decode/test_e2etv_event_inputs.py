import pytest
import torch

from vllm.v1.spec_decode.e2etv_event_inputs import (
    discard_event_state,
    stash_event_state,
    take_event_state,
)


def test_event_state_survives_delay_and_flattens_scheduled_prefixes() -> None:
    state = {}
    hidden = torch.arange(2 * 4 * 3).view(8, 3)
    previous = torch.arange(8).view(2, 4)
    positions = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]])
    stash_event_state(state, ["a", "b"], 4, hidden, previous, positions)

    first = take_event_state(state, ["a", "b"], [0, 2])
    assert first is not None
    assert first.req_ids == ("b",)
    assert first.num_draft_tokens == (2,)
    torch.testing.assert_close(first.draft_hidden, hidden.view(2, 4, 3)[1, :2])
    torch.testing.assert_close(first.previous_token_ids, previous[1, :2])
    torch.testing.assert_close(first.global_positions, positions[1, :2])
    assert set(state) == {"a"}

    second = take_event_state(state, ["a"], [3])
    assert second is not None
    assert second.req_ids == ("a",)
    torch.testing.assert_close(second.draft_hidden, hidden.view(2, 4, 3)[0, :3])
    assert not state


def test_event_state_fails_closed_when_alignment_is_missing() -> None:
    with pytest.raises(RuntimeError, match="missing or short"):
        take_event_state({}, ["missing"], [1])


def test_discard_event_state_removes_only_named_requests() -> None:
    state = {}
    stash_event_state(
        state,
        ["a", "b"],
        1,
        torch.zeros(2, 3),
        torch.zeros(2, 1, dtype=torch.long),
        torch.zeros(2, 1, dtype=torch.long),
    )
    discard_event_state(state, ["b", "unknown"])
    assert set(state) == {"a"}
