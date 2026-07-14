# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import pytest
import torch

from vllm.v1.worker.tidar_draft_state import (
    discard_tidar_draft_state,
    stash_tidar_draft_state,
    take_tidar_draft_state,
)


def test_tidar_draft_state_survives_scheduler_delay() -> None:
    probs_by_req_id = {}
    logits_by_req_id = {}
    delayed_probs = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
    delayed_logits = torch.tensor([[7.0, 2.0, 1.0], [1.0, 3.0, 6.0]])
    probs_by_req_id["delayed"] = delayed_probs
    logits_by_req_id["delayed"] = delayed_logits

    active_probs = torch.tensor([[0.2, 0.5, 0.3], [0.4, 0.4, 0.2]])
    active_logits = torch.tensor([[2.0, 5.0, 3.0], [4.0, 4.0, 2.0]])
    stash_tidar_draft_state(
        probs_by_req_id,
        logits_by_req_id,
        ["active"],
        num_spec_tokens=2,
        draft_probs=active_probs,
        draft_logits=active_logits,
    )

    # A proposal for the currently scheduled request must not globally clear
    # the q/logit state owned by a request delayed by token-budget exhaustion.
    assert torch.equal(probs_by_req_id["delayed"], delayed_probs)
    assert torch.equal(logits_by_req_id["delayed"], delayed_logits)

    taken_probs, taken_logits = take_tidar_draft_state(
        probs_by_req_id,
        logits_by_req_id,
        ["delayed"],
        np.array([2], dtype=np.int32),
        require_probs=True,
    )
    assert torch.equal(taken_probs, delayed_probs)
    assert torch.equal(taken_logits, delayed_logits)
    assert "delayed" not in probs_by_req_id
    assert "delayed" not in logits_by_req_id

    # Taking one request's drafts must leave another request's pending state.
    assert torch.equal(probs_by_req_id["active"], active_probs)
    assert torch.equal(logits_by_req_id["active"], active_logits)


def test_stochastic_tidar_fails_closed_when_q_is_missing() -> None:
    probs_by_req_id = {}
    logits_by_req_id = {"missing-q": torch.ones(2, 3)}

    with pytest.raises(RuntimeError, match="Refusing the NO_DRAFT_PROBS/Dirac"):
        take_tidar_draft_state(
            probs_by_req_id,
            logits_by_req_id,
            ["missing-q"],
            np.array([2], dtype=np.int32),
            require_probs=True,
        )

    # The failed verification attempt must not consume the remaining state.
    assert "missing-q" in logits_by_req_id


def test_tidar_draft_state_is_discarded_on_preemption() -> None:
    probs_by_req_id = {"preempted": torch.ones(2, 3)}
    logits_by_req_id = {"preempted": torch.ones(2, 3)}

    discard_tidar_draft_state(
        probs_by_req_id,
        logits_by_req_id,
        ["preempted"],
    )

    assert "preempted" not in probs_by_req_id
    assert "preempted" not in logits_by_req_id
