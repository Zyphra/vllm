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
    token_probs_by_req_id = {}
    logsumexp_by_req_id = {}
    delayed_probs = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
    delayed_logits = torch.tensor([[7.0, 2.0, 1.0], [1.0, 3.0, 6.0]])
    probs_by_req_id["delayed"] = delayed_probs
    logits_by_req_id["delayed"] = delayed_logits

    active_probs = torch.tensor([[0.2, 0.5, 0.3], [0.4, 0.4, 0.2]])
    active_logits = torch.tensor([[2.0, 5.0, 3.0], [4.0, 4.0, 2.0]])
    stash_tidar_draft_state(
        probs_by_req_id,
        logits_by_req_id,
        token_probs_by_req_id,
        logsumexp_by_req_id,
        ["active"],
        num_spec_tokens=2,
        draft_probs=active_probs,
        draft_logits=active_logits,
        draft_token_probs=None,
        draft_logsumexp=None,
    )

    # A proposal for the currently scheduled request must not globally clear
    # the q/logit state owned by a request delayed by token-budget exhaustion.
    assert torch.equal(probs_by_req_id["delayed"], delayed_probs)
    assert torch.equal(logits_by_req_id["delayed"], delayed_logits)

    (
        taken_probs,
        taken_logits,
        taken_token_probs,
        taken_logsumexp,
    ) = take_tidar_draft_state(
        probs_by_req_id,
        logits_by_req_id,
        token_probs_by_req_id,
        logsumexp_by_req_id,
        ["delayed"],
        np.array([2], dtype=np.int32),
        require_probs=True,
    )
    assert torch.equal(taken_probs, delayed_probs)
    assert torch.equal(taken_logits, delayed_logits)
    assert taken_token_probs is None
    assert taken_logsumexp is None
    assert "delayed" not in probs_by_req_id
    assert "delayed" not in logits_by_req_id

    # Taking one request's drafts must leave another request's pending state.
    assert torch.equal(probs_by_req_id["active"], active_probs)
    assert torch.equal(logits_by_req_id["active"], active_logits)


def test_stochastic_tidar_fails_closed_when_q_is_missing() -> None:
    probs_by_req_id = {}
    logits_by_req_id = {"missing-q": torch.ones(2, 3)}
    token_probs_by_req_id = {}
    logsumexp_by_req_id = {}

    with pytest.raises(RuntimeError, match="Refusing the NO_DRAFT_PROBS/Dirac"):
        take_tidar_draft_state(
            probs_by_req_id,
            logits_by_req_id,
            token_probs_by_req_id,
            logsumexp_by_req_id,
            ["missing-q"],
            np.array([2], dtype=np.int32),
            require_probs=True,
        )

    # The failed verification attempt must not consume the remaining state.
    assert "missing-q" in logits_by_req_id


def test_tidar_draft_state_is_discarded_on_preemption() -> None:
    probs_by_req_id = {"preempted": torch.ones(2, 3)}
    logits_by_req_id = {"preempted": torch.ones(2, 3)}
    token_probs_by_req_id = {"preempted": torch.ones(2)}
    logsumexp_by_req_id = {"preempted": torch.ones(2)}

    discard_tidar_draft_state(
        probs_by_req_id,
        logits_by_req_id,
        token_probs_by_req_id,
        logsumexp_by_req_id,
        ["preempted"],
    )

    assert "preempted" not in probs_by_req_id
    assert "preempted" not in logits_by_req_id
    assert "preempted" not in token_probs_by_req_id
    assert "preempted" not in logsumexp_by_req_id


def test_compact_tidar_draft_state_survives_scheduler_delay() -> None:
    probs_by_req_id = {}
    expected_logits = torch.tensor([[2.0, 1.0], [1.0, 2.0]])
    expected_token_probs = torch.tensor([0.7, 0.8])
    expected_logsumexp = torch.tensor([2.3, 2.3])
    logits_by_req_id = {"delayed": expected_logits}
    token_probs_by_req_id = {"delayed": expected_token_probs}
    logsumexp_by_req_id = {"delayed": expected_logsumexp}

    # Replacing active's compact state must leave delayed's draft state intact.
    stash_tidar_draft_state(
        probs_by_req_id,
        logits_by_req_id,
        token_probs_by_req_id,
        logsumexp_by_req_id,
        ["active"],
        num_spec_tokens=2,
        draft_probs=None,
        draft_logits=torch.tensor([[3.0, 1.0], [1.0, 3.0]]),
        draft_token_probs=torch.tensor([0.6, 0.9]),
        draft_logsumexp=torch.tensor([3.1, 3.1]),
    )

    taken = take_tidar_draft_state(
        probs_by_req_id,
        logits_by_req_id,
        token_probs_by_req_id,
        logsumexp_by_req_id,
        ["delayed"],
        np.array([2], dtype=np.int32),
        require_probs=True,
    )
    taken_probs, taken_logits, taken_token_probs, taken_logsumexp = taken

    assert taken_probs is None
    assert torch.equal(taken_logits, expected_logits)
    assert torch.equal(taken_token_probs, expected_token_probs)
    assert torch.equal(taken_logsumexp, expected_logsumexp)
    assert "delayed" not in logits_by_req_id
    assert "delayed" not in token_probs_by_req_id
    assert "delayed" not in logsumexp_by_req_id
    assert "active" in logits_by_req_id
    assert "active" in token_probs_by_req_id
    assert "active" in logsumexp_by_req_id
