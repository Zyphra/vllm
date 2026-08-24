# SPDX-License-Identifier: Apache-2.0
"""Exact DSpARK inputs for same-update TiDAR verifier events.

This module contains only the transient GPU-side proposal state needed by the
default-off online E2E-TV runtime.  It performs no file I/O, sampling, replay,
or optimization.  Callers allocate the request-state mapping only when the
runtime is explicitly enabled.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Sequence

import numpy as np

import torch


@dataclass
class TiDARE2ETVEventBatch:
    """Request-aligned proposal inputs for one verifier invocation."""

    req_ids: tuple[str, ...]
    num_draft_tokens: tuple[int, ...]
    draft_hidden: torch.Tensor
    previous_token_ids: torch.Tensor
    global_positions: torch.Tensor
    # Exact verifier rows consumed by the target output projection.  Proposal
    # inputs exist before the verifier forward, so the model runner attaches
    # these rows after computing the target logits and before rejection
    # sampling consumes the event.
    target_hidden: torch.Tensor | None = None


@dataclass
class _PerRequestEvent:
    draft_hidden: torch.Tensor
    previous_token_ids: torch.Tensor
    global_positions: torch.Tensor


def discard_event_state(
    state_by_req_id: dict[str, _PerRequestEvent],
    req_ids: Iterable[str],
) -> None:
    for req_id in req_ids:
        state_by_req_id.pop(req_id, None)


def stash_event_state(
    state_by_req_id: dict[str, _PerRequestEvent],
    req_ids: Sequence[str],
    num_spec_tokens: int,
    draft_hidden: torch.Tensor,
    previous_token_ids: torch.Tensor,
    global_positions: torch.Tensor,
) -> None:
    """Replace proposal state for the requests that just drafted."""

    if num_spec_tokens <= 0:
        raise ValueError("num_spec_tokens must be positive")
    batch_size = len(req_ids)
    expected = batch_size * num_spec_tokens
    if draft_hidden.ndim != 2 or draft_hidden.shape[0] != expected:
        raise ValueError(
            "draft_hidden must be [len(req_ids) * num_spec_tokens, hidden], "
            f"got {tuple(draft_hidden.shape)} for {batch_size=} and "
            f"{num_spec_tokens=}"
        )
    expected_grid = (batch_size, num_spec_tokens)
    if tuple(previous_token_ids.shape) != expected_grid:
        raise ValueError(
            f"previous_token_ids must be {expected_grid}, got "
            f"{tuple(previous_token_ids.shape)}"
        )
    if tuple(global_positions.shape) != expected_grid:
        raise ValueError(
            f"global_positions must be {expected_grid}, got "
            f"{tuple(global_positions.shape)}"
        )

    hidden = draft_hidden.view(batch_size, num_spec_tokens, -1)
    for index, req_id in enumerate(req_ids):
        state_by_req_id[req_id] = _PerRequestEvent(
            draft_hidden=hidden[index].detach(),
            previous_token_ids=previous_token_ids[index].detach(),
            global_positions=global_positions[index].detach(),
        )


def take_event_state(
    state_by_req_id: dict[str, _PerRequestEvent],
    req_ids: Sequence[str],
    num_draft_tokens: Sequence[int],
) -> TiDARE2ETVEventBatch | None:
    """Consume and flatten state for draft-bearing scheduled requests."""

    draft_reqs = [
        (req_id, int(num_draft_tokens[index]))
        for index, req_id in enumerate(req_ids)
        if int(num_draft_tokens[index]) > 0
    ]
    if not draft_reqs:
        return None

    missing = [
        req_id
        for req_id, count in draft_reqs
        if (
            (event := state_by_req_id.get(req_id)) is None
            or event.draft_hidden.shape[0] < count
            or event.previous_token_ids.shape[0] < count
            or event.global_positions.shape[0] < count
        )
    ]
    if missing:
        raise RuntimeError(
            "E2E-TV runtime is enabled but exact proposal state is missing "
            f"or short for draft-bearing requests: {missing}"
        )

    events = [state_by_req_id[req_id] for req_id, _ in draft_reqs]
    result = TiDARE2ETVEventBatch(
        req_ids=tuple(req_id for req_id, _ in draft_reqs),
        num_draft_tokens=tuple(count for _, count in draft_reqs),
        draft_hidden=torch.cat(
            [event.draft_hidden[:count] for event, (_, count) in zip(events, draft_reqs)]
        ).contiguous(),
        previous_token_ids=torch.cat(
            [
                event.previous_token_ids[:count]
                for event, (_, count) in zip(events, draft_reqs)
            ]
        ).contiguous(),
        global_positions=torch.cat(
            [event.global_positions[:count] for event, (_, count) in zip(events, draft_reqs)]
        ).contiguous(),
    )
    discard_event_state(state_by_req_id, (req_id for req_id, _ in draft_reqs))
    return result


def attach_target_hidden(
    event_batch: TiDARE2ETVEventBatch,
    *,
    sample_hidden_states: torch.Tensor,
    target_logits_indices: torch.Tensor,
) -> TiDARE2ETVEventBatch:
    """Attach the exact pre-projection verifier rows for this event batch."""

    expected = sum(event_batch.num_draft_tokens)
    if event_batch.target_hidden is not None:
        raise RuntimeError("E2E-TV target hidden state was attached more than once")
    if sample_hidden_states.ndim != 2:
        raise ValueError(
            "sample_hidden_states must be [num_sampled_positions, hidden], "
            f"got {tuple(sample_hidden_states.shape)}"
        )
    if target_logits_indices.ndim != 1 or target_logits_indices.numel() != expected:
        raise ValueError(
            "target_logits_indices must contain one row per draft token; "
            f"expected {expected}, got {tuple(target_logits_indices.shape)}"
        )
    indices = target_logits_indices.to(torch.long)
    if indices.numel():
        minimum = int(indices.min().item())
        maximum = int(indices.max().item())
        if minimum < 0 or maximum >= sample_hidden_states.shape[0]:
            raise ValueError(
                "target_logits_indices are outside sample_hidden_states: "
                f"min={minimum}, max={maximum}, rows={sample_hidden_states.shape[0]}"
            )
    target_hidden = sample_hidden_states[indices]
    if target_hidden.shape[0] != expected:
        raise RuntimeError("attached E2E-TV target hidden state has wrong token count")
    return replace(event_batch, target_hidden=target_hidden.detach())


def target_logits_indices_from_cu_num_logits(
    cu_num_logits: Sequence[int] | np.ndarray,
) -> torch.Tensor:
    """Return verifier rows, excluding each request's bonus-token row.

    V2 lays sampled hidden states out request-by-request as
    ``[draft verifier rows..., bonus row]``.  This helper derives the exact
    rows consumed by rejection sampling without relying on the V1 metadata
    builder.  It intentionally returns a CPU tensor; callers move the tiny
    index vector to the hidden-state device only when capture is enabled.
    """

    cumulative = np.asarray(cu_num_logits, dtype=np.int64)
    if cumulative.ndim != 1 or cumulative.size < 2:
        raise ValueError("cu_num_logits must be a one-dimensional prefix sum")
    if cumulative[0] != 0 or np.any(cumulative[1:] < cumulative[:-1]):
        raise ValueError("cu_num_logits must be a nondecreasing prefix sum from 0")

    indices: list[int] = []
    for start, stop in zip(cumulative[:-1], cumulative[1:], strict=True):
        width = int(stop - start)
        if width < 1:
            raise ValueError("each request must own at least one bonus-token row")
        indices.extend(range(int(start), int(stop) - 1))
    return torch.tensor(indices, dtype=torch.int64)


__all__ = (
    "TiDARE2ETVEventBatch",
    "attach_target_hidden",
    "discard_event_state",
    "stash_event_state",
    "target_logits_indices_from_cu_num_logits",
    "take_event_state",
)
