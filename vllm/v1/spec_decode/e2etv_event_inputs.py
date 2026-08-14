# SPDX-License-Identifier: Apache-2.0
"""Exact DSpARK inputs for same-update TiDAR verifier events.

This module contains only the transient GPU-side proposal state needed by the
default-off online E2E-TV runtime.  It performs no file I/O, sampling, replay,
or optimization.  Callers allocate the request-state mapping only when the
runtime is explicitly enabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch


@dataclass
class TiDARE2ETVEventBatch:
    """Request-aligned proposal inputs for one verifier invocation."""

    req_ids: tuple[str, ...]
    num_draft_tokens: tuple[int, ...]
    draft_hidden: torch.Tensor
    previous_token_ids: torch.Tensor
    global_positions: torch.Tensor


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


__all__ = (
    "TiDARE2ETVEventBatch",
    "discard_event_state",
    "stash_event_state",
    "take_event_state",
)
