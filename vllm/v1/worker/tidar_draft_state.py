# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable, Sequence

import torch


def discard_tidar_draft_state(
    probs_by_req_id: dict[str, torch.Tensor],
    logits_by_req_id: dict[str, torch.Tensor],
    token_probs_by_req_id: dict[str, torch.Tensor],
    logsumexp_by_req_id: dict[str, torch.Tensor],
    req_ids: Iterable[str],
) -> None:
    for req_id in req_ids:
        probs_by_req_id.pop(req_id, None)
        logits_by_req_id.pop(req_id, None)
        token_probs_by_req_id.pop(req_id, None)
        logsumexp_by_req_id.pop(req_id, None)


def stash_tidar_draft_state(
    probs_by_req_id: dict[str, torch.Tensor],
    logits_by_req_id: dict[str, torch.Tensor],
    token_probs_by_req_id: dict[str, torch.Tensor],
    logsumexp_by_req_id: dict[str, torch.Tensor],
    req_ids: Sequence[str],
    num_spec_tokens: int | None,
    draft_probs: torch.Tensor | None,
    draft_logits: torch.Tensor | None,
    draft_token_probs: torch.Tensor | None,
    draft_logsumexp: torch.Tensor | None,
) -> None:
    """Replace state for this proposal batch without dropping delayed requests."""
    if num_spec_tokens is None:
        discard_tidar_draft_state(
            probs_by_req_id,
            logits_by_req_id,
            token_probs_by_req_id,
            logsumexp_by_req_id,
            req_ids,
        )
        return

    per_req_probs = None
    if draft_probs is not None:
        per_req_probs = draft_probs.view(-1, num_spec_tokens, draft_probs.shape[-1])
    per_req_logits = None
    if draft_logits is not None:
        per_req_logits = draft_logits.view(-1, num_spec_tokens, draft_logits.shape[-1])
    per_req_token_probs = None
    if draft_token_probs is not None:
        per_req_token_probs = draft_token_probs.view(-1, num_spec_tokens)
    per_req_logsumexp = None
    if draft_logsumexp is not None:
        per_req_logsumexp = draft_logsumexp.view(-1, num_spec_tokens)

    for i, req_id in enumerate(req_ids):
        if per_req_probs is not None and i < per_req_probs.shape[0]:
            probs_by_req_id[req_id] = per_req_probs[i]
        else:
            probs_by_req_id.pop(req_id, None)
        if per_req_logits is not None and i < per_req_logits.shape[0]:
            logits_by_req_id[req_id] = per_req_logits[i]
        else:
            logits_by_req_id.pop(req_id, None)
        if per_req_token_probs is not None and i < per_req_token_probs.shape[0]:
            token_probs_by_req_id[req_id] = per_req_token_probs[i]
        else:
            token_probs_by_req_id.pop(req_id, None)
        if per_req_logsumexp is not None and i < per_req_logsumexp.shape[0]:
            logsumexp_by_req_id[req_id] = per_req_logsumexp[i]
        else:
            logsumexp_by_req_id.pop(req_id, None)


def take_tidar_draft_state(
    probs_by_req_id: dict[str, torch.Tensor],
    logits_by_req_id: dict[str, torch.Tensor],
    token_probs_by_req_id: dict[str, torch.Tensor],
    logsumexp_by_req_id: dict[str, torch.Tensor],
    req_ids: Sequence[str],
    num_draft_tokens: Sequence[int],
    *,
    require_probs: bool,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Gather and consume q/logit state for scheduled draft-bearing rows.

    A request can retain draft IDs while another request exhausts the
    scheduler's token budget. Its proposal state must survive until that
    request is scheduled. For stochastic TiDAR, silently losing q would route
    rejection sampling through the Dirac-proposal fallback and make the
    acceptance ratio incorrect, so fail closed instead.
    """
    draft_reqs = [
        (req_id, int(num_draft_tokens[i]))
        for i, req_id in enumerate(req_ids)
        if int(num_draft_tokens[i]) > 0
    ]
    if not draft_reqs:
        return None, None, None, None

    missing_probs = [
        req_id
        for req_id, n in draft_reqs
        if ((probs := probs_by_req_id.get(req_id)) is None or probs.shape[0] < n)
    ]
    missing_token_probs = [
        req_id
        for req_id, n in draft_reqs
        if ((probs := token_probs_by_req_id.get(req_id)) is None or probs.shape[0] < n)
    ]
    missing_logsumexp = [
        req_id
        for req_id, n in draft_reqs
        if ((logsumexp := logsumexp_by_req_id.get(req_id)) is None
            or logsumexp.shape[0] < n)
    ]
    missing_logits = [
        req_id
        for req_id, n in draft_reqs
        if ((logits := logits_by_req_id.get(req_id)) is None or logits.shape[0] < n)
    ]

    has_dense_probs = not missing_probs
    has_compact_probs = (
        not missing_token_probs and not missing_logsumexp and not missing_logits
    )
    if require_probs and not (has_dense_probs or has_compact_probs):
        raise RuntimeError(
            "Stochastic TiDAR requires a complete dense or compact proposal "
            "distribution for every draft-bearing scheduled request, but q "
            "state is missing or short: "
            f"dense={missing_probs}, token={missing_token_probs}, "
            f"logsumexp={missing_logsumexp}, logits={missing_logits}. "
            "Refusing the NO_DRAFT_PROBS/Dirac fallback because it would "
            "compute an incorrect rejection-sampling ratio."
        )

    probs_chunks: list[torch.Tensor] = []
    if has_dense_probs:
        probs_chunks = [probs_by_req_id[req_id][:n] for req_id, n in draft_reqs]
    draft_probs = torch.cat(probs_chunks, dim=0).contiguous() if probs_chunks else None

    logits_chunks: list[torch.Tensor] = []
    if not missing_logits:
        logits_chunks = [logits_by_req_id[req_id][:n] for req_id, n in draft_reqs]
    draft_logits = (
        torch.cat(logits_chunks, dim=0).contiguous() if logits_chunks else None
    )

    token_probs_chunks: list[torch.Tensor] = []
    logsumexp_chunks: list[torch.Tensor] = []
    if has_compact_probs and not has_dense_probs:
        token_probs_chunks = [
            token_probs_by_req_id[req_id][:n] for req_id, n in draft_reqs
        ]
        logsumexp_chunks = [
            logsumexp_by_req_id[req_id][:n] for req_id, n in draft_reqs
        ]
    draft_token_probs = (
        torch.cat(token_probs_chunks).contiguous() if token_probs_chunks else None
    )
    draft_logsumexp = (
        torch.cat(logsumexp_chunks).contiguous() if logsumexp_chunks else None
    )

    discard_tidar_draft_state(
        probs_by_req_id,
        logits_by_req_id,
        token_probs_by_req_id,
        logsumexp_by_req_id,
        (req_id for req_id, _ in draft_reqs),
    )
    return draft_probs, draft_logits, draft_token_probs, draft_logsumexp
