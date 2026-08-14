# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from vllm.v1.spec_decode.e2etv_event_inputs import TiDARE2ETVEventBatch


@dataclass
class SpecDecodeMetadata:
    # [num_tokens]
    draft_token_ids: torch.Tensor
    # [batch_size]
    num_draft_tokens: list[int]
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor
    # [batch_size]
    cu_num_sampled_tokens: torch.Tensor
    # [num_tokens]
    target_logits_indices: torch.Tensor
    # [batch_size]
    bonus_logits_indices: torch.Tensor
    # [num_tokens + batch_size]
    logits_indices: torch.Tensor
    # [num_tokens, vocab_size] — full draft probability distribution per
    # draft position. None when the drafter is a Dirac (e.g. argmax /
    # ngram), which lets the rejection sampler stay on its NO_DRAFT_PROBS
    # branch (acceptance = p_AR(v*), residual = p_AR masked excluding v*).
    draft_probs: Optional[torch.Tensor] = None
    # [num_tokens] -- q(draft_token) for compact stochastic drafters. This
    # avoids retaining a full [num_tokens, vocab_size] draft-probability
    # tensor while still providing the exact acceptance ratio p(x) / q(x).
    draft_token_probs: Optional[torch.Tensor] = None
    # [num_tokens, vocab_size] — raw drafter logits per draft position.
    # When set, downstream consumers should use these directly for
    # mix-logit (mixed = a*target + (1-a)*draft) instead of
    # softmax→log roundtripping through draft_probs.
    draft_logits: Optional[torch.Tensor] = None
    # [num_tokens] -- logsumexp(draft_logits / draft_temperature). Together
    # with draft_logits this reconstructs q(v) only for residual sampling.
    draft_logsumexp: Optional[torch.Tensor] = None
    # Temperature used to sample the compact draft distribution.
    draft_temperature: Optional[float] = None
    # Default-off qualification payload carrying the detached DSpARK head
    # inputs that produced these draft tokens.  It is consumed only by the
    # bounded E2E-TV event writer in the rejection sampler.
    e2etv_event_batch: Optional[TiDARE2ETVEventBatch] = None

    def __post_init__(self):
        self.max_spec_len = max(self.num_draft_tokens)

    @classmethod
    def make_dummy(
        cls,
        draft_token_ids: list[list[int]],
        device: torch.device,
    ) -> "SpecDecodeMetadata":
        batch_size = len(draft_token_ids)
        num_draft_tokens = [len(ids) for ids in draft_token_ids]
        num_sampled_tokens = [len(ids) + 1 for ids in draft_token_ids]
        flattened_draft_token_ids = sum(draft_token_ids, [])
        num_tokens = len(flattened_draft_token_ids)

        draft_token_ids_tensor = torch.tensor(
            flattened_draft_token_ids, dtype=torch.int32, device=device
        )
        cu_num_draft_tokens = np.cumsum(num_draft_tokens, dtype=np.int32)
        cu_num_draft_tokens_tensor = torch.from_numpy(cu_num_draft_tokens).to(device)
        cu_num_sampled_tokens = np.cumsum(num_sampled_tokens, dtype=np.int32)
        cu_num_sampled_tokens_tensor = torch.from_numpy(cu_num_sampled_tokens).to(
            device
        )

        target_logits_indices = torch.zeros(
            num_tokens, dtype=torch.int32, device=device
        )
        bonus_logits_indices = torch.zeros(batch_size, dtype=torch.int32, device=device)
        logits_indices = torch.zeros(
            num_tokens + batch_size, dtype=torch.int32, device=device
        )
        return cls(
            draft_token_ids=draft_token_ids_tensor,
            num_draft_tokens=num_draft_tokens,
            cu_num_draft_tokens=cu_num_draft_tokens_tensor,
            cu_num_sampled_tokens=cu_num_sampled_tokens_tensor,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
        )
