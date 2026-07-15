# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.rejection_sample import (
    get_rejection_logprob_token_ids,
    stochastic_rejection_sample,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


def test_stochastic_rejection_sample_compact_draft_state() -> None:
    device = "cuda"
    num_speculative_steps = 3
    max_num_reqs = 4
    vocab_size = 17
    draft_temperature = 0.6

    idx_mapping = torch.tensor([2, 0], dtype=torch.int64, device=device)
    cu_num_logits = torch.tensor([0, 4, 5], dtype=torch.int32, device=device)
    positions = torch.tensor(
        [100, 101, 102, 103, 200], dtype=torch.int64, device=device
    )
    input_ids = torch.tensor([9, 1, 2, 3, 8], dtype=torch.int64, device=device)
    seeds = torch.tensor([31, 41, 51, 61], dtype=torch.int64, device=device)

    generator = torch.Generator(device=device).manual_seed(7)
    draft_logits = torch.randn(
        2,
        num_speculative_steps,
        vocab_size,
        generator=generator,
        device=device,
    )
    scaled_draft_logits = draft_logits / draft_temperature
    draft_logsumexp = torch.logsumexp(scaled_draft_logits, dim=-1)
    draft_token_probs = torch.zeros(
        2,
        num_speculative_steps,
        dtype=torch.float32,
        device=device,
    )
    draft_ids = input_ids[1:4]
    draft_token_probs[0] = torch.softmax(
        scaled_draft_logits[0], dim=-1
    ).gather(1, draft_ids[:, None]).squeeze(1)

    target_logits = torch.randn(
        5, vocab_size, generator=generator, device=device
    )
    target_logits[:3] = scaled_draft_logits[0]
    target_logsumexp = torch.logsumexp(target_logits, dim=-1)
    target_token_probs = torch.zeros(5, dtype=torch.float32, device=device)
    target_token_probs[:3] = draft_token_probs[0]
    bonus_sampled = torch.tensor([7, 6], dtype=torch.int64, device=device)
    draft_batch_indices = torch.tensor(
        [1, -1, 0, -1], dtype=torch.int64, device=device
    )

    sampled, num_sampled = stochastic_rejection_sample(
        bonus_sampled,
        input_ids,
        cu_num_logits,
        idx_mapping,
        draft_batch_indices,
        positions,
        seeds,
        num_speculative_steps,
        draft_token_probs,
        draft_logits,
        draft_logsumexp,
        draft_temperature,
        target_token_probs,
        target_logits,
        target_logsumexp,
    )

    assert sampled.cpu().tolist() == [[1, 2, 3, 7], [6, -1, -1, -1]]
    assert num_sampled.cpu().tolist() == [4, 1]
    logprob_token_ids = get_rejection_logprob_token_ids(
        sampled, num_sampled, cu_num_logits, total_num_logits=5
    )
    assert logprob_token_ids.cpu().tolist() == [1, 2, 3, 7, 6]


def test_stochastic_rejection_sample_recovers_from_residual() -> None:
    device = "cuda"
    num_speculative_steps = 3
    vocab_size = 17
    draft_temperature = 0.6

    draft_logits = torch.full(
        (1, num_speculative_steps, vocab_size),
        float("-inf"),
        device=device,
    )
    draft_logits[:, :, 0] = 0.0
    draft_logsumexp = torch.zeros((1, num_speculative_steps), device=device)
    draft_token_probs = torch.ones((1, num_speculative_steps), device=device)
    target_logits = torch.full((2, vocab_size), float("-inf"), device=device)
    target_logits[0, 1] = 0.0
    target_logits[1, 4] = 0.0

    sampled, num_sampled = stochastic_rejection_sample(
        bonus_sampled=torch.tensor([4], dtype=torch.int64, device=device),
        input_ids=torch.tensor([9, 0], dtype=torch.int64, device=device),
        cu_num_logits=torch.tensor([0, 2], dtype=torch.int32, device=device),
        idx_mapping=torch.tensor([0], dtype=torch.int64, device=device),
        draft_batch_indices=torch.tensor([0], dtype=torch.int64, device=device),
        positions=torch.tensor([300, 301], dtype=torch.int64, device=device),
        seeds=torch.tensor([31], dtype=torch.int64, device=device),
        num_speculative_steps=num_speculative_steps,
        draft_token_probs=draft_token_probs,
        draft_logits=draft_logits,
        draft_logsumexp=draft_logsumexp,
        draft_temperature=draft_temperature,
        target_token_probs=torch.zeros(2, device=device),
        target_logits=target_logits,
        target_logsumexp=torch.logsumexp(target_logits, dim=-1),
    )

    assert sampled[0, 0].item() == 1
    assert num_sampled.item() == 1
