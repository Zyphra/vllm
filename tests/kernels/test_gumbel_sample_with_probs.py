# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.v1.worker.gpu.sample.gumbel import (
    gumbel_sample,
    gumbel_sample_with_probs,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("batch_size", [1, 32])
def test_gumbel_sample_with_probs_matches_reference(batch_size: int) -> None:
    torch.manual_seed(1234)
    vocab_size = 32001
    temperature = 0.6
    logits = torch.randn(
        batch_size,
        vocab_size,
        dtype=torch.float32,
        device="cuda",
    )
    idx_mapping = torch.arange(batch_size, dtype=torch.int64, device="cuda")
    temperatures = torch.full(
        (batch_size,), temperature, dtype=torch.float32, device="cuda"
    )
    seeds = torch.arange(
        1000, 1000 + batch_size, dtype=torch.int64, device="cuda"
    )
    positions = torch.arange(
        2000, 2000 + batch_size, dtype=torch.int64, device="cuda"
    )

    expected_tokens = gumbel_sample(
        logits,
        idx_mapping,
        temperatures,
        seeds,
        positions,
        apply_temperature=True,
    )
    scaled_logits = logits / temperature
    expected_logsumexp = torch.logsumexp(scaled_logits, dim=-1)
    expected_selected = scaled_logits.gather(
        1, expected_tokens.unsqueeze(1)).squeeze(1)
    expected_probs = torch.exp(expected_selected - expected_logsumexp)

    tokens, selected_probs, logsumexp = gumbel_sample_with_probs(
        logits,
        idx_mapping,
        temperatures,
        seeds,
        positions,
        apply_temperature=True,
    )

    assert torch.equal(tokens, expected_tokens)
    assert torch.allclose(logsumexp, expected_logsumexp, atol=2e-6, rtol=1e-6)
    assert torch.allclose(selected_probs, expected_probs, atol=1e-7, rtol=2e-6)
