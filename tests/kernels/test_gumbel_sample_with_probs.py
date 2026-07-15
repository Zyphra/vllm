# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.v1.worker.gpu.sample.gumbel import (
    gumbel_sample,
    gumbel_sample_with_probs,
    tidar_sample_bonus_tokens,
    tidar_target_stats,
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

    prob_token_ids = (expected_tokens + 1) % vocab_size
    expected_prob_token_probs = torch.softmax(scaled_logits, dim=-1).gather(
        1, prob_token_ids.unsqueeze(1)
    ).squeeze(1)
    tokens, prob_token_probs, logsumexp = gumbel_sample_with_probs(
        logits,
        idx_mapping,
        temperatures,
        seeds,
        positions,
        apply_temperature=True,
        prob_token_ids=prob_token_ids,
    )

    assert torch.equal(tokens, expected_tokens)
    assert torch.allclose(logsumexp, expected_logsumexp, atol=2e-6, rtol=1e-6)
    assert torch.allclose(
        prob_token_probs, expected_prob_token_probs, atol=1e-7, rtol=2e-6
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("separate_logprob_logits", [False, True])
def test_tidar_target_stats(separate_logprob_logits: bool) -> None:
    generator = torch.Generator(device="cuda").manual_seed(1234)
    num_rows = 37
    vocab_size = 32001
    target_logits = torch.randn(
        num_rows,
        vocab_size,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    prob_token_ids = torch.randint(
        vocab_size,
        (num_rows,),
        dtype=torch.int64,
        device="cuda",
        generator=generator,
    )
    logprob_logits = None
    expected_logprob_logits = target_logits
    if separate_logprob_logits:
        logprob_logits = torch.randn(
            target_logits.shape,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        expected_logprob_logits = logprob_logits

    selected_probs, target_lse, logprob_lse, top1_ids = tidar_target_stats(
        target_logits,
        prob_token_ids,
        logprob_logits,
    )
    expected_target_lse = torch.logsumexp(target_logits.float(), dim=-1)
    expected_selected = torch.exp(
        target_logits.gather(1, prob_token_ids[:, None]).squeeze(1)
        - expected_target_lse
    )
    expected_logprob_lse = torch.logsumexp(
        expected_logprob_logits.float(), dim=-1
    )

    torch.testing.assert_close(target_lse, expected_target_lse)
    torch.testing.assert_close(selected_probs, expected_selected)
    torch.testing.assert_close(logprob_lse, expected_logprob_lse)
    assert torch.equal(top1_ids, expected_logprob_logits.argmax(dim=-1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
def test_tidar_samples_only_bonus_rows() -> None:
    generator = torch.Generator(device="cuda").manual_seed(1234)
    vocab_size = 32001
    cu_num_logits = torch.tensor([0, 4, 6, 10], dtype=torch.int32, device="cuda")
    num_rows = int(cu_num_logits[-1])
    logits = torch.randn(
        num_rows,
        vocab_size,
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )
    idx_mapping = torch.tensor(
        [2, 2, 2, 2, 0, 0, 3, 3, 3, 3], dtype=torch.int64, device="cuda"
    )
    temperatures = torch.ones(4, dtype=torch.float32, device="cuda")
    seeds = torch.tensor([101, 102, 103, 104], dtype=torch.int64, device="cuda")
    positions = torch.arange(200, 210, dtype=torch.int64, device="cuda")

    expected_all, _, _ = gumbel_sample_with_probs(
        logits,
        idx_mapping,
        temperatures,
        seeds,
        positions,
        apply_temperature=False,
        prob_token_ids=torch.zeros(num_rows, dtype=torch.int64, device="cuda"),
    )
    bonus_rows = cu_num_logits[1:].to(torch.int64) - 1
    actual = tidar_sample_bonus_tokens(
        logits,
        cu_num_logits,
        idx_mapping,
        seeds,
        positions,
    )

    assert torch.equal(actual, expected_all[bonus_rows])
