# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.v1.worker.gpu.sample.logprob import (
    compute_top1_logprobs,
    compute_topk_logprobs,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_compute_top1_logprobs(dtype: torch.dtype) -> None:
    batch_size = 17
    vocab_size = 32001
    generator = torch.Generator(device="cuda").manual_seed(1234)
    logits = torch.randn(
        batch_size,
        vocab_size,
        dtype=dtype,
        device="cuda",
        generator=generator,
    )
    sampled_token_ids = torch.randint(
        vocab_size,
        (batch_size,),
        dtype=torch.int64,
        device="cuda",
        generator=generator,
    )

    output = compute_top1_logprobs(logits, sampled_token_ids)
    expected_logprobs = torch.log_softmax(logits.float(), dim=-1)
    expected_top1 = logits.argmax(dim=-1)
    expected_ids = torch.stack((sampled_token_ids, expected_top1), dim=-1)
    expected_values = expected_logprobs.gather(1, expected_ids)
    sampled_logits = logits.gather(1, sampled_token_ids[:, None])
    expected_ranks = (logits >= sampled_logits).sum(dim=-1)

    assert torch.equal(output.logprob_token_ids, expected_ids)
    torch.testing.assert_close(output.logprobs, expected_values)
    assert torch.equal(output.selected_token_ranks, expected_ranks)


def test_compute_topk_logprobs_uses_top1_fast_path() -> None:
    logits = torch.randn(4, 1025, dtype=torch.float32, device="cuda")
    sampled_token_ids = torch.tensor([0, 1, 2, 3], device="cuda")

    expected = compute_top1_logprobs(logits, sampled_token_ids)
    actual = compute_topk_logprobs(logits, 1, sampled_token_ids)

    assert torch.equal(actual.logprob_token_ids, expected.logprob_token_ids)
    assert torch.equal(actual.logprobs, expected.logprobs)
    assert torch.equal(actual.selected_token_ranks, expected.selected_token_ranks)
