# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.v1.worker.gpu.sample.logprob import (
    compute_top1_logprobs,
    compute_top1_logprobs_from_stats,
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


def test_compute_top1_logprobs_positive_infinity() -> None:
    logits = torch.tensor(
        [
            [float("inf"), 1.0, 0.0],
            [0.0, float("inf"), float("inf")],
        ],
        dtype=torch.float32,
        device="cuda",
    )
    sampled_token_ids = torch.tensor([0, 1], device="cuda")

    output = compute_top1_logprobs(logits, sampled_token_ids)

    assert torch.equal(
        output.logprob_token_ids,
        torch.tensor([[0, 0], [1, 1]], device="cuda"),
    )
    assert torch.equal(output.logprobs, torch.zeros(2, 2, device="cuda"))
    assert torch.equal(
        output.selected_token_ranks,
        torch.tensor([1, 2], device="cuda"),
    )


def test_compute_top1_logprobs_undefined_rows_remain_nonfinite() -> None:
    logits = torch.tensor(
        [
            [float("inf"), float("nan"), 0.0],
            [float("-inf"), float("-inf"), float("-inf")],
            [0.0, float("inf"), 1.0],
        ],
        dtype=torch.float32,
        device="cuda",
    )
    sampled_token_ids = torch.tensor([0, 0, 0], device="cuda")

    output = compute_top1_logprobs(logits, sampled_token_ids)

    assert not torch.isfinite(output.logprobs[:, 0]).any()
    assert torch.equal(
        output.selected_token_ranks,
        torch.tensor([-1, -2, -3], device="cuda"),
    )


def test_compute_topk_logprobs_uses_top1_fast_path() -> None:
    logits = torch.randn(4, 1025, dtype=torch.float32, device="cuda")
    sampled_token_ids = torch.tensor([0, 1, 2, 3], device="cuda")

    expected = compute_top1_logprobs(logits, sampled_token_ids)
    actual = compute_topk_logprobs(logits, 1, sampled_token_ids)

    assert torch.equal(actual.logprob_token_ids, expected.logprob_token_ids)
    assert torch.equal(actual.logprobs, expected.logprobs)
    assert torch.equal(actual.selected_token_ranks, expected.selected_token_ranks)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_compute_top1_logprobs_from_stats(dtype: torch.dtype) -> None:
    generator = torch.Generator(device="cuda").manual_seed(4321)
    logits = torch.randn(
        33,
        32001,
        dtype=dtype,
        device="cuda",
        generator=generator,
    )
    sampled_token_ids = torch.randint(
        logits.shape[1],
        (logits.shape[0],),
        dtype=torch.int64,
        device="cuda",
        generator=generator,
    )
    logsumexp = torch.logsumexp(logits.float(), dim=-1)
    top1_token_ids = logits.argmax(dim=-1)

    expected = compute_top1_logprobs(logits, sampled_token_ids)
    actual = compute_top1_logprobs_from_stats(
        logits,
        sampled_token_ids,
        logsumexp,
        top1_token_ids,
    )

    assert torch.equal(actual.logprob_token_ids, expected.logprob_token_ids)
    torch.testing.assert_close(actual.logprobs, expected.logprobs)
    assert torch.equal(actual.selected_token_ranks, expected.selected_token_ranks)


def test_compute_top1_logprobs_from_stats_positive_infinity() -> None:
    logits = torch.tensor(
        [
            [float("inf"), 1.0, 0.0],
            [0.0, float("inf"), float("inf")],
        ],
        dtype=torch.float32,
        device="cuda",
    )
    sampled_token_ids = torch.tensor([0, 1], device="cuda")
    logsumexp = torch.logsumexp(logits, dim=-1)
    top1_token_ids = logits.argmax(dim=-1)

    expected = compute_top1_logprobs(logits, sampled_token_ids)
    actual = compute_top1_logprobs_from_stats(
        logits,
        sampled_token_ids,
        logsumexp,
        top1_token_ids,
    )

    assert torch.equal(actual.logprob_token_ids, expected.logprob_token_ids)
    assert torch.equal(actual.logprobs, expected.logprobs)
    assert torch.equal(actual.selected_token_ranks, expected.selected_token_ranks)


def test_compute_top1_logprobs_from_stats_undefined_rows_remain_nonfinite() -> None:
    logits = torch.tensor(
        [
            [float("inf"), float("nan"), 0.0],
            [float("-inf"), float("-inf"), float("-inf")],
            [0.0, float("inf"), 1.0],
        ],
        dtype=torch.float32,
        device="cuda",
    )
    sampled_token_ids = torch.tensor([0, 0, 0], device="cuda")
    logsumexp = torch.logsumexp(logits, dim=-1)
    top1_token_ids = logits.nan_to_num(nan=float("-inf")).argmax(dim=-1)

    output = compute_top1_logprobs_from_stats(
        logits,
        sampled_token_ids,
        logsumexp,
        top1_token_ids,
    )

    assert not torch.isfinite(output.logprobs[:, 0]).any()
