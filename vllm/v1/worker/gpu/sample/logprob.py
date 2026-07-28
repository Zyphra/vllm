# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.outputs import LogprobsTensors


@triton.jit
def _topk_log_softmax_kernel(
    output_ptr,
    logits_ptr,
    logits_stride,
    topk_ids_ptr,
    topk,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    PADDED_TOPK: tl.constexpr,
):
    req_idx = tl.program_id(0)
    row_ptr = logits_ptr + req_idx * logits_stride

    max_val = float("-inf")
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(row_ptr + block, mask=block < vocab_size, other=float("-inf"))
        max_val = tl.max(tl.maximum(logits, max_val))
    max_val = max_val.to(tl.float32)  # type: ignore

    se = 0.0
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(row_ptr + block, mask=block < vocab_size, other=0.0)
        # NOTE(woosuk): Make sure that logits and all following operations use FP32.
        logits = logits.to(tl.float32)
        e = tl.exp(logits - max_val)
        e = tl.where(block < vocab_size, e, 0.0)
        se += tl.sum(e)
    lse = tl.log(se)

    k_offset = tl.arange(0, PADDED_TOPK)
    k_mask = k_offset < topk
    topk_ids = tl.load(topk_ids_ptr + req_idx * topk + k_offset, mask=k_mask, other=0)

    logits = tl.load(row_ptr + topk_ids, mask=k_mask)
    logits = logits.to(tl.float32)
    o = logits - max_val - lse
    tl.store(output_ptr + req_idx * topk + k_offset, o, mask=k_mask)


@triton.jit
def _ranks_kernel(
    output_ptr,
    logits_ptr,
    logits_stride,
    token_ids_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    row_ptr = logits_ptr + req_idx * logits_stride

    token_id = tl.load(token_ids_ptr + req_idx)
    x = tl.load(row_ptr + token_id)

    n = 0
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(row_ptr + block, mask=block < vocab_size, other=float("-inf"))
        n += tl.sum((logits >= x).to(tl.int32))
    tl.store(output_ptr + req_idx, n)


def compute_token_logprobs(
    logits: torch.Tensor, token_ids: torch.Tensor
) -> torch.Tensor:
    batch_size, vocab_size = logits.shape
    token_ids = token_ids.to(torch.int64)
    num_logprobs = token_ids.shape[1]
    logprobs = logits.new_empty((batch_size, num_logprobs), dtype=torch.float32)
    _topk_log_softmax_kernel[(batch_size,)](
        logprobs,
        logits,
        logits.stride(0),
        token_ids,
        num_logprobs,
        vocab_size,
        BLOCK_SIZE=1024,  # type: ignore
        PADDED_TOPK=triton.next_power_of_2(num_logprobs),
    )
    return logprobs


@triton.jit
def _top1_logprobs_kernel(
    logprob_token_ids_ptr,
    logprobs_ptr,
    token_ranks_ptr,
    logits_ptr,
    logits_stride,
    sampled_token_ids_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    row_ptr = logits_ptr + req_idx * logits_stride
    sampled_token_id = tl.load(sampled_token_ids_ptr + req_idx)
    sampled_logit = tl.load(row_ptr + sampled_token_id).to(tl.float32)

    max_val = float("-inf")
    max_token_id = 0
    rank = 0
    num_pos_inf = 0
    num_nan = 0
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < vocab_size
        logits = tl.load(
            row_ptr + block, mask=mask, other=float("-inf")
        ).to(tl.float32)
        block_max = tl.max(logits, axis=0)
        use_block = block_max > max_val
        max_token_id = tl.where(
            use_block, i + tl.argmax(logits, axis=0), max_token_id
        )
        max_val = tl.maximum(max_val, block_max)
        rank += tl.sum((mask & (logits >= sampled_logit)).to(tl.int32))
        num_pos_inf += tl.sum((mask & (logits == float("inf"))).to(tl.int32))
        num_nan += tl.sum((mask & (logits != logits)).to(tl.int32))

    se = 0.0
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < vocab_size
        logits = tl.load(row_ptr + block, mask=mask, other=0.0).to(tl.float32)
        se += tl.sum(tl.where(mask, tl.exp(logits - max_val), 0.0))
    log_normalizer = max_val + tl.log(se)
    sampled_logprob = sampled_logit - log_normalizer
    top1_logprob = max_val - log_normalizer

    # A positive-infinity logit wins the sampler deterministically, but the
    # usual log-softmax expression evaluates inf - inf to NaN. Preserve
    # fail-closed behavior for NaNs and for an unexpectedly sampled finite
    # token; only canonicalize a clean sampled +inf row.
    clean_pos_inf_row = (num_pos_inf > 0) & (num_nan == 0)
    sampled_logprob = tl.where(
        clean_pos_inf_row,
        tl.where(sampled_logit == float("inf"), 0.0, float("-inf")),
        sampled_logprob,
    )
    top1_logprob = tl.where(clean_pos_inf_row, 0.0, top1_logprob)
    # Invalid rows fail closed downstream. Encode their subtype in the rank
    # field so the scheduler can report the source without another GPU scan.
    rank = tl.where(
        num_nan > 0,
        -1,
        tl.where(
            max_val == float("-inf"),
            -2,
            tl.where(
                clean_pos_inf_row & (sampled_logit != float("inf")),
                -3,
                rank,
            ),
        ),
    )

    output_offset = req_idx * 2
    tl.store(logprob_token_ids_ptr + output_offset, sampled_token_id)
    tl.store(logprob_token_ids_ptr + output_offset + 1, max_token_id)
    tl.store(logprobs_ptr + output_offset, sampled_logprob)
    tl.store(logprobs_ptr + output_offset + 1, top1_logprob)
    tl.store(token_ranks_ptr + req_idx, rank)


def compute_top1_logprobs(
    logits: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    cu_num_logits: list[int] | None = None,
) -> LogprobsTensors:
    batch_size, vocab_size = logits.shape
    logprob_token_ids = torch.empty(
        (batch_size, 2), dtype=torch.int64, device=logits.device
    )
    logprobs = torch.empty(
        (batch_size, 2), dtype=torch.float32, device=logits.device
    )
    token_ranks = torch.empty(
        batch_size, dtype=torch.int64, device=logits.device
    )
    _top1_logprobs_kernel[(batch_size,)](
        logprob_token_ids,
        logprobs,
        token_ranks,
        logits,
        logits.stride(0),
        sampled_token_ids,
        vocab_size,
        BLOCK_SIZE=1024,
    )
    return LogprobsTensors(
        logprob_token_ids=logprob_token_ids,
        logprobs=logprobs,
        selected_token_ranks=token_ranks,
        cu_num_generated_tokens=cu_num_logits,
    )


@triton.jit
def _top1_logprobs_from_stats_kernel(
    logprob_token_ids_ptr,
    logprobs_ptr,
    token_ranks_ptr,
    logits_ptr,
    logits_stride,
    sampled_token_ids_ptr,
    logsumexp_ptr,
    top1_token_ids_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_ptr = logits_ptr + row_idx * logits_stride
    sampled_token_id = tl.load(sampled_token_ids_ptr + row_idx)
    top1_token_id = tl.load(top1_token_ids_ptr + row_idx)
    sampled_logit = tl.load(row_ptr + sampled_token_id).to(tl.float32)
    top1_logit = tl.load(row_ptr + top1_token_id).to(tl.float32)
    logsumexp = tl.load(logsumexp_ptr + row_idx).to(tl.float32)

    rank = 0
    num_pos_inf = 0
    num_nan = 0
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < vocab_size
        logits = tl.load(
            row_ptr + block, mask=mask, other=float("-inf")
        ).to(tl.float32)
        rank += tl.sum((mask & (logits >= sampled_logit)).to(tl.int32))
        num_pos_inf += tl.sum((mask & (logits == float("inf"))).to(tl.int32))
        num_nan += tl.sum((mask & (logits != logits)).to(tl.int32))

    sampled_logprob = sampled_logit - logsumexp
    top1_logprob = top1_logit - logsumexp
    # Match the standalone top-1 kernel: logsumexp is +inf when a clean row
    # contains +inf, so the ordinary subtraction would otherwise yield NaN.
    clean_pos_inf_row = (
        (num_pos_inf > 0)
        & (num_nan == 0)
        & (logsumexp == float("inf"))
    )
    sampled_logprob = tl.where(
        clean_pos_inf_row,
        tl.where(sampled_logit == float("inf"), 0.0, float("-inf")),
        sampled_logprob,
    )
    top1_logprob = tl.where(
        clean_pos_inf_row,
        tl.where(top1_logit == float("inf"), 0.0, float("-inf")),
        top1_logprob,
    )

    output_offset = row_idx * 2
    tl.store(logprob_token_ids_ptr + output_offset, sampled_token_id)
    tl.store(logprob_token_ids_ptr + output_offset + 1, top1_token_id)
    tl.store(logprobs_ptr + output_offset, sampled_logprob)
    tl.store(logprobs_ptr + output_offset + 1, top1_logprob)
    tl.store(token_ranks_ptr + row_idx, rank)


def compute_top1_logprobs_from_stats(
    logits: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    logsumexp: torch.Tensor,
    top1_token_ids: torch.Tensor,
    cu_num_logits: list[int] | None = None,
) -> LogprobsTensors:
    """Finish top-1 logprobs using reduction statistics already computed."""
    batch_size, vocab_size = logits.shape
    assert sampled_token_ids.shape == (batch_size,)
    assert logsumexp.shape == (batch_size,)
    assert top1_token_ids.shape == (batch_size,)
    logprob_token_ids = torch.empty(
        (batch_size, 2), dtype=torch.int64, device=logits.device
    )
    logprobs = torch.empty(
        (batch_size, 2), dtype=torch.float32, device=logits.device
    )
    token_ranks = torch.empty(
        batch_size, dtype=torch.int64, device=logits.device
    )
    _top1_logprobs_from_stats_kernel[(batch_size,)](
        logprob_token_ids,
        logprobs,
        token_ranks,
        logits,
        logits.stride(0),
        sampled_token_ids,
        logsumexp,
        top1_token_ids,
        vocab_size,
        BLOCK_SIZE=8192,
    )
    return LogprobsTensors(
        logprob_token_ids=logprob_token_ids,
        logprobs=logprobs,
        selected_token_ranks=token_ranks,
        cu_num_generated_tokens=cu_num_logits,
    )


def compute_topk_logprobs(
    logits: torch.Tensor,
    num_logprobs: int,
    sampled_token_ids: torch.Tensor,
    cu_num_logits: list[int] | None = None,
) -> LogprobsTensors:
    assert num_logprobs >= 0
    if num_logprobs == 1:
        return compute_top1_logprobs(logits, sampled_token_ids, cu_num_logits)

    batch_size, vocab_size = logits.shape
    logprob_token_ids = sampled_token_ids.unsqueeze(-1)
    if num_logprobs > 0:
        topk_indices = torch.topk(logits, num_logprobs, dim=-1).indices
        logprob_token_ids = torch.cat((logprob_token_ids, topk_indices), dim=1)

    # NOTE(woosuk): Here, to save GPU memory, we do not materialize the full
    # logprobs tensor. Instead, we only compute and return the logprobs of
    # the topk + 1 tokens.
    logprobs = compute_token_logprobs(logits, logprob_token_ids)
    token_ranks = torch.empty(batch_size, dtype=torch.int64, device=logits.device)
    _ranks_kernel[(batch_size,)](
        token_ranks,
        logits,
        logits.stride(0),
        sampled_token_ids,
        vocab_size,
        BLOCK_SIZE=8192,  # type: ignore
    )
    return LogprobsTensors(
        logprob_token_ids=logprob_token_ids,
        logprobs=logprobs,
        selected_token_ranks=token_ranks,
        cu_num_generated_tokens=cu_num_logits,
    )
