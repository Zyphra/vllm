# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _rejection_sample_kernel(
    sampled_ptr,  # [num_reqs, num_speculative_steps + 1]
    sampled_stride,
    num_sampled_ptr,  # [num_reqs]
    target_sampled_ptr,  # [num_draft_tokens + num_reqs]
    input_ids_ptr,  # [num_draft_tokens + num_reqs]
    cu_num_logits_ptr,  # [num_reqs + 1]
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_tokens = end_idx - start_idx

    num_sampled = 0
    rejected = False
    for i in range(num_tokens - 1):
        if not rejected:
            target_sampled = tl.load(target_sampled_ptr + start_idx + i)
            draft_sampled = tl.load(input_ids_ptr + start_idx + i + 1)
            tl.store(sampled_ptr + req_idx * sampled_stride + i, target_sampled)
            num_sampled += 1
            if target_sampled != draft_sampled:
                rejected = True
    if not rejected:
        target_sampled = tl.load(target_sampled_ptr + start_idx + num_tokens - 1)
        tl.store(
            sampled_ptr + req_idx * sampled_stride + num_tokens - 1, target_sampled
        )
        num_sampled += 1
    tl.store(num_sampled_ptr + req_idx, num_sampled)


def rejection_sample(
    # [num_draft_tokens + num_reqs]
    target_sampled: torch.Tensor,
    # [num_draft_tokens + num_reqs]
    input_ids: torch.Tensor,
    # [num_reqs + 1]
    cu_num_logits: torch.Tensor,
    num_speculative_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_reqs = cu_num_logits.shape[0] - 1
    sampled = torch.empty(
        num_reqs,
        num_speculative_steps + 1,
        dtype=target_sampled.dtype,
        device=target_sampled.device,
    )
    num_sampled = torch.empty(
        num_reqs,
        dtype=torch.int32,
        device=target_sampled.device,
    )
    _rejection_sample_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        target_sampled,
        input_ids,
        cu_num_logits,
        num_warps=1,
    )
    return sampled, num_sampled


@triton.jit
def _stochastic_accept_kernel(
    sampled_ptr,
    sampled_stride,
    num_sampled_ptr,
    rejected_pos_ptr,
    target_sampled_ptr,
    input_ids_ptr,
    cu_num_logits_ptr,
    idx_mapping_ptr,
    positions_ptr,
    seeds_ptr,
    draft_token_probs_ptr,
    draft_token_probs_stride,
    target_token_probs_ptr,
    MAX_SPEC_LEN: tl.constexpr,
):
    req_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_draft_tokens = end_idx - start_idx - 1
    seed = tl.load(seeds_ptr + req_state_idx)

    rejected = False
    rejected_pos = -1
    num_sampled = 0
    for pos in range(MAX_SPEC_LEN):
        if pos < num_draft_tokens and not rejected:
            draft_token_id = tl.load(input_ids_ptr + start_idx + pos + 1)
            draft_prob = tl.load(
                draft_token_probs_ptr
                + req_state_idx * draft_token_probs_stride
                + pos
            )
            target_prob = tl.load(target_token_probs_ptr + start_idx + pos)
            absolute_pos = tl.load(positions_ptr + start_idx + pos)
            uniform = tl.rand(seed + 0x9E3779B9, absolute_pos)
            accepted = uniform * draft_prob <= target_prob
            if accepted:
                tl.store(
                    sampled_ptr + req_idx * sampled_stride + pos,
                    draft_token_id,
                )
                num_sampled += 1
            else:
                rejected = True
                rejected_pos = pos
                num_sampled += 1

    if not rejected:
        bonus_token = tl.load(target_sampled_ptr + end_idx - 1)
        tl.store(
            sampled_ptr + req_idx * sampled_stride + num_draft_tokens,
            bonus_token,
        )
        num_sampled += 1

    tl.store(num_sampled_ptr + req_idx, num_sampled)
    tl.store(rejected_pos_ptr + req_idx, rejected_pos)


@triton.jit
def _residual_sample_blocks_kernel(
    local_argmax_ptr,
    local_argmax_stride,
    local_max_ptr,
    local_max_stride,
    rejected_pos_ptr,
    cu_num_logits_ptr,
    idx_mapping_ptr,
    positions_ptr,
    seeds_ptr,
    target_logits_ptr,
    target_logits_stride,
    target_logsumexp_ptr,
    draft_logits_ptr,
    draft_logits_req_stride,
    draft_logits_pos_stride,
    draft_logsumexp_ptr,
    draft_logsumexp_stride,
    draft_temperature,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    USE_FP64_NOISE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    rejected_pos = tl.load(rejected_pos_ptr + req_idx)
    block_idx = tl.program_id(1)
    local_offset = req_idx * local_argmax_stride + block_idx
    if rejected_pos < 0:
        tl.store(local_argmax_ptr + local_offset, 0)
        tl.store(local_max_ptr + local_offset, float("-inf"))
        return

    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    target_row = start_idx + rejected_pos

    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    target_logits = tl.load(
        target_logits_ptr + target_row * target_logits_stride + block,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)
    target_logsumexp = tl.load(target_logsumexp_ptr + target_row).to(tl.float32)
    target_probs = tl.exp(target_logits - target_logsumexp)

    draft_logits = tl.load(
        draft_logits_ptr
        + req_state_idx * draft_logits_req_stride
        + rejected_pos * draft_logits_pos_stride
        + block,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)
    draft_logsumexp = tl.load(
        draft_logsumexp_ptr
        + req_state_idx * draft_logsumexp_stride
        + rejected_pos
    ).to(tl.float32)
    draft_probs = tl.exp(draft_logits / draft_temperature - draft_logsumexp)
    residual = tl.maximum(target_probs - draft_probs, 0.0)

    seed = tl.load(seeds_ptr + req_state_idx)
    absolute_pos = tl.load(positions_ptr + target_row)
    residual_seed = tl.randint(seed + 0x6A09E667, absolute_pos)
    random = tl.rand(residual_seed, block)
    if USE_FP64_NOISE:
        random = random.to(tl.float64)
    noise = -tl.log(-tl.log(random + 1e-20) + 1e-20)
    noisy_residual = tl.where(
        mask & (residual > 0.0),
        tl.log(residual) + noise.to(tl.float32),
        float("-inf"),
    )
    idx = tl.argmax(noisy_residual, axis=0)
    tl.store(
        local_argmax_ptr + local_offset,
        block_idx * BLOCK_SIZE + idx,
    )
    tl.store(local_max_ptr + local_offset, tl.max(noisy_residual, axis=0))


@triton.jit
def _residual_sample_reduce_kernel(
    sampled_ptr,
    sampled_stride,
    rejected_pos_ptr,
    local_argmax_ptr,
    local_argmax_stride,
    local_max_ptr,
    local_max_stride,
    num_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    rejected_pos = tl.load(rejected_pos_ptr + req_idx)
    if rejected_pos < 0:
        return

    block = tl.arange(0, BLOCK_SIZE)
    mask = block < num_blocks
    local_max = tl.load(
        local_max_ptr + req_idx * local_max_stride + block,
        mask=mask,
        other=float("-inf"),
    )
    winning_block = tl.argmax(local_max, axis=0)
    recovered_token = tl.load(
        local_argmax_ptr
        + req_idx * local_argmax_stride
        + winning_block
    )
    tl.store(
        sampled_ptr + req_idx * sampled_stride + rejected_pos,
        recovered_token,
    )


def stochastic_rejection_sample(
    target_sampled: torch.Tensor,
    input_ids: torch.Tensor,
    cu_num_logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    positions: torch.Tensor,
    seeds: torch.Tensor,
    num_speculative_steps: int,
    draft_token_probs: torch.Tensor,
    draft_logits: torch.Tensor,
    draft_logsumexp: torch.Tensor,
    draft_temperature: float,
    target_token_probs: torch.Tensor,
    target_logits: torch.Tensor,
    target_logsumexp: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact rejection sampling from compact per-request draft state."""
    num_reqs = cu_num_logits.shape[0] - 1
    vocab_size = target_logits.shape[1]
    assert draft_temperature > 0.0
    assert target_sampled.shape == input_ids.shape == target_token_probs.shape
    assert target_logsumexp.shape == target_sampled.shape
    assert draft_token_probs.shape[:2] == (
        seeds.shape[0],
        num_speculative_steps,
    )
    assert draft_logsumexp.shape == draft_token_probs.shape
    assert draft_logits.shape[:2] == draft_token_probs.shape
    assert draft_logits.shape[2] == vocab_size

    sampled = torch.full(
        (num_reqs, num_speculative_steps + 1),
        -1,
        dtype=target_sampled.dtype,
        device=target_sampled.device,
    )
    num_sampled = torch.empty(num_reqs, dtype=torch.int32, device=sampled.device)
    rejected_pos = torch.empty_like(num_sampled)
    _stochastic_accept_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        rejected_pos,
        target_sampled,
        input_ids,
        cu_num_logits,
        idx_mapping,
        positions,
        seeds,
        draft_token_probs,
        draft_token_probs.stride(0),
        target_token_probs,
        MAX_SPEC_LEN=num_speculative_steps,
        num_warps=1,
    )

    block_size = 1024
    num_blocks = triton.cdiv(vocab_size, block_size)
    local_argmax = torch.empty(
        (num_reqs, num_blocks), dtype=torch.int64, device=sampled.device
    )
    local_max = torch.empty(
        (num_reqs, num_blocks), dtype=torch.float32, device=sampled.device
    )
    _residual_sample_blocks_kernel[(num_reqs, num_blocks)](
        local_argmax,
        local_argmax.stride(0),
        local_max,
        local_max.stride(0),
        rejected_pos,
        cu_num_logits,
        idx_mapping,
        positions,
        seeds,
        target_logits,
        target_logits.stride(0),
        target_logsumexp,
        draft_logits,
        draft_logits.stride(0),
        draft_logits.stride(1),
        draft_logsumexp,
        draft_logsumexp.stride(0),
        draft_temperature,
        vocab_size,
        BLOCK_SIZE=block_size,
        USE_FP64_NOISE=True,
    )
    _residual_sample_reduce_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        rejected_pos,
        local_argmax,
        local_argmax.stride(0),
        local_max,
        local_max.stride(0),
        num_blocks,
        BLOCK_SIZE=triton.next_power_of_2(num_blocks),
    )
    return sampled, num_sampled


@triton.jit
def _rejection_logprob_token_ids_kernel(
    output_ptr,
    sampled_ptr,
    sampled_stride,
    num_sampled_ptr,
    cu_num_logits_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_logits = end_idx - start_idx
    num_sampled = tl.load(num_sampled_ptr + req_idx)
    block = tl.arange(0, BLOCK_SIZE)
    valid = block < num_sampled
    token_ids = tl.load(
        sampled_ptr + req_idx * sampled_stride + block,
        mask=valid,
        other=0,
    )
    tl.store(
        output_ptr + start_idx + block,
        token_ids,
        mask=block < num_logits,
    )


def get_rejection_logprob_token_ids(
    sampled: torch.Tensor,
    num_sampled: torch.Tensor,
    cu_num_logits: torch.Tensor,
    total_num_logits: int,
) -> torch.Tensor:
    num_reqs = sampled.shape[0]
    token_ids = torch.empty(
        total_num_logits,
        dtype=torch.int64,
        device=sampled.device,
    )
    _rejection_logprob_token_ids_kernel[(num_reqs,)](
        token_ids,
        sampled,
        sampled.stride(0),
        num_sampled,
        cu_num_logits,
        BLOCK_SIZE=triton.next_power_of_2(sampled.shape[1]),
        num_warps=1,
    )
    return token_ids
