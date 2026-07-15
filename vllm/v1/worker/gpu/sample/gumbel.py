# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _temperature_kernel(
    logits_ptr,
    logits_stride,
    idx_mapping_ptr,
    temperature_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)
    temperature = tl.load(temperature_ptr + req_state_idx).to(tl.float32)
    if temperature == 0.0 or temperature == 1.0:
        # Early return to avoid loading logits.
        return

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size

    logits = tl.load(logits_ptr + batch_idx * logits_stride + block, mask=mask)
    logits = logits.to(tl.float32)
    logits = logits / temperature
    tl.store(logits_ptr + batch_idx * logits_stride + block, logits, mask=mask)


def apply_temperature(
    logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
) -> None:
    num_reqs, vocab_size = logits.shape
    BLOCK_SIZE = 8192
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    _temperature_kernel[(num_reqs, num_blocks)](
        logits,
        logits.stride(0),
        idx_mapping,
        temperature,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )


@triton.jit
def _gumbel_sample_kernel(
    local_argmax_ptr,
    local_argmax_stride,
    local_max_ptr,
    local_max_stride,
    logits_ptr,
    logits_stride,
    idx_mapping_ptr,
    seeds_ptr,
    pos_ptr,
    temp_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    APPLY_TEMPERATURE: tl.constexpr,
    USE_FP64_NOISE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    logits = tl.load(
        logits_ptr + batch_idx * logits_stride + block,
        mask=mask,
        other=float("-inf"),
    )
    logits = logits.to(tl.float32)

    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    if temp != 0.0:
        # Calculate the seed for gumbel noise.
        seed = tl.load(seeds_ptr + req_state_idx)
        pos = tl.load(pos_ptr + batch_idx)
        gumbel_seed = tl.randint(seed, pos)

        # Generate gumbel noise.
        r = tl.rand(gumbel_seed, block)
        if USE_FP64_NOISE:
            r = r.to(tl.float64)
        gumbel_noise = -tl.log(-tl.log(r + 1e-20) + 1e-20)
        gumbel_noise = gumbel_noise.to(tl.float32)

        # Apply temperature.
        if APPLY_TEMPERATURE:
            # NOTE(woosuk): Match the behavior of _temperature_kernel.
            # E.g., if the kernel uses tl.div_rn, we should use tl.div_rn here too.
            logits = logits / temp

        # Apply gumbel noise.
        logits = tl.where(mask, logits + gumbel_noise, float("-inf"))

    idx = tl.argmax(logits, axis=0)
    token_id = block_idx * BLOCK_SIZE + idx
    value = tl.max(logits, axis=0)
    tl.store(local_argmax_ptr + batch_idx * local_argmax_stride + block_idx, token_id)
    tl.store(local_max_ptr + batch_idx * local_max_stride + block_idx, value)


def gumbel_sample(
    logits: torch.Tensor,  # [num_reqs, vocab_size]
    idx_mapping: torch.Tensor,  # [num_reqs]
    temperature: torch.Tensor,  # [num_reqs]
    seed: torch.Tensor,  # [num_reqs]
    pos: torch.Tensor,  # [num_reqs]
    apply_temperature: bool,
    use_fp64_noise: bool = True,
) -> torch.Tensor:
    num_reqs, vocab_size = logits.shape
    BLOCK_SIZE = 1024 if use_fp64_noise else 2048
    num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
    local_argmax = torch.empty(
        num_reqs,
        num_blocks,
        dtype=torch.int64,
        device=logits.device,
    )
    local_max = torch.empty(
        num_reqs,
        num_blocks,
        dtype=torch.float32,
        device=logits.device,
    )
    _gumbel_sample_kernel[(num_reqs, num_blocks)](
        local_argmax,
        local_argmax.stride(0),
        local_max,
        local_max.stride(0),
        logits,
        logits.stride(0),
        idx_mapping,
        seed,
        pos,
        temperature,
        vocab_size,
        BLOCK_SIZE=BLOCK_SIZE,
        APPLY_TEMPERATURE=apply_temperature,
        USE_FP64_NOISE=use_fp64_noise,
    )
    # NOTE(woosuk): Use int64 for later indexing.
    max_block_idx = local_max.argmax(dim=-1, keepdim=True)
    sampled = local_argmax.gather(dim=-1, index=max_block_idx).view(-1)
    return sampled


@triton.jit
def _gumbel_sample_with_probs_kernel(
    local_argmax_ptr,
    local_argmax_stride,
    local_noisy_max_ptr,
    local_noisy_max_stride,
    local_logsumexp_ptr,
    local_logsumexp_stride,
    logits_ptr,
    logits_stride,
    idx_mapping_ptr,
    seeds_ptr,
    pos_ptr,
    temp_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    APPLY_TEMPERATURE: tl.constexpr,
    USE_FP64_NOISE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    logits = tl.load(
        logits_ptr + batch_idx * logits_stride + block,
        mask=mask,
        other=float("-inf"),
    ).to(tl.float32)

    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    if APPLY_TEMPERATURE:
        logits = logits / temp

    block_max = tl.max(logits, axis=0)
    block_logsumexp = block_max + tl.log(
        tl.sum(tl.exp(logits - block_max), axis=0)
    )

    seed = tl.load(seeds_ptr + req_state_idx)
    pos = tl.load(pos_ptr + batch_idx)
    gumbel_seed = tl.randint(seed, pos)
    random = tl.rand(gumbel_seed, block)
    if USE_FP64_NOISE:
        random = random.to(tl.float64)
    noise = -tl.log(-tl.log(random + 1e-20) + 1e-20)
    noisy_logits = tl.where(
        mask,
        logits + noise.to(tl.float32),
        float("-inf"),
    )

    idx = tl.argmax(noisy_logits, axis=0)
    tl.store(
        local_argmax_ptr + batch_idx * local_argmax_stride + block_idx,
        block_idx * BLOCK_SIZE + idx,
    )
    tl.store(
        local_noisy_max_ptr + batch_idx * local_noisy_max_stride + block_idx,
        tl.max(noisy_logits, axis=0),
    )
    tl.store(
        local_logsumexp_ptr + batch_idx * local_logsumexp_stride + block_idx,
        block_logsumexp,
    )


@triton.jit
def _reduce_gumbel_sample_with_probs_kernel(
    sampled_ptr,
    selected_probs_ptr,
    logsumexp_ptr,
    local_argmax_ptr,
    local_argmax_stride,
    local_noisy_max_ptr,
    local_noisy_max_stride,
    local_logsumexp_ptr,
    local_logsumexp_stride,
    logits_ptr,
    logits_stride,
    idx_mapping_ptr,
    temp_ptr,
    prob_token_ids_ptr,
    num_blocks,
    BLOCK_SIZE: tl.constexpr,
    APPLY_TEMPERATURE: tl.constexpr,
    USE_PROB_TOKEN_IDS: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    block = tl.arange(0, BLOCK_SIZE)
    mask = block < num_blocks

    noisy_max = tl.load(
        local_noisy_max_ptr + batch_idx * local_noisy_max_stride + block,
        mask=mask,
        other=float("-inf"),
    )
    winning_block = tl.argmax(noisy_max, axis=0)
    sampled = tl.load(
        local_argmax_ptr + batch_idx * local_argmax_stride + winning_block
    )

    block_logsumexp = tl.load(
        local_logsumexp_ptr + batch_idx * local_logsumexp_stride + block,
        mask=mask,
        other=float("-inf"),
    )
    max_logsumexp = tl.max(block_logsumexp, axis=0)
    total_logsumexp = max_logsumexp + tl.log(
        tl.sum(tl.exp(block_logsumexp - max_logsumexp), axis=0)
    )

    prob_token_id = sampled
    if USE_PROB_TOKEN_IDS:
        prob_token_id = tl.load(prob_token_ids_ptr + batch_idx)
    selected_logit = tl.load(
        logits_ptr + batch_idx * logits_stride + prob_token_id
    ).to(tl.float32)
    if APPLY_TEMPERATURE:
        req_state_idx = tl.load(idx_mapping_ptr + batch_idx)
        temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
        selected_logit = selected_logit / temp

    tl.store(sampled_ptr + batch_idx, sampled)
    tl.store(selected_probs_ptr + batch_idx, tl.exp(selected_logit - total_logsumexp))
    tl.store(logsumexp_ptr + batch_idx, total_logsumexp)


def gumbel_sample_with_probs(
    logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    temperature: torch.Tensor,
    seed: torch.Tensor,
    pos: torch.Tensor,
    apply_temperature: bool,
    use_fp64_noise: bool = True,
    prob_token_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample logits and return compact probability data in two kernels.

    All mapped temperatures must be positive. This helper is for stochastic
    sampling paths that need q(x) and the log-normalizer for rejection sampling.
    By default q(x) is returned for the sampled token. ``prob_token_ids`` can
    select a different token per row, such as the draft token under a target
    distribution.
    """
    num_reqs, vocab_size = logits.shape
    if prob_token_ids is not None:
        assert prob_token_ids.shape == (num_reqs,)
    block_size = 1024 if use_fp64_noise else 2048
    num_blocks = triton.cdiv(vocab_size, block_size)
    local_argmax = torch.empty(
        num_reqs, num_blocks, dtype=torch.int64, device=logits.device
    )
    local_noisy_max = torch.empty(
        num_reqs, num_blocks, dtype=torch.float32, device=logits.device
    )
    local_logsumexp = torch.empty_like(local_noisy_max)

    _gumbel_sample_with_probs_kernel[(num_reqs, num_blocks)](
        local_argmax,
        local_argmax.stride(0),
        local_noisy_max,
        local_noisy_max.stride(0),
        local_logsumexp,
        local_logsumexp.stride(0),
        logits,
        logits.stride(0),
        idx_mapping,
        seed,
        pos,
        temperature,
        vocab_size,
        BLOCK_SIZE=block_size,
        APPLY_TEMPERATURE=apply_temperature,
        USE_FP64_NOISE=use_fp64_noise,
    )

    sampled = torch.empty(num_reqs, dtype=torch.int64, device=logits.device)
    selected_probs = torch.empty(
        num_reqs, dtype=torch.float32, device=logits.device
    )
    logsumexp = torch.empty_like(selected_probs)
    _reduce_gumbel_sample_with_probs_kernel[(num_reqs,)](
        sampled,
        selected_probs,
        logsumexp,
        local_argmax,
        local_argmax.stride(0),
        local_noisy_max,
        local_noisy_max.stride(0),
        local_logsumexp,
        local_logsumexp.stride(0),
        logits,
        logits.stride(0),
        idx_mapping,
        temperature,
        prob_token_ids,
        num_blocks,
        BLOCK_SIZE=triton.next_power_of_2(num_blocks),
        APPLY_TEMPERATURE=apply_temperature,
        USE_PROB_TOKEN_IDS=prob_token_ids is not None,
    )
    return sampled, selected_probs, logsumexp
