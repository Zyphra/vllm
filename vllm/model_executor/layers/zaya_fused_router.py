"""Optional fused Zaya top-1 router.

The kernel keeps the router's BF16 rounding boundaries and FP32 selection
policy while replacing the per-token Python/Torch sequence with one static
Triton program. It is default-off until checkpoint numerics and FULL-graph
activation are qualified.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from vllm.utils.torch_utils import direct_register_custom_op

ZAYA_FUSED_ROUTER = os.getenv("VLLM_ZAYA_FUSED_ROUTER", "0") == "1"


@triton.jit
def _bf16_round(x):
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _matvec_gelu(src, weight, bias, dst, src_row, dst_row,
                 dim: tl.constexpr, block_j: tl.constexpr,
                 block_k: tl.constexpr):
    for j0 in range(0, dim, block_j):
        j = j0 + tl.arange(0, block_j)
        acc = tl.zeros((block_j,), dtype=tl.float32)
        for k0 in range(0, dim, block_k):
            k = k0 + tl.arange(0, block_k)
            x = tl.load(src + src_row * dim + k)
            w = tl.load(weight + j[:, None] * dim + k[None, :]).to(
                tl.float32)
            acc += tl.sum(w * x[None, :], axis=1)
        acc = _bf16_round(acc + tl.load(bias + j).to(tl.float32))
        acc = _bf16_round(
            0.5 * acc * (1.0 + tl.math.erf(acc * 0.7071067811865476)))
        tl.store(dst + dst_row * dim + j, acc)


@triton.jit
def _zaya_router_kernel(
    hidden,
    prev,
    eda_scale,
    norm_weight,
    w0,
    b0,
    w1,
    b1,
    w2,
    balancing_biases,
    scratch,
    probs,
    choices,
    next_states,
    hidden_stride,
    prev_stride,
    next_stride,
    eps,
    D: tl.constexpr,
    E: tl.constexpr,
    E_PAD: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_EDA: tl.constexpr,
):
    row = tl.program_id(0)
    d = tl.arange(0, D)
    hs = tl.load(hidden + row * hidden_stride + d).to(tl.float32)
    if USE_EDA:
        previous = tl.load(prev + row * prev_stride + d).to(tl.float32)
        scale = tl.load(eda_scale + d).to(tl.float32)
        hs = _bf16_round(hs + _bf16_round(previous * scale))
    tl.store(next_states + row * next_stride + d, hs.to(tl.bfloat16))

    mean_square = tl.sum(hs * hs) / D
    norm = tl.load(norm_weight + d).to(tl.float32)
    normalized = _bf16_round(
        hs * tl.math.rsqrt(mean_square + eps) * norm)

    scratch_base = scratch + row * (2 * D)
    tl.store(scratch_base + d, normalized)
    tl.debug_barrier()
    _matvec_gelu(scratch, w0, b0, scratch, row * 2, row * 2 + 1, D,
                 BLOCK_J, BLOCK_K)
    tl.debug_barrier()
    _matvec_gelu(scratch, w1, b1, scratch, row * 2 + 1, row * 2, D,
                 BLOCK_J, BLOCK_K)
    tl.debug_barrier()

    experts = tl.arange(0, E_PAD)
    logits = tl.zeros((E_PAD,), dtype=tl.float32)
    for k0 in range(0, D, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(scratch_base + k)
        w = tl.load(
            w2 + experts[:, None] * D + k[None, :],
            mask=experts[:, None] < E,
            other=0.0,
        ).to(tl.float32)
        logits += tl.sum(w * x[None, :], axis=1)
    logits = _bf16_round(logits)

    neg_inf = float("-inf")
    logits = tl.where(experts < E, logits, neg_inf)
    maximum = tl.max(logits, 0)
    exp_logits = tl.exp(logits - maximum)
    exp_logits = tl.where(experts < E, exp_logits, 0.0)
    probabilities = exp_logits / tl.sum(exp_logits, 0)

    biases = tl.load(
        balancing_biases + experts, mask=experts < E, other=neg_inf
    ).to(tl.float32)
    choice_logits = tl.where(experts < E, probabilities + biases, neg_inf)
    choice = tl.argmax(choice_logits, 0)
    selected_probability = tl.sum(
        tl.where(experts == choice, probabilities, 0.0), 0)
    tl.store(probs + row, selected_probability.to(tl.bfloat16))
    tl.store(choices + row, choice.to(tl.int64))


def fused_router_into(
    hidden_states: torch.Tensor,
    previous_states: torch.Tensor,
    eda_scale: torch.Tensor,
    norm_weight: torch.Tensor,
    w0: torch.Tensor,
    b0: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    balancing_biases: torch.Tensor,
    probs: torch.Tensor,
    choices: torch.Tensor,
    next_states: torch.Tensor,
    scratch: torch.Tensor,
    eps: float,
    use_eda: bool,
) -> None:
    rows, dim = hidden_states.shape
    experts = w2.shape[0]
    padded_experts = max(32, triton.next_power_of_2(experts))
    _zaya_router_kernel[(rows,)](
        hidden_states,
        previous_states,
        eda_scale,
        norm_weight,
        w0,
        b0,
        w1,
        b1,
        w2,
        balancing_biases,
        scratch,
        probs,
        choices,
        next_states,
        hidden_states.stride(0),
        previous_states.stride(0),
        next_states.stride(0),
        eps,
        D=dim,
        E=experts,
        E_PAD=padded_experts,
        BLOCK_J=256,
        BLOCK_K=64,
        USE_EDA=use_eda,
        num_warps=8,
        num_stages=2,
    )


def zaya_fused_router(
    hidden_states: torch.Tensor,
    previous_states: torch.Tensor,
    eda_scale: torch.Tensor,
    norm_weight: torch.Tensor,
    w0: torch.Tensor,
    b0: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    balancing_biases: torch.Tensor,
    probs: torch.Tensor,
    choices: torch.Tensor,
    next_states: torch.Tensor,
    scratch: torch.Tensor,
    eps: float,
    use_eda: bool,
) -> None:
    fused_router_into(
        hidden_states,
        previous_states,
        eda_scale,
        norm_weight,
        w0,
        b0,
        w1,
        b1,
        w2,
        balancing_biases,
        probs,
        choices,
        next_states,
        scratch,
        eps,
        use_eda,
    )


def zaya_fused_router_fake(*args, **kwargs) -> None:
    return None


direct_register_custom_op(
    op_name="zaya_fused_router",
    op_func=zaya_fused_router,
    mutates_args=["probs", "choices", "next_states", "scratch"],
    fake_impl=zaya_fused_router_fake,
)
