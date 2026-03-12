"""Fused Triton kernels for the SMoE router.

Two fused operations that replace multiple small kernel launches:

1.  ``fused_eda_rmsnorm``
    EDA scaled-add  +  RMSNorm in **one** kernel launch / memory pass.
    Replaces:  ``hs = x + prev * scale``  →  ``clone``  →  ``rmsnorm(hs)``

2.  ``fused_softmax_topk``
    softmax  +  bias-add  +  argmax/topk  +  gather in **one** kernel launch.
    Replaces:  ``softmax``  →  ``+ bias``  →  ``topk``  →  ``gather``
"""

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# 1. Fused EDA-add + RMSNorm
# ---------------------------------------------------------------------------

@triton.jit
def _fused_eda_rmsnorm_kernel(
    x_ptr, prev_ptr, scale_ptr, weight_ptr,
    out_norm_ptr, out_prenorm_ptr,
    D, stride_row, eps,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    base = row * stride_row

    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
    orig_dtype = x.dtype
    x = x.to(tl.float32)

    prev = tl.load(prev_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    hs = x + prev * scale

    tl.store(out_prenorm_ptr + base + offs, hs.to(orig_dtype), mask=mask)

    var = tl.sum(hs * hs, axis=0) / D
    normed = hs * tl.math.rsqrt(var + eps)

    w = tl.load(weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    normed = normed * w

    tl.store(out_norm_ptr + base + offs, normed.to(orig_dtype), mask=mask)


def fused_eda_rmsnorm(
    x: torch.Tensor,
    prev: torch.Tensor,
    scale: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused EDA scaled-add + RMSNorm in a single kernel.

    Computes::

        prenorm = x + prev * scale
        normed  = RMSNorm(prenorm, weight, eps)

    Returns ``(normed, prenorm)`` — both in the dtype of *x*.
    """
    S, D = x.shape
    BLOCK_D = triton.next_power_of_2(D)
    if BLOCK_D > 8192:
        raise ValueError(f"D={D} too large for fused_eda_rmsnorm (max 8192)")

    out_norm = torch.empty_like(x)
    out_prenorm = torch.empty_like(x)

    _fused_eda_rmsnorm_kernel[(S,)](
        x, prev, scale, weight,
        out_norm, out_prenorm,
        D, x.stride(0), eps,
        BLOCK_D=BLOCK_D,
    )
    return out_norm, out_prenorm


# ---------------------------------------------------------------------------
# 2. Fused softmax + bias + topk + gather
# ---------------------------------------------------------------------------

@triton.jit
def _fused_softmax_topk_kernel(
    logits_ptr, biases_ptr,
    route_prob_ptr, expert_idx_ptr,
    stride_logits, E,
    BLOCK_E: tl.constexpr,
    TOPK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_E)
    mask = offs < E

    logits = tl.load(logits_ptr + row * stride_logits + offs,
                     mask=mask, other=float('-inf'))
    out_dtype = logits.dtype
    logits = logits.to(tl.float32)

    mx = tl.max(logits, axis=0)
    exp_v = tl.exp(logits - mx)
    probs = exp_v / tl.sum(exp_v, axis=0)

    biases = tl.load(biases_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    biased = tl.where(mask, probs + biases, float('-inf'))

    # ── top-1 ──────────────────────────────────────────────────────────────
    mx0 = tl.max(biased, axis=0)
    eq0 = biased == mx0
    idx_0 = tl.min(tl.where(eq0, offs, BLOCK_E), axis=0)
    prob_0 = tl.sum(tl.where(offs == idx_0, probs, 0.0), axis=0)

    base = row * TOPK
    tl.store(expert_idx_ptr + base, idx_0.to(tl.int64))
    tl.store(route_prob_ptr + base, prob_0.to(out_dtype))

    # ── top-2 (only compiled when TOPK >= 2) ──────────────────────────────
    if TOPK >= 2:
        biased = tl.where(offs == idx_0, float('-inf'), biased)
        mx1 = tl.max(biased, axis=0)
        eq1 = biased == mx1
        idx_1 = tl.min(tl.where(eq1, offs, BLOCK_E), axis=0)
        prob_1 = tl.sum(tl.where(offs == idx_1, probs, 0.0), axis=0)

        tl.store(expert_idx_ptr + base + 1, idx_1.to(tl.int64))
        tl.store(route_prob_ptr + base + 1, prob_1.to(out_dtype))


def fused_softmax_topk(
    logits: torch.Tensor,
    biases: torch.Tensor,
    topk: int,
    use_mod: bool = False,
    skip_idx: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused softmax + bias-add + argmax/topk + gather.

    Supports *topk* = 1 or 2 via a Triton kernel.  Falls back to PyTorch
    for *topk* > 2.

    Returns ``(route_prob, expert_idx)``  — shapes ``(S, topk)`` each.
    ``route_prob`` has the same dtype as *logits*; ``expert_idx`` is int64.
    """
    S, E = logits.shape

    if topk > 2:
        return _softmax_topk_fallback(logits, biases, topk, use_mod, skip_idx)

    BLOCK_E = triton.next_power_of_2(E)
    route_prob = torch.empty(S, topk, dtype=logits.dtype, device=logits.device)
    expert_idx = torch.empty(S, topk, dtype=torch.int64, device=logits.device)

    _fused_softmax_topk_kernel[(S,)](
        logits, biases,
        route_prob, expert_idx,
        logits.stride(0), E,
        BLOCK_E=BLOCK_E, TOPK=topk,
    )

    if use_mod and topk > 1:
        _apply_mod_masking(route_prob, expert_idx, skip_idx)

    return route_prob, expert_idx


def _apply_mod_masking(
    route_prob: torch.Tensor,
    expert_idx: torch.Tensor,
    skip_idx: int,
) -> None:
    """In-place MOD post-processing on the tiny (S, topk) tensors.

    Once a skip expert appears at position *k*, all later positions
    are forced to *skip_idx* and their probs are set to prob(skip_idx).
    """
    n_mask = expert_idx == skip_idx
    cumsum = torch.cumsum(n_mask, dim=-1)
    fix = cumsum > 0
    changed = fix & ~n_mask
    expert_idx.masked_fill_(fix, skip_idx)
    if changed.any():
        first_skip_prob = route_prob[:, 0:1].expand_as(route_prob)
        route_prob[changed] = first_skip_prob[changed]


def _softmax_topk_fallback(logits, biases, topk, use_mod, skip_idx):
    """PyTorch fallback for topk > 2."""
    expert_prob = torch.softmax(logits, dim=-1)
    biased = expert_prob.detach().to(torch.float32) + biases
    _, expert_idx = torch.topk(biased, topk, dim=-1)
    if use_mod:
        n_mask = expert_idx == skip_idx
        cumsum = torch.cumsum(n_mask, dim=-1)
        expert_idx = expert_idx.masked_fill(cumsum > 0, skip_idx)
    route_prob = torch.gather(expert_prob, dim=1, index=expert_idx)
    return route_prob, expert_idx
