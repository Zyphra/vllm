# SPDX-License-Identifier: Apache-2.0
"""TiDAR two-forward (TF) paged split-K attention.

A specialization of the single-forward (SF) split-K kernel
(`sf_attention.py`) to the TF layout: per request, a SINGLE K+1 inline block
attending to the paged prefix, with the block masked **causal** (verify pass)
or **bidirectional** (draft pass). It is SF with ``P_props=0`` plus one
``IS_CAUSAL`` flag.

Reused from SF verbatim:
  * ``_sf_prefix_splitk_partial`` — Phase-1 paged-prefix split-K. Reads the
    cache through ``kv_cache.stride()`` so it tolerates the SMoE mamba-page-
    padded (non-contiguous) block stride — the exact thing AITER's
    ``flash_attn_with_kvcache`` gets wrong (it assumes a contiguous stride).
  * ``_sf_pick_num_splits`` — static (capture-safe) split heuristic.

New here: ``_tf_reduce_tail`` (Phase-2 single-block, causal/bidirectional) and
``_tf_fwd_paged_gqa`` (fused single-pass for splits==1, bit-exact to the
two-kernel path).

Design notes: docs/tf_full_splitk_design.md. Never MFMA-tune (verifier-feeding).
"""
import os
from typing import Optional

import torch
import triton
import triton.language as tl

def _tf_pick_num_splits(n_blocks_ub, base_programs, target=768, max_splits=16):
    """Static split count (capture-safe: no host sync on runtime prefix)."""
    if n_blocks_ub <= 1 or base_programs <= 0:
        return 1
    s = (target + base_programs - 1) // base_programs
    return max(1, min(s, max_splits, n_blocks_ub))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=["total_per_req_q", "block_size", "num_kv_heads", "head_dim", "G",
         "BLOCK_Q", "SPLITS"],
)
@triton.jit
def _tf_prefix_splitk_partial(
    Q, PartAcc, PartM, PartL,
    KV_CACHE_K, KV_CACHE_V, BlockTable, prefix_lens,
    stride_q_t, stride_q_h, stride_q_d,
    stride_pa_s, stride_pa_t, stride_pa_h, stride_pa_d,
    stride_pm_s, stride_pm_t, stride_pm_h,
    stride_kvc_block, stride_kvc_pos, stride_kvc_h, stride_kvc_d,
    stride_bt_r, stride_bt_b,
    total_per_req_q: tl.constexpr, block_size: tl.constexpr,
    num_heads: tl.constexpr, num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr, softmax_scale: tl.constexpr,
    G: tl.constexpr, BLOCK_Q: tl.constexpr, SPLITS: tl.constexpr,
):
    # Phase-1: paged prefix, split-K, online-softmax partials. Reads the cache
    # through kv_cache.stride() -> tolerates the mamba-page-padded (non-contig)
    # block stride. GQA-grouped (G query heads stacked in M). Mask-agnostic
    # (prefix is unconditionally visible). Verbatim from SF.
    pid_q = tl.program_id(0)
    pid_kv = tl.program_id(1)
    pid_rs = tl.program_id(2)
    pid_r = pid_rs // SPLITS
    pid_s = pid_rs % SPLITS

    prefix_len = tl.load(prefix_lens + pid_r)
    n_blocks = (prefix_len + block_size - 1) // block_size
    blocks_per_split = (n_blocks + SPLITS - 1) // SPLITS
    bt_start = pid_s * blocks_per_split
    bt_end = tl.minimum(bt_start + blocks_per_split, n_blocks)

    offs_d = tl.arange(0, head_dim)
    offs_m = tl.arange(0, G * BLOCK_Q)
    g_id = offs_m // BLOCK_Q
    qpos = offs_m % BLOCK_Q
    q_head = pid_kv * G + g_id
    qpos_g = pid_q * BLOCK_Q + qpos
    m_mask = qpos_g < total_per_req_q
    q_token_idx = pid_r * total_per_req_q + qpos_g
    q = tl.load(Q + q_token_idx[:, None] * stride_q_t + q_head[:, None] * stride_q_h
                + offs_d[None, :] * stride_q_d,
                mask=m_mask[:, None], other=0.0).to(tl.float32)

    NEG = -1.0e30
    m_i = tl.full((G * BLOCK_Q,), NEG, dtype=tl.float32)
    l_i = tl.zeros((G * BLOCK_Q,), dtype=tl.float32)
    acc = tl.zeros((G * BLOCK_Q, head_dim), dtype=tl.float32)

    bt_row_ptr = BlockTable + pid_r * stride_bt_r
    for bt_idx in range(bt_start, bt_end):
        block_id = tl.load(bt_row_ptr + bt_idx * stride_bt_b).to(tl.int64)
        offs_kv_in_blk = tl.arange(0, block_size)
        global_kv_pos = bt_idx * block_size + offs_kv_in_blk
        kv_mask = global_kv_pos < prefix_len
        k_ptrs = KV_CACHE_K + (block_id * stride_kvc_block
                               + offs_kv_in_blk[:, None] * stride_kvc_pos
                               + pid_kv * stride_kvc_h
                               + offs_d[None, :] * stride_kvc_d)
        v_ptrs = KV_CACHE_V + (block_id * stride_kvc_block
                               + offs_kv_in_blk[:, None] * stride_kvc_pos
                               + pid_kv * stride_kvc_h
                               + offs_d[None, :] * stride_kvc_d)
        k_block = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0).to(tl.float32)
        v_block = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0).to(tl.float32)
        attn_mask = kv_mask[None, :] & m_mask[:, None]
        qk = tl.dot(q, tl.trans(k_block)) * softmax_scale
        qk = tl.where(attn_mask, qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p, v_block)
        m_i = m_new

    base_t = pid_r * total_per_req_q + qpos_g
    pa = (PartAcc + pid_s * stride_pa_s + base_t[:, None] * stride_pa_t
          + q_head[:, None] * stride_pa_h + offs_d[None, :] * stride_pa_d)
    tl.store(pa, acc, mask=m_mask[:, None])
    pm = PartM + pid_s * stride_pm_s + base_t * stride_pm_t + q_head * stride_pm_h
    tl.store(pm, m_i, mask=m_mask)
    pl = PartL + pid_s * stride_pm_s + base_t * stride_pm_t + q_head * stride_pm_h
    tl.store(pl, l_i, mask=m_mask)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=["S_block", "num_heads", "head_dim", "SPLITS", "BLOCK_Q", "BLOCK_KV",
         "IS_CAUSAL"],
)
@triton.jit
def _tf_reduce_tail(
    Q, PartAcc, PartM, PartL, Inline_K, Inline_V, Out,
    stride_q_t, stride_q_h, stride_q_d,
    stride_pa_s, stride_pa_t, stride_pa_h, stride_pa_d,
    stride_pm_s, stride_pm_t, stride_pm_h,
    stride_ink_t, stride_ink_h, stride_ink_d,
    stride_inv_t, stride_inv_h, stride_inv_d,
    stride_o_t, stride_o_h, stride_o_d,
    S_block: tl.constexpr, num_heads: tl.constexpr, num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr, softmax_scale: tl.constexpr,
    SPLITS: tl.constexpr, BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_r = tl.program_id(2)

    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_mask = offs_q < S_block
    offs_d = tl.arange(0, head_dim)
    kv_head_id = pid_h * num_kv_heads // num_heads

    q_token_idx = pid_r * S_block + offs_q
    q = tl.load(Q + q_token_idx[:, None] * stride_q_t + pid_h * stride_q_h
                + offs_d[None, :] * stride_q_d,
                mask=q_mask[:, None], other=0.0).to(tl.float32)

    NEG = -1.0e30
    m_i = tl.full((BLOCK_Q,), NEG, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_Q, head_dim), dtype=tl.float32)

    # Merge the per-split prefix partials (online softmax).
    for s in range(0, SPLITS):
        pm = tl.load(PartM + s * stride_pm_s + q_token_idx * stride_pm_t
                     + pid_h * stride_pm_h, mask=q_mask, other=NEG)
        pl = tl.load(PartL + s * stride_pm_s + q_token_idx * stride_pm_t
                     + pid_h * stride_pm_h, mask=q_mask, other=0.0)
        pa = tl.load(PartAcc + s * stride_pa_s + q_token_idx[:, None] * stride_pa_t
                     + pid_h * stride_pa_h + offs_d[None, :] * stride_pa_d,
                     mask=q_mask[:, None], other=0.0)
        m_new = tl.maximum(m_i, pm)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(pm - m_new)
        l_i = alpha * l_i + beta * pl
        acc = acc * alpha[:, None] + beta[:, None] * pa
        m_i = m_new

    # Phase-2: the single inline K+1 block. Causal (verify) or full (draft).
    for kv_start in range(0, S_block, BLOCK_KV):
        offs_kv = kv_start + tl.arange(0, BLOCK_KV)
        kv_mask = offs_kv < S_block
        kv_token_idx = pid_r * S_block + offs_kv
        k_block = tl.load(Inline_K + kv_token_idx[:, None] * stride_ink_t
                          + kv_head_id * stride_ink_h + offs_d[None, :] * stride_ink_d,
                          mask=kv_mask[:, None], other=0.0).to(tl.float32)
        v_block = tl.load(Inline_V + kv_token_idx[:, None] * stride_inv_t
                          + kv_head_id * stride_inv_h + offs_d[None, :] * stride_inv_d,
                          mask=kv_mask[:, None], other=0.0).to(tl.float32)
        attn_mask = q_mask[:, None] & kv_mask[None, :]
        if IS_CAUSAL:
            attn_mask = attn_mask & (offs_kv[None, :] <= offs_q[:, None])
        qk = tl.dot(q, tl.trans(k_block)) * softmax_scale
        qk = tl.where(attn_mask, qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p, v_block)
        m_i = m_new

    acc = acc / l_i[:, None]
    o_ptrs = (Out + q_token_idx[:, None] * stride_o_t + pid_h * stride_o_h
              + offs_d[None, :] * stride_o_d)
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=q_mask[:, None])


@triton.jit
def _tf_fwd_paged_gqa(
    Q, Inline_K, Inline_V, Out,
    KV_CACHE_K, KV_CACHE_V, BlockTable, prefix_lens,
    stride_q_t, stride_q_h, stride_q_d,
    stride_ink_t, stride_ink_h, stride_ink_d,
    stride_inv_t, stride_inv_h, stride_inv_d,
    stride_o_t, stride_o_h, stride_o_d,
    stride_kvc_block, stride_kvc_pos, stride_kvc_h, stride_kvc_d,
    stride_bt_r, stride_bt_b,
    S_block: tl.constexpr, block_size: tl.constexpr, num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr, head_dim: tl.constexpr,
    softmax_scale: tl.constexpr,
    G: tl.constexpr, BLOCK_Q: tl.constexpr, BLOCK_KV: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_kv = tl.program_id(1)
    pid_r = tl.program_id(2)

    offs_m = tl.arange(0, G * BLOCK_Q)
    g_id = offs_m // BLOCK_Q
    qpos = pid_q * BLOCK_Q + (offs_m % BLOCK_Q)
    q_head = pid_kv * G + g_id
    row_mask = qpos < S_block
    offs_d = tl.arange(0, head_dim)

    prefix_len = tl.load(prefix_lens + pid_r)
    q_token_idx = pid_r * S_block + qpos
    q = tl.load(Q + q_token_idx[:, None] * stride_q_t + q_head[:, None] * stride_q_h
                + offs_d[None, :] * stride_q_d,
                mask=row_mask[:, None], other=0.0).to(tl.float32)

    NEG = -1.0e30
    m_i = tl.full((G * BLOCK_Q,), NEG, dtype=tl.float32)
    l_i = tl.zeros((G * BLOCK_Q,), dtype=tl.float32)
    acc = tl.zeros((G * BLOCK_Q, head_dim), dtype=tl.float32)

    # Phase-1: paged prefix (unconditionally visible).
    bt_row_ptr = BlockTable + pid_r * stride_bt_r
    n_prefix_blocks = (prefix_len + block_size - 1) // block_size
    for bt_idx in range(0, n_prefix_blocks):
        block_id = tl.load(bt_row_ptr + bt_idx * stride_bt_b).to(tl.int64)
        offs_kv_in_blk = tl.arange(0, block_size)
        global_kv_pos = bt_idx * block_size + offs_kv_in_blk
        kv_mask = global_kv_pos < prefix_len
        k_block = tl.load(KV_CACHE_K + (block_id * stride_kvc_block
                          + offs_kv_in_blk[:, None] * stride_kvc_pos
                          + pid_kv * stride_kvc_h + offs_d[None, :] * stride_kvc_d),
                          mask=kv_mask[:, None], other=0.0).to(tl.float32)
        v_block = tl.load(KV_CACHE_V + (block_id * stride_kvc_block
                          + offs_kv_in_blk[:, None] * stride_kvc_pos
                          + pid_kv * stride_kvc_h + offs_d[None, :] * stride_kvc_d),
                          mask=kv_mask[:, None], other=0.0).to(tl.float32)
        attn_mask = kv_mask[None, :] & row_mask[:, None]
        qk = tl.dot(q, tl.trans(k_block)) * softmax_scale
        qk = tl.where(attn_mask, qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p, v_block)
        m_i = m_new

    # Phase-2: single inline K+1 block, causal (verify) or full (draft).
    for kv_start in range(0, S_block, BLOCK_KV):
        offs_kv = kv_start + tl.arange(0, BLOCK_KV)
        kv_mask = offs_kv < S_block
        kv_token_idx = pid_r * S_block + offs_kv
        k_block = tl.load(Inline_K + kv_token_idx[:, None] * stride_ink_t
                          + pid_kv * stride_ink_h + offs_d[None, :] * stride_ink_d,
                          mask=kv_mask[:, None], other=0.0).to(tl.float32)
        v_block = tl.load(Inline_V + kv_token_idx[:, None] * stride_inv_t
                          + pid_kv * stride_inv_h + offs_d[None, :] * stride_inv_d,
                          mask=kv_mask[:, None], other=0.0).to(tl.float32)
        attn_mask = row_mask[:, None] & kv_mask[None, :]
        if IS_CAUSAL:
            attn_mask = attn_mask & (offs_kv[None, :] <= qpos[:, None])
        qk = tl.dot(q, tl.trans(k_block)) * softmax_scale
        qk = tl.where(attn_mask, qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p, v_block)
        m_i = m_new

    acc = acc / l_i[:, None]
    o_ptrs = (Out + q_token_idx[:, None] * stride_o_t + q_head[:, None] * stride_o_h
              + offs_d[None, :] * stride_o_d)
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=row_mask[:, None])


def tf_attention_triton_paged(
    q: torch.Tensor,            # [num_reqs * S_block, H, D]
    inline_k: torch.Tensor,     # [num_reqs * S_block, H_kv, D]
    inline_v: torch.Tensor,
    kv_cache_k: torch.Tensor,   # [num_blocks, block_size, H_kv, D] (paged prefix)
    kv_cache_v: torch.Tensor,
    block_table: torch.Tensor,  # [num_reqs, max_blocks] int32
    prefix_lens: torch.Tensor,  # [num_reqs] int32
    s_block: int,               # = K+1 (the per-request query/block length)
    block_size: int,
    softmax_scale: float,
    is_causal: bool,            # verify -> True, draft -> False
    out: Optional[torch.Tensor] = None,
    block_q: int = 16,   # Triton requires tl.arange(0, G * BLOCK_Q) to have a
                         # power-of-two range; SMoE uses power-of-two G.
    block_kv: int = 128,
) -> torch.Tensor:
    """Paged TF attention: K+1 inline block + paged prefix, split-K.

    Adaptive dispatch (mirrors SF): static split heuristic picks splits==1 ->
    bit-exact fused GQA kernel; else split-K two-kernel path. Both read the
    paged prefix via cache strides (padded-KV safe). ``is_causal`` selects the
    verify (causal) vs draft (bidirectional) block mask.
    """
    assert kv_cache_k.shape[1] == block_size
    T_q, num_heads, head_dim = q.shape
    num_reqs = prefix_lens.shape[0]
    num_kv_heads = inline_v.shape[1]
    G = num_heads // num_kv_heads
    if out is None:
        out = torch.empty_like(q)

    q_blocks = triton.cdiv(s_block, block_q)
    n_blocks_ub = block_table.shape[1]
    base = q_blocks * num_kv_heads * num_reqs
    force_no_splits = (
        os.environ.get("VLLM_TIDAR_TF_PAGED_NO_SPLITS", "0") == "1"
        or os.environ.get("VLLM_TIDAR_FA_NO_SPLITS", "0") == "1"
    )
    splits = 1 if force_no_splits else _tf_pick_num_splits(n_blocks_ub, base)

    if splits == 1:
        grid = (q_blocks, num_kv_heads, num_reqs)
        _tf_fwd_paged_gqa[grid](
            q, inline_k, inline_v, out, kv_cache_k, kv_cache_v,
            block_table, prefix_lens,
            q.stride(0), q.stride(1), q.stride(2),
            inline_k.stride(0), inline_k.stride(1), inline_k.stride(2),
            inline_v.stride(0), inline_v.stride(1), inline_v.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            kv_cache_k.stride(0), kv_cache_k.stride(1),
            kv_cache_k.stride(2), kv_cache_k.stride(3),
            block_table.stride(0), block_table.stride(1),
            S_block=s_block, block_size=block_size, num_heads=num_heads,
            num_kv_heads=num_kv_heads, head_dim=head_dim,
            softmax_scale=softmax_scale, G=G, BLOCK_Q=block_q, BLOCK_KV=block_kv,
            IS_CAUSAL=is_causal, num_warps=8, num_stages=2)
        return out

    part_acc = torch.empty((splits, T_q, num_heads, head_dim),
                           device=q.device, dtype=torch.float32)
    part_m = torch.empty((splits, T_q, num_heads), device=q.device, dtype=torch.float32)
    part_l = torch.empty((splits, T_q, num_heads), device=q.device, dtype=torch.float32)

    # Phase-1 (total_per_req_q == s_block here).
    grid_p = (q_blocks, num_kv_heads, num_reqs * splits)
    _tf_prefix_splitk_partial[grid_p](
        q, part_acc, part_m, part_l, kv_cache_k, kv_cache_v,
        block_table, prefix_lens,
        q.stride(0), q.stride(1), q.stride(2),
        part_acc.stride(0), part_acc.stride(1), part_acc.stride(2), part_acc.stride(3),
        part_m.stride(0), part_m.stride(1), part_m.stride(2),
        kv_cache_k.stride(0), kv_cache_k.stride(1),
        kv_cache_k.stride(2), kv_cache_k.stride(3),
        block_table.stride(0), block_table.stride(1),
        total_per_req_q=s_block, block_size=block_size,
        num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
        softmax_scale=softmax_scale, G=G, BLOCK_Q=block_q, SPLITS=splits)

    grid_r = (q_blocks, num_heads, num_reqs)
    _tf_reduce_tail[grid_r](
        q, part_acc, part_m, part_l, inline_k, inline_v, out,
        q.stride(0), q.stride(1), q.stride(2),
        part_acc.stride(0), part_acc.stride(1), part_acc.stride(2), part_acc.stride(3),
        part_m.stride(0), part_m.stride(1), part_m.stride(2),
        inline_k.stride(0), inline_k.stride(1), inline_k.stride(2),
        inline_v.stride(0), inline_v.stride(1), inline_v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        S_block=s_block, num_heads=num_heads, num_kv_heads=num_kv_heads,
        head_dim=head_dim, softmax_scale=softmax_scale, SPLITS=splits,
        BLOCK_Q=block_q, BLOCK_KV=block_kv, IS_CAUSAL=is_causal)
    return out
