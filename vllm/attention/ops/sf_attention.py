# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernel for TiDAR single-forward attention.

Implements the structured attention pattern from the TiDAR paper
(Figure 3 right) in a single kernel launch per attention layer,
replacing the multi-call FlashAttention path that currently runs as
2 flash_attn calls + per-layer prefix gather + cat.

Mask semantics (per-request):
  Layout: [verify (V=K+1) | prop_0 (Kp1=K+1) | ... | prop_{P-1} (Kp1)]
  KV layout: [prefix (N) | verify (V) | prop_0 | prop_1 | ... | prop_{P-1}]

  Verify query q (q_local < V):
    - prefix:                   ALLOW
    - verify slot j:            ALLOW iff j <= q_local (causal)
    - any proposal slot:        DENY

  Proposal query in block p (q_local >= V):
    p_idx = (q_local - V) // Kp1
    - prefix:                   ALLOW
    - verify slot j:            ALLOW iff j <= acc_levels[p_idx]
    - proposal block p_kv:      ALLOW iff p_kv == p_idx (bidirectional)

Inputs are INLINE K/V tensors (caller pre-gathers prefix from cache
and concatenates with the newly-projected verify + proposal K/V).
This avoids the paged-cache + Python-int slicing that prevented the
multi-call FA path from being cudagraph-capturable.

Expected throughput: ~140 tok/s captured at SF dense P=17 K+1 (vs
60 tok/s with FlexAttention or multi-call FA captured fallback).

WIP: kernel body written but un-tested on hardware; correctness test
against multi-call FA path in scripts/_sf_triton_smoke.py.
"""

from __future__ import annotations

from typing import Optional

import torch

from vllm.triton_utils import tl, triton

# -----------------------------------------------------------------------
# Kernel
# -----------------------------------------------------------------------

@triton.jit
def _sf_attention_fwd_kernel(
    # Pointers.
    Q,                  # [T_q, H_q, D] flat
    K,                  # [T_kv, H_kv, D] flat (inline: prefix + verify + props)
    V,                  # [T_kv, H_kv, D]
    Out,                # [T_q, H_q, D]
    # Per-request offsets (Python-int constexpr lists baked at build time
    # when num_reqs is small; for batch=1 these are just int values).
    cu_q_lens,          # [num_reqs+1] int32 -- cu_seqlens for Q
    cu_kv_lens,         # [num_reqs+1] int32 -- cu_seqlens for KV
    prefix_lens,        # [num_reqs] int32
    acc_levels,         # [P_props] int32
    # Strides.
    stride_q_t, stride_q_h, stride_q_d,
    stride_k_t, stride_k_h, stride_k_d,
    stride_v_t, stride_v_h, stride_v_d,
    stride_o_t, stride_o_h, stride_o_d,
    # Shape / mask constants.
    verify_len: tl.constexpr,
    Kp1: tl.constexpr,           # = K+1, proposal segment length
    P_props: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    softmax_scale: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    """One program processes BLOCK_Q queries of one head of one request.

    Grid: (cdiv(T_q_per_req, BLOCK_Q), num_heads, num_reqs).
    """
    pid_q = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_r = tl.program_id(2)

    # Per-request Q range.
    q_start = tl.load(cu_q_lens + pid_r)
    q_end = tl.load(cu_q_lens + pid_r + 1)
    q_len = q_end - q_start

    block_q_start = pid_q * BLOCK_Q
    if block_q_start >= q_len:
        return

    offs_q = block_q_start + tl.arange(0, BLOCK_Q)        # local-to-req
    q_mask = offs_q < q_len

    # Per-request KV range.
    kv_start = tl.load(cu_kv_lens + pid_r)
    kv_end = tl.load(cu_kv_lens + pid_r + 1)
    kv_len = kv_end - kv_start
    prefix_len = tl.load(prefix_lens + pid_r)

    # KV segment boundaries within the per-req kv span:
    #   [0, prefix_len)              = prefix
    #   [prefix_len, prefix_len + V) = verify
    #   [prefix_len + V, kv_len)     = proposals (P*Kp1 slots)
    v_seg_start = prefix_len
    v_seg_end = prefix_len + verify_len
    p_seg_start = v_seg_end

    # Q segment classification (verify vs proposal).
    q_is_verify = offs_q < verify_len                   # [BLOCK_Q] bool
    # For proposal queries: p_idx = (offs_q - V) // Kp1.
    q_local_minus_v = tl.maximum(offs_q - verify_len, 0)
    q_p_idx = q_local_minus_v // Kp1                    # 0 for verify (unused)
    q_within_in_block = q_local_minus_v - q_p_idx * Kp1

    # Map num_q_heads -> num_kv_heads (GQA).
    kv_head_id = pid_h * num_kv_heads // num_heads

    # Q offsets in global tensor.
    q_token_idx = q_start + offs_q
    offs_d = tl.arange(0, head_dim)
    q_ptrs = (
        Q
        + q_token_idx[:, None] * stride_q_t
        + pid_h * stride_q_h
        + offs_d[None, :] * stride_q_d
    )
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)
    q = q.to(tl.float32)

    # FlashAttention online softmax state. m_i is initialized to a large
    # finite negative (not -inf) so that blocks with no valid entries
    # produce alpha = exp(m_i - m_i) = 1 (no-op) instead of exp(-inf - -inf)
    # = NaN. SF mask logic guarantees the actually-valid block always
    # dominates this offset.
    NEG = -1.0e30
    m_i = tl.full((BLOCK_Q,), NEG, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_Q, head_dim), dtype=tl.float32)

    # Stream KV in blocks.
    for kv_block_start in range(0, kv_len, BLOCK_KV):
        offs_kv = kv_block_start + tl.arange(0, BLOCK_KV)
        kv_mask = offs_kv < kv_len

        # Load K, V for this block.
        kv_token_idx = kv_start + offs_kv
        k_ptrs = (
            K
            + kv_token_idx[:, None] * stride_k_t
            + kv_head_id * stride_k_h
            + offs_d[None, :] * stride_k_d
        )
        v_ptrs = (
            V
            + kv_token_idx[:, None] * stride_v_t
            + kv_head_id * stride_v_h
            + offs_d[None, :] * stride_v_d
        )
        k_block = tl.load(k_ptrs, mask=kv_mask[:, None], other=0.0).to(tl.float32)
        v_block = tl.load(v_ptrs, mask=kv_mask[:, None], other=0.0).to(tl.float32)

        # ---- Build SF mask: which (q, kv) pairs are allowed? ----
        # KV segment classification.
        kv_is_prefix = offs_kv < v_seg_start            # [BLOCK_KV]
        kv_is_verify = (offs_kv >= v_seg_start) & (offs_kv < v_seg_end)
        kv_is_prop = offs_kv >= p_seg_start
        kv_local_in_verify = offs_kv - v_seg_start      # only meaningful if kv_is_verify
        kv_local_in_prop = offs_kv - p_seg_start        # only meaningful if kv_is_prop
        kv_p_idx = tl.maximum(kv_local_in_prop, 0) // Kp1
        kv_within_in_block = kv_local_in_prop - kv_p_idx * Kp1

        # Build [BLOCK_Q, BLOCK_KV] mask.
        # Verify queries.
        verify_q_allows_prefix = kv_is_prefix[None, :]
        verify_q_allows_verify = (
            kv_is_verify[None, :]
            & (kv_local_in_verify[None, :] <= offs_q[:, None])
        )
        verify_q_mask = (
            q_is_verify[:, None]
            & (verify_q_allows_prefix | verify_q_allows_verify)
        )

        # Proposal queries.
        # acc_levels[q_p_idx] -- per-q gather.
        q_acc_level = tl.load(
            acc_levels + tl.minimum(q_p_idx, P_props - 1),
            mask=~q_is_verify, other=0
        )  # [BLOCK_Q]

        prop_q_allows_prefix = kv_is_prefix[None, :]
        prop_q_allows_verify = (
            kv_is_verify[None, :]
            & (kv_local_in_verify[None, :] <= q_acc_level[:, None])
        )
        prop_q_allows_own = (
            kv_is_prop[None, :]
            & (kv_p_idx[None, :] == q_p_idx[:, None])
        )
        prop_q_mask = (
            (~q_is_verify)[:, None]
            & (prop_q_allows_prefix | prop_q_allows_verify | prop_q_allows_own)
        )

        attn_mask = verify_q_mask | prop_q_mask
        attn_mask = attn_mask & q_mask[:, None] & kv_mask[None, :]

        # ---- Attention compute ----
        # qk: [BLOCK_Q, BLOCK_KV] (all fp32)
        qk = tl.dot(q, tl.trans(k_block)) * softmax_scale
        qk = tl.where(attn_mask, qk, float("-inf"))

        # Online softmax update.
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_new = alpha * l_i + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p, v_block)
        m_i = m_new
        l_i = l_new

    # Final normalize.
    acc = acc / l_i[:, None]

    # Write output.
    o_ptrs = (
        Out
        + q_token_idx[:, None] * stride_o_t
        + pid_h * stride_o_h
        + offs_d[None, :] * stride_o_d
    )
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=q_mask[:, None])


# -----------------------------------------------------------------------
# Padded-layout kernel (captured-mode safe).
#
# Per-req KV layout (fixed stride MAX_PREFIX + total_per_req_q):
#   [prefix_padded(MAX_PREFIX), verify(K+1), prop_0..P-1(P * (K+1))]
# Only positions [0..prefix_lens[r]) of the prefix region are valid;
# [prefix_lens[r]..MAX_PREFIX) is padding the caller filled with cache
# data the kernel masks out. Iterating over the full per-req KV span
# yields a static loop count, so the kernel is cudagraph-capturable.
#
# Early-skip: KV blocks entirely in the padding region [prefix_len,
# MAX_PREFIX) contribute nothing; we test once per block and skip the
# qk matmul + softmax update for those blocks.
# -----------------------------------------------------------------------

@triton.jit
def _sf_attention_fwd_kernel_padded(
    Q,                  # [T_q, H_q, D]
    K,                  # [num_reqs * (MAX_PREFIX + total_per_req_q), H_kv, D]
    V,
    Out,                # [T_q, H_q, D]
    prefix_lens,        # [num_reqs] int32 -- GPU value, runtime read
    acc_levels,         # [P_props] int32
    stride_q_t, stride_q_h, stride_q_d,
    stride_k_t, stride_k_h, stride_k_d,
    stride_v_t, stride_v_h, stride_v_d,
    stride_o_t, stride_o_h, stride_o_d,
    verify_len: tl.constexpr,
    Kp1: tl.constexpr,
    P_props: tl.constexpr,
    MAX_PREFIX: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    softmax_scale: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_r = tl.program_id(2)

    total_per_req_q: tl.constexpr = verify_len + P_props * Kp1
    kv_per_req: tl.constexpr = MAX_PREFIX + total_per_req_q

    # Q_START >= 0 lets callers skip the first Q_START queries per req
    # (e.g., to delegate verify-segment attention to FA3). The grid is
    # sized for cdiv(total_per_req_q - Q_START, BLOCK_Q), so pid_q
    # already iterates only the unskipped portion.
    block_q_start = Q_START + pid_q * BLOCK_Q
    if block_q_start >= total_per_req_q:
        return

    offs_q = block_q_start + tl.arange(0, BLOCK_Q)
    q_mask = offs_q < total_per_req_q

    # Per-req prefix length, runtime GPU read (captured-safe).
    prefix_len = tl.load(prefix_lens + pid_r)

    # Segment offsets within the per-req KV span (all constexpr).
    v_seg_start: tl.constexpr = MAX_PREFIX
    v_seg_end: tl.constexpr = MAX_PREFIX + verify_len
    p_seg_start: tl.constexpr = v_seg_end

    q_is_verify = offs_q < verify_len
    q_local_minus_v = tl.maximum(offs_q - verify_len, 0)
    q_p_idx = q_local_minus_v // Kp1
    q_within_in_block = q_local_minus_v - q_p_idx * Kp1

    kv_head_id = pid_h * num_kv_heads // num_heads

    q_token_idx = pid_r * total_per_req_q + offs_q
    offs_d = tl.arange(0, head_dim)
    q_ptrs = (
        Q
        + q_token_idx[:, None] * stride_q_t
        + pid_h * stride_q_h
        + offs_d[None, :] * stride_q_d
    )
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)

    NEG = -1.0e30
    m_i = tl.full((BLOCK_Q,), NEG, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_Q, head_dim), dtype=tl.float32)

    for kv_block_start in range(0, kv_per_req, BLOCK_KV):
        # Early-skip: if the entire block lies in the padding region
        # [prefix_len, MAX_PREFIX), nothing in it is valid for any query
        # type, so skip the load + matmul.
        kv_block_end = kv_block_start + BLOCK_KV
        in_padding = (kv_block_start >= prefix_len) & (
            kv_block_end <= v_seg_start)
        if not in_padding:
            offs_kv = kv_block_start + tl.arange(0, BLOCK_KV)
            kv_mask = offs_kv < kv_per_req

            kv_token_idx = pid_r * kv_per_req + offs_kv
            k_ptrs = (
                K
                + kv_token_idx[:, None] * stride_k_t
                + kv_head_id * stride_k_h
                + offs_d[None, :] * stride_k_d
            )
            v_ptrs = (
                V
                + kv_token_idx[:, None] * stride_v_t
                + kv_head_id * stride_v_h
                + offs_d[None, :] * stride_v_d
            )
            k_block = tl.load(k_ptrs, mask=kv_mask[:, None],
                              other=0.0).to(tl.float32)
            v_block = tl.load(v_ptrs, mask=kv_mask[:, None],
                              other=0.0).to(tl.float32)

            kv_is_actual_prefix = offs_kv < prefix_len
            kv_is_verify = (offs_kv >= v_seg_start) & (offs_kv < v_seg_end)
            kv_is_prop = offs_kv >= p_seg_start
            kv_local_in_verify = offs_kv - v_seg_start
            kv_local_in_prop = offs_kv - p_seg_start
            kv_p_idx = tl.maximum(kv_local_in_prop, 0) // Kp1
            kv_within_in_block = kv_local_in_prop - kv_p_idx * Kp1

            verify_q_mask = q_is_verify[:, None] & (
                kv_is_actual_prefix[None, :]
                | (kv_is_verify[None, :]
                   & (kv_local_in_verify[None, :] <= offs_q[:, None]))
            )
            q_acc_level = tl.load(
                acc_levels + tl.minimum(q_p_idx, P_props - 1),
                mask=~q_is_verify, other=0,
            )
            prop_q_mask = (~q_is_verify)[:, None] & (
                kv_is_actual_prefix[None, :]
                | (kv_is_verify[None, :]
                   & (kv_local_in_verify[None, :] <= q_acc_level[:, None]))
                | (kv_is_prop[None, :]
                   & (kv_p_idx[None, :] == q_p_idx[:, None]))
            )
            attn_mask = (verify_q_mask | prop_q_mask) & q_mask[:, None] & kv_mask[None, :]

            qk = tl.dot(q, tl.trans(k_block)) * softmax_scale
            qk = tl.where(attn_mask, qk, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
            l_new = alpha * l_i + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p, v_block)
            m_i = m_new
            l_i = l_new

    acc = acc / l_i[:, None]

    o_ptrs = (
        Out
        + q_token_idx[:, None] * stride_o_t
        + pid_h * stride_o_h
        + offs_d[None, :] * stride_o_d
    )
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=q_mask[:, None])


# -----------------------------------------------------------------------
# Compact-iteration kernel (captured-mode safe).
#
# Same padded buffer layout as ``_sf_attention_fwd_kernel_padded``, but
# iterates KV in two phases:
#   Phase 1 — prefix:        kv positions [0, prefix_lens[r]) (variable
#                             iteration count, set by GPU tensor).
#   Phase 2 — verify + props: kv positions [MAX_PREFIX,
#                             MAX_PREFIX + total_per_req_q)
#                             (fixed iteration count = constexpr).
#
# The padding band [prefix_lens[r], MAX_PREFIX) is SKIPPED entirely
# rather than iterated-then-masked. For short prefixes this slashes the
# kernel-side loop count (e.g., prefix=94, MAX_PREFIX=4096, BLOCK_KV=128
# drops from 35 iterations to 4).
# -----------------------------------------------------------------------

@triton.jit
def _sf_attention_fwd_kernel_compact(
    Q, K, V, Out,
    prefix_lens,
    acc_levels,
    stride_q_t, stride_q_h, stride_q_d,
    stride_k_t, stride_k_h, stride_k_d,
    stride_v_t, stride_v_h, stride_v_d,
    stride_o_t, stride_o_h, stride_o_d,
    verify_len: tl.constexpr,
    Kp1: tl.constexpr,
    P_props: tl.constexpr,
    MAX_PREFIX: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    softmax_scale: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_r = tl.program_id(2)

    total_per_req_q: tl.constexpr = verify_len + P_props * Kp1
    kv_per_req: tl.constexpr = MAX_PREFIX + total_per_req_q

    # Q_START >= 0 lets callers skip the first Q_START queries per req
    # (e.g., to delegate verify-segment attention to FA3). The grid is
    # sized for cdiv(total_per_req_q - Q_START, BLOCK_Q), so pid_q
    # already iterates only the unskipped portion.
    block_q_start = Q_START + pid_q * BLOCK_Q
    if block_q_start >= total_per_req_q:
        return

    offs_q = block_q_start + tl.arange(0, BLOCK_Q)
    q_mask = offs_q < total_per_req_q

    prefix_len = tl.load(prefix_lens + pid_r)

    q_is_verify = offs_q < verify_len
    q_local_minus_v = tl.maximum(offs_q - verify_len, 0)
    q_p_idx = q_local_minus_v // Kp1
    q_within_in_block = q_local_minus_v - q_p_idx * Kp1
    q_acc_level = tl.load(
        acc_levels + tl.minimum(q_p_idx, P_props - 1),
        mask=~q_is_verify, other=0,
    )

    kv_head_id = pid_h * num_kv_heads // num_heads

    q_token_idx = pid_r * total_per_req_q + offs_q
    offs_d = tl.arange(0, head_dim)
    q_ptrs = (
        Q
        + q_token_idx[:, None] * stride_q_t
        + pid_h * stride_q_h
        + offs_d[None, :] * stride_q_d
    )
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)

    NEG = -1.0e30
    m_i = tl.full((BLOCK_Q,), NEG, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_Q, head_dim), dtype=tl.float32)

    # ---- Phase 1: prefix [0, prefix_len). Both verify and prop Qs see
    # everything here (prefix is unconditionally visible). ----
    for kv_block_start in range(0, prefix_len, BLOCK_KV):
        offs_kv = kv_block_start + tl.arange(0, BLOCK_KV)
        kv_mask = offs_kv < prefix_len

        # Buffer index for prefix is offs_kv directly.
        kv_token_idx = pid_r * kv_per_req + offs_kv
        k_ptrs = (
            K
            + kv_token_idx[:, None] * stride_k_t
            + kv_head_id * stride_k_h
            + offs_d[None, :] * stride_k_d
        )
        v_ptrs = (
            V
            + kv_token_idx[:, None] * stride_v_t
            + kv_head_id * stride_v_h
            + offs_d[None, :] * stride_v_d
        )
        k_block = tl.load(k_ptrs, mask=kv_mask[:, None],
                          other=0.0).to(tl.float32)
        v_block = tl.load(v_ptrs, mask=kv_mask[:, None],
                          other=0.0).to(tl.float32)

        attn_mask = kv_mask[None, :] & q_mask[:, None]
        qk = tl.dot(q, tl.trans(k_block)) * softmax_scale
        qk = tl.where(attn_mask, qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_new = alpha * l_i + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p, v_block)
        m_i = m_new
        l_i = l_new

    # ---- Phase 2: verify + props at buffer offset MAX_PREFIX.
    # Position offsets within phase 2 are [0, total_per_req_q). ----
    v_seg_end: tl.constexpr = verify_len
    p_seg_start: tl.constexpr = verify_len

    for kv_phase2_start in range(0, total_per_req_q, BLOCK_KV):
        offs_p2 = kv_phase2_start + tl.arange(0, BLOCK_KV)
        kv_mask = offs_p2 < total_per_req_q

        # Buffer index = pid_r * kv_per_req + MAX_PREFIX + offs_p2.
        kv_token_idx = pid_r * kv_per_req + MAX_PREFIX + offs_p2
        k_ptrs = (
            K
            + kv_token_idx[:, None] * stride_k_t
            + kv_head_id * stride_k_h
            + offs_d[None, :] * stride_k_d
        )
        v_ptrs = (
            V
            + kv_token_idx[:, None] * stride_v_t
            + kv_head_id * stride_v_h
            + offs_d[None, :] * stride_v_d
        )
        k_block = tl.load(k_ptrs, mask=kv_mask[:, None],
                          other=0.0).to(tl.float32)
        v_block = tl.load(v_ptrs, mask=kv_mask[:, None],
                          other=0.0).to(tl.float32)

        # SF mask for phase 2 region (verify + props). Own-block among
        # proposals is causal: mask k sees mask 0..k.
        kv_is_verify = offs_p2 < v_seg_end
        kv_is_prop = offs_p2 >= p_seg_start
        kv_local_in_verify = offs_p2
        kv_local_in_prop = offs_p2 - p_seg_start
        kv_p_idx = tl.maximum(kv_local_in_prop, 0) // Kp1
        kv_within_in_block = kv_local_in_prop - kv_p_idx * Kp1

        verify_q_mask = q_is_verify[:, None] & (
            kv_is_verify[None, :]
            & (kv_local_in_verify[None, :] <= offs_q[:, None])
        )
        prop_q_mask = (~q_is_verify)[:, None] & (
            (kv_is_verify[None, :]
             & (kv_local_in_verify[None, :] <= q_acc_level[:, None]))
            | (kv_is_prop[None, :]
               & (kv_p_idx[None, :] == q_p_idx[:, None]))
        )
        attn_mask = ((verify_q_mask | prop_q_mask)
                     & q_mask[:, None] & kv_mask[None, :])

        qk = tl.dot(q, tl.trans(k_block)) * softmax_scale
        qk = tl.where(attn_mask, qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_new = alpha * l_i + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p, v_block)
        m_i = m_new
        l_i = l_new

    acc = acc / l_i[:, None]

    # Output index: recomputed since kv_token_idx clobbered the earlier
    # q_token_idx variable in the inner loops.
    out_q_token_idx = pid_r * total_per_req_q + offs_q
    o_ptrs = (
        Out
        + out_q_token_idx[:, None] * stride_o_t
        + pid_h * stride_o_h
        + offs_d[None, :] * stride_o_d
    )
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=q_mask[:, None])


# -----------------------------------------------------------------------
# Paged-cache kernel (no inline prefix gather).
#
# Reads prefix K/V directly from the paged cache via the per-request
# block_table; reads verify + proposal K/V from a small inline tensor
# (just the current step's projection). Eliminates the
# [num_reqs, MAX_PREFIX + total_per_req_q] padded buffer + its
# slice-assignment writes that dominated SF's GPU time (1213 ms in 100
# decode tokens — 76% of total).
#
# Per-(req, head, BLOCK_Q) program iterates KV in 2 phases as before
# (compact, no padding band):
#   Phase 1 — prefix [0, prefix_lens[r]): read paged via block_table[r].
#   Phase 2 — verify + props [0, total_per_req_q): read inline_kv.
# -----------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_Q": 16, "BLOCK_KV":  64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 16, "BLOCK_KV": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_KV":  64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_KV":  64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_KV": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_KV": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_Q": 32, "BLOCK_KV": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_Q": 64, "BLOCK_KV":  64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_Q": 64, "BLOCK_KV": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_Q": 64, "BLOCK_KV": 128}, num_warps=8, num_stages=3),
    ],
    key=["verify_len", "Kp1", "P_props", "num_heads", "head_dim", "block_size"],
)
@triton.jit
def _sf_attention_fwd_kernel_paged(
    Q,                  # [T_q, H_q, D]
    Inline_K,           # [num_reqs * total_per_req_q, H_kv, D]
    Inline_V,
    Out,                # [T_q, H_q, D]
    KV_CACHE_K,         # [num_total_blocks, block_size, H_kv, D]
    KV_CACHE_V,
    BlockTable,         # [num_reqs, max_blocks]
    prefix_lens,
    acc_levels,
    stride_q_t, stride_q_h, stride_q_d,
    stride_ink_t, stride_ink_h, stride_ink_d,
    stride_inv_t, stride_inv_h, stride_inv_d,
    stride_o_t, stride_o_h, stride_o_d,
    stride_kvc_block, stride_kvc_pos, stride_kvc_h, stride_kvc_d,
    stride_bt_r, stride_bt_b,
    verify_len: tl.constexpr,
    Kp1: tl.constexpr,
    P_props: tl.constexpr,
    block_size: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    softmax_scale: tl.constexpr,
    Q_START: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_r = tl.program_id(2)

    total_per_req_q: tl.constexpr = verify_len + P_props * Kp1

    # Q_START >= 0 lets callers skip the first Q_START queries per req
    # (e.g., to delegate verify-segment attention to FA3). The grid is
    # sized for cdiv(total_per_req_q - Q_START, BLOCK_Q), so pid_q
    # already iterates only the unskipped portion.
    block_q_start = Q_START + pid_q * BLOCK_Q
    if block_q_start >= total_per_req_q:
        return

    offs_q = block_q_start + tl.arange(0, BLOCK_Q)
    q_mask = offs_q < total_per_req_q

    prefix_len = tl.load(prefix_lens + pid_r)

    q_is_verify = offs_q < verify_len
    q_local_minus_v = tl.maximum(offs_q - verify_len, 0)
    q_p_idx = q_local_minus_v // Kp1
    # Within-own-block position (0..K). Used for causal-within-block
    # attention among proposal masks: mask k sees [bonus, mask_1, ...
    # mask_{k-1}], matching the TF drafter's causal pattern. Bidirectional
    # within-block leaks future drafts to the position-k draft and hurts
    # acceptance.
    q_within_in_block = q_local_minus_v - q_p_idx * Kp1
    q_acc_level = tl.load(
        acc_levels + tl.minimum(q_p_idx, P_props - 1),
        mask=~q_is_verify, other=0,
    )

    kv_head_id = pid_h * num_kv_heads // num_heads

    q_token_idx = pid_r * total_per_req_q + offs_q
    offs_d = tl.arange(0, head_dim)
    q_ptrs = (
        Q
        + q_token_idx[:, None] * stride_q_t
        + pid_h * stride_q_h
        + offs_d[None, :] * stride_q_d
    )
    q = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)

    NEG = -1.0e30
    m_i = tl.full((BLOCK_Q,), NEG, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_Q, head_dim), dtype=tl.float32)

    # ---- Phase 1: paged prefix read via block_table[pid_r]. ----
    # Iterate block_table indices [0..ceil(prefix_len/block_size)).
    # Within each block, read block_size positions (mask trailing ones
    # in the final block to stay within prefix_len).
    bt_row_ptr = BlockTable + pid_r * stride_bt_r
    n_prefix_blocks = (prefix_len + block_size - 1) // block_size
    for bt_idx in range(0, n_prefix_blocks):
        block_id = tl.load(bt_row_ptr + bt_idx * stride_bt_b).to(tl.int64)
        # Each block contributes block_size tokens, possibly truncated
        # for the last block of the prefix.
        offs_kv_in_blk = tl.arange(0, block_size)
        global_kv_pos = bt_idx * block_size + offs_kv_in_blk
        kv_mask = global_kv_pos < prefix_len

        kv_slot = (block_id * stride_kvc_block
                   + offs_kv_in_blk[:, None] * stride_kvc_pos
                   + kv_head_id * stride_kvc_h
                   + offs_d[None, :] * stride_kvc_d)
        # kv_slot is per-position; reshape pointers.
        k_ptrs = KV_CACHE_K + (
            block_id * stride_kvc_block
            + offs_kv_in_blk[:, None] * stride_kvc_pos
            + kv_head_id * stride_kvc_h
            + offs_d[None, :] * stride_kvc_d
        )
        v_ptrs = KV_CACHE_V + (
            block_id * stride_kvc_block
            + offs_kv_in_blk[:, None] * stride_kvc_pos
            + kv_head_id * stride_kvc_h
            + offs_d[None, :] * stride_kvc_d
        )
        k_block = tl.load(k_ptrs, mask=kv_mask[:, None],
                          other=0.0).to(tl.float32)
        v_block = tl.load(v_ptrs, mask=kv_mask[:, None],
                          other=0.0).to(tl.float32)

        # Prefix is unconditionally visible to verify and prop queries.
        attn_mask = kv_mask[None, :] & q_mask[:, None]

        qk = tl.dot(q, tl.trans(k_block)) * softmax_scale
        qk = tl.where(attn_mask, qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_new = alpha * l_i + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p, v_block)
        m_i = m_new
        l_i = l_new

    # ---- Phase 2: verify + props from contiguous inline buffer. ----
    p_seg_start: tl.constexpr = verify_len  # within inline buffer

    for kv_phase2_start in range(0, total_per_req_q, BLOCK_KV):
        offs_p2 = kv_phase2_start + tl.arange(0, BLOCK_KV)
        kv_mask = offs_p2 < total_per_req_q

        kv_token_idx = pid_r * total_per_req_q + offs_p2
        k_ptrs = (
            Inline_K
            + kv_token_idx[:, None] * stride_ink_t
            + kv_head_id * stride_ink_h
            + offs_d[None, :] * stride_ink_d
        )
        v_ptrs = (
            Inline_V
            + kv_token_idx[:, None] * stride_inv_t
            + kv_head_id * stride_inv_h
            + offs_d[None, :] * stride_inv_d
        )
        k_block = tl.load(k_ptrs, mask=kv_mask[:, None],
                          other=0.0).to(tl.float32)
        v_block = tl.load(v_ptrs, mask=kv_mask[:, None],
                          other=0.0).to(tl.float32)

        kv_is_verify = offs_p2 < verify_len
        kv_is_prop = offs_p2 >= p_seg_start
        kv_local_in_verify = offs_p2
        kv_local_in_prop = offs_p2 - p_seg_start
        kv_p_idx = tl.maximum(kv_local_in_prop, 0) // Kp1
        # Within-own-block KV position (0..K). 0 = bonus mask, 1..K =
        # draft masks. See q_within_in_block for the causal motivation.
        kv_within_in_block = kv_local_in_prop - kv_p_idx * Kp1

        verify_q_mask = q_is_verify[:, None] & (
            kv_is_verify[None, :]
            & (kv_local_in_verify[None, :] <= offs_q[:, None])
        )
        prop_q_mask = (~q_is_verify)[:, None] & (
            (kv_is_verify[None, :]
             & (kv_local_in_verify[None, :] <= q_acc_level[:, None]))
            | (kv_is_prop[None, :]
               & (kv_p_idx[None, :] == q_p_idx[:, None]))
        )
        attn_mask = ((verify_q_mask | prop_q_mask)
                     & q_mask[:, None] & kv_mask[None, :])

        qk = tl.dot(q, tl.trans(k_block)) * softmax_scale
        qk = tl.where(attn_mask, qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_new = alpha * l_i + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p, v_block)
        m_i = m_new
        l_i = l_new

    acc = acc / l_i[:, None]

    out_q_token_idx = pid_r * total_per_req_q + offs_q
    o_ptrs = (
        Out
        + out_q_token_idx[:, None] * stride_o_t
        + pid_h * stride_o_h
        + offs_d[None, :] * stride_o_d
    )
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=q_mask[:, None])


def sf_attention_triton_paged(
    q: torch.Tensor,            # [num_reqs * total_per_req_q, H, D]
    inline_k: torch.Tensor,     # [num_reqs * total_per_req_q, H_kv, D]
    inline_v: torch.Tensor,
    kv_cache_k: torch.Tensor,   # [num_total_blocks, block_size, H_kv, D]
    kv_cache_v: torch.Tensor,
    block_table: torch.Tensor,  # [num_reqs, max_blocks] int32
    prefix_lens: torch.Tensor,
    acc_levels: torch.Tensor,
    verify_len: int,
    K_drafts: int,
    P_props: int,
    block_size: int,
    softmax_scale: float,
    block_q: int = 32,
    block_kv: int = 128,
    out: Optional[torch.Tensor] = None,
    q_start: int = 0,
) -> torch.Tensor:
    """Paged-cache variant — no inline prefix buffer.

    Reads prefix K/V from the paged cache via ``block_table`` inside the
    Triton kernel, eliminating the slice-assignment writes that
    dominated SF's per-step GPU time.

    ``out`` (optional): caller-provided output buffer of shape
    [num_reqs * total_per_req_q, H, D] (same as q). When passed the
    kernel writes directly into it, avoiding an extra copy. If None,
    a fresh tensor is allocated.
    """
    T_q, num_heads, head_dim = q.shape
    Kp1 = K_drafts + 1
    total_per_req_q = verify_len + P_props * Kp1
    num_reqs = prefix_lens.shape[0]
    if out is None:
        out = torch.empty_like(q)

    # block_size must be a power-of-2 constexpr to match BLOCK_KV
    # constexpr conventions. Block sizes are typically 16; we don't
    # require BLOCK_KV == block_size here, but the prefix phase reads
    # block_size positions per loop iter.
    assert kv_cache_k.shape[1] == block_size
    # BLOCK_Q / BLOCK_KV / num_warps / num_stages are selected by
    # @triton.autotune on the kernel. The grid lambda receives the
    # tuned META dict and adjusts the q-axis tile count accordingly.
    # q_start>0 skips the first q_start queries per req (used by the
    # split-attention path that runs verify on FA3).
    q_work = total_per_req_q - q_start
    grid = lambda META: (
        triton.cdiv(q_work, META["BLOCK_Q"]),
        num_heads, num_reqs)
    _sf_attention_fwd_kernel_paged[grid](
        q, inline_k, inline_v, out,
        kv_cache_k, kv_cache_v,
        block_table, prefix_lens, acc_levels,
        q.stride(0), q.stride(1), q.stride(2),
        inline_k.stride(0), inline_k.stride(1), inline_k.stride(2),
        inline_v.stride(0), inline_v.stride(1), inline_v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        kv_cache_k.stride(0), kv_cache_k.stride(1),
        kv_cache_k.stride(2), kv_cache_k.stride(3),
        block_table.stride(0), block_table.stride(1),
        verify_len=verify_len,
        Kp1=Kp1,
        P_props=P_props,
        block_size=block_size,
        num_heads=num_heads,
        num_kv_heads=inline_v.shape[1],
        head_dim=head_dim,
        softmax_scale=softmax_scale,
        Q_START=q_start,
    )
    return out


def sf_attention_triton_compact(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    prefix_lens: torch.Tensor,
    acc_levels: torch.Tensor,
    verify_len: int,
    K_drafts: int,
    P_props: int,
    MAX_PREFIX: int,
    softmax_scale: float,
    block_q: int = 32,
    block_kv: int = 128,
) -> torch.Tensor:
    """Compact-iteration variant of ``sf_attention_triton_padded``.

    Same padded buffer layout (so the caller is unchanged) but the
    kernel uses a 2-phase loop that skips the padding band between
    actual prefix end and MAX_PREFIX. Wins are largest when actual
    prefix is much smaller than MAX_PREFIX.
    """
    T_q, num_heads, head_dim = q.shape
    Kp1 = K_drafts + 1
    total_per_req_q = verify_len + P_props * Kp1
    num_reqs = prefix_lens.shape[0]
    out = torch.empty_like(q)

    grid = (triton.cdiv(total_per_req_q, block_q), num_heads, num_reqs)
    _sf_attention_fwd_kernel_compact[grid](
        q, k, v, out,
        prefix_lens, acc_levels,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        verify_len=verify_len,
        Kp1=Kp1,
        P_props=P_props,
        MAX_PREFIX=MAX_PREFIX,
        num_heads=num_heads,
        num_kv_heads=v.shape[1],
        head_dim=head_dim,
        softmax_scale=softmax_scale,
        BLOCK_Q=block_q,
        BLOCK_KV=block_kv,
    )
    return out


def sf_attention_triton_padded(
    q: torch.Tensor,           # [num_reqs * total_per_req_q, H, D]
    k: torch.Tensor,           # [num_reqs * (MAX_PREFIX + total_per_req_q), H_kv, D]
    v: torch.Tensor,
    prefix_lens: torch.Tensor, # [num_reqs] int32 GPU
    acc_levels: torch.Tensor,  # [P_props] int32 GPU
    verify_len: int,
    K_drafts: int,
    P_props: int,
    MAX_PREFIX: int,
    softmax_scale: float,
    block_q: int = 32,
    block_kv: int = 128,
) -> torch.Tensor:
    """Padded-layout SF attention. KV stride per req is constexpr
    (MAX_PREFIX + verify + P*Kp1) so the kernel is cudagraph-capturable
    while still respecting per-request runtime ``prefix_lens``.
    """
    T_q, num_heads, head_dim = q.shape
    Kp1 = K_drafts + 1
    total_per_req_q = verify_len + P_props * Kp1
    num_reqs = prefix_lens.shape[0]
    out = torch.empty_like(q)

    grid = (triton.cdiv(total_per_req_q, block_q), num_heads, num_reqs)
    _sf_attention_fwd_kernel_padded[grid](
        q, k, v, out,
        prefix_lens, acc_levels,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        verify_len=verify_len,
        Kp1=Kp1,
        P_props=P_props,
        MAX_PREFIX=MAX_PREFIX,
        num_heads=num_heads,
        num_kv_heads=v.shape[1],
        head_dim=head_dim,
        softmax_scale=softmax_scale,
        BLOCK_Q=block_q,
        BLOCK_KV=block_kv,
    )
    return out


# -----------------------------------------------------------------------
# Python wrapper
# -----------------------------------------------------------------------

def sf_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q_lens: torch.Tensor,
    cu_kv_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    acc_levels: torch.Tensor,
    verify_len: int,
    K_drafts: int,
    P_props: int,
    softmax_scale: float,
    block_q: int = 32,
    block_kv: int = 128,
) -> torch.Tensor:
    """Single-kernel SF attention.

    Args:
      q: [T_q, num_heads, head_dim] -- concatenated per-req queries
        in the order [verify, prop_0, ..., prop_{P-1}] per req.
      k, v: [T_kv, num_kv_heads, head_dim] -- per-req KV in order
        [prefix, verify, prop_0, ..., prop_{P-1}] per req.
      cu_q_lens: [num_reqs+1] int32 cumulative q-len.
      cu_kv_lens: [num_reqs+1] int32 cumulative kv-len.
      prefix_lens: [num_reqs] int32.
      acc_levels: [P_props] int32 -- acceptance level for each proposal.
      verify_len: K+1 (anchor + K drafts).
      K_drafts: K, the number of speculative tokens.
      P_props: number of proposals.

    Returns:
      out: [T_q, num_heads, head_dim].
    """
    T_q, num_heads, head_dim = q.shape
    T_kv, num_kv_heads, head_dim_k = k.shape
    assert head_dim == head_dim_k
    assert v.shape == k.shape
    assert num_heads % num_kv_heads == 0, "GQA: num_heads divisible by num_kv_heads"

    Kp1 = K_drafts + 1
    num_reqs = prefix_lens.shape[0]

    out = torch.empty_like(q)

    # Grid: (cdiv(max_q_per_req, BLOCK_Q), num_heads, num_reqs).
    # We bound by max q_len; the kernel exits early past the per-req len.
    max_q_per_req = verify_len + P_props * Kp1
    grid = (triton.cdiv(max_q_per_req, block_q), num_heads, num_reqs)

    _sf_attention_fwd_kernel[grid](
        q, k, v, out,
        cu_q_lens, cu_kv_lens, prefix_lens, acc_levels,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        verify_len=verify_len,
        Kp1=Kp1,
        P_props=P_props,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        softmax_scale=softmax_scale,
        BLOCK_Q=block_q,
        BLOCK_KV=block_kv,
    )
    return out
