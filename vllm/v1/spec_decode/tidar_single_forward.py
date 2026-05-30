# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TiDAR single-forward primitives: mask, input layout, slot mapping.

Implements the sparse-proposal single-forward design from
``docs/tidar_single_forward_design_2026-05-13.md``. Per request, one
model forward processes ``K * (1 + P)`` tokens laid out as

    [verify (K) | proposal_1 (K) | proposal_2 (K) | ... | proposal_P (K)]

where ``P`` and the per-proposal acc levels ``{p_1, ..., p_P}`` are
runtime-tunable (subject to the captured-cudagraph shape constraint:
changing ``P`` requires recapture; changing levels does not).

This module is deliberately framework-light: functions take raw tensors
in / out so they can be unit-tested without spinning up a vLLM engine.
The runner-side glue (CommonAttentionMetadata, scratch block tables,
FlexAttention metadata builder hooks) lives upstream.
"""

from typing import Optional

import torch

# Sentinel used by FlashAttention / PagedAttention to mean
# "don't reshape_and_cache this token". Mirrors
# ``vllm.v1.attention.backends.utils.PAD_SLOT_ID``.
PAD_SLOT_ID = -1


# ----------------------------------------------------------------------
# 1. Attention mask
# ----------------------------------------------------------------------

def tidar_mask_mod(
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
    prefix_len: int,
    K: int,
    proposal_acc_levels: torch.Tensor,
    verify_len: Optional[int] = None,
    no_bonus_layout: bool = False,
) -> torch.Tensor:
    """Tensor-friendly TiDAR single-forward attention mask.

    Designed as the ``logical_mask_mod`` for vLLM's FlexAttention backend.
    All scalar / broadcastable tensor inputs work; the function also
    accepts ``[Q, 1]`` / ``[1, KV]`` tensors so unit tests can build
    the full mask grid in one shot.

    Layout per request (in FA logical-position space):

        [prefix (prefix_len), verify (verify_len), prop_1 (K), ..., prop_P (K)]

    where ``verify_len`` defaults to ``K`` (no anchor) but is most
    commonly ``K + 1`` for production TiDAR (= anchor + K drafts; the
    anchor is the previous step's bonus token, re-fed so the verifier
    re-derives its logits for verifying ``d_1``).

    Args:
        q_idx:               query logical position (scalar or [Q, 1])
        kv_idx:              kv logical position (scalar or [1, KV])
        prefix_len:          # of cached prefix tokens in this request
        K:                   drafts per proposal-mask block
        proposal_acc_levels: int tensor of shape [P]. Semantics:
            ``acc_levels[p]`` = "kv_local positions visible to proposal p
            in the verify segment". Under the K+1 verify convention
            (anchor at kv_local=0, drafts at 1..K), an acc level of
            ``j`` means the proposal sees ``anchor + j drafts`` =
            ``j+1`` verify positions in total (so ``kv_local <= j``).
            This matches the existing ``commit_spec_decode_state``
            semantics where ``num_accepted=j`` means j drafts accepted.
        verify_len:          length of the verify segment per request.
            None -> defaults to ``K`` (no-anchor convention; only used
            by the early standalone tests). Pass ``K + 1`` for the
            production K+1-input verify forward.

    Returns:
        bool tensor: True iff query at q_idx may attend to kv at kv_idx.
    """
    if verify_len is None:
        verify_len = K

    # Prefix is always visible to every new-token query.
    kv_in_prefix = kv_idx < prefix_len

    # Indices in the new-token region. Negative for prefix kvs; safe
    # because the final OR masks those rows back to True via kv_in_prefix.
    q_local = q_idx - prefix_len
    kv_local = kv_idx - prefix_len

    # Segment classification: 0 = verify, 1..P = proposal_p (p = seg-1).
    # We compute it directly from q_local rather than ``q_local // K`` so
    # that ``verify_len`` (which may differ from K, e.g., K+1 with anchor)
    # doesn't get misclassified at the verify/proposal boundary.
    q_is_verify = q_local < verify_len
    kv_is_verify = kv_local < verify_len

    # Layout-mode signaled by closure-captured ``no_bonus_layout``
    # (Python bool baked at metadata-build time from VLLM_TIDAR_NO_BONUS):
    #   False -> default K+1 layout; proposal_seg_len = K+1 (slot 0 at
    #            bonus position), verify is causal.
    #   True  -> hybrid K-mask + no-bonus; proposal_seg_len = K (paper-
    #            aligned), verify still K+1 + causal (slot 0 = anchor =
    #            latest accepted token, ignored by rejection sampler).
    # ZAP-ONLY variant: always use K+1 layout. no_bonus_layout
    # parameter is ignored; only the rejection sampler bonus zap
    # differs from default.
    proposal_seg_len = K + 1
    q_prop_local = q_local - verify_len  # negative when q is verify
    q_p_index = torch.clamp(q_prop_local, min=0) // proposal_seg_len
    kv_prop_local = kv_local - verify_len
    kv_p_index = torch.clamp(kv_prop_local, min=0) // proposal_seg_len

    # Verify-segment query: causal in both modes. Slot 0 attends to
    # prefix only; slot k attends to prefix + slots 0..k. (Under hybrid
    # no_bonus this is fine: slot 0 is the latest-accepted-token re-
    # processed, slots 1..K are the K drafts; rejection sampler skips
    # slot 0's bonus output.)
    verify_mask = kv_is_verify & (kv_local <= q_local)

    # Proposal-segment query: gather p_j (acc level) by q_p_index.
    p_j = proposal_acc_levels[q_p_index]

    # Proposal sees verify[:p_j+1] (= anchor + p_j drafts in both
    # layouts: anchor at slot 0, p_j drafts at slots 1..p_j).
    prop_sees_verify = kv_is_verify & (kv_local <= p_j)
    # Proposal-to-own-block: CAUSAL (not bidirectional). The SMoE ckpt
    # was trained with SBD pattern only during training; inference is
    # deployed with regular causal attention (eval_cmoe doesn't apply
    # the SBD mask). Bidirectional within proposal block is off-
    # distribution and produces 0% acceptance on this ckpt.
    prop_sees_self = (
        (~kv_is_verify)
        & (kv_p_index == q_p_index)
        & (kv_prop_local <= q_prop_local))
    prop_mask = prop_sees_verify | prop_sees_self

    new_token_mask = torch.where(q_is_verify, verify_mask, prop_mask)

    return kv_in_prefix | new_token_mask


def make_tidar_mask_mod_factory(
    prefix_lens: torch.Tensor,
    request_lookup: torch.Tensor,
    K: int,
    proposal_acc_levels: torch.Tensor,
    verify_len: Optional[int] = None,
    no_bonus_layout: bool = False,
):
    """Wrap ``tidar_mask_mod`` for flex_attention's per-call signature.

    flex_attention invokes ``mask_mod(b, h, q_idx, kv_idx) -> bool`` with
    scalar tensors. We close over per-request prefix lengths and look up
    the right one via ``request_lookup[q_idx]`` -- the same pattern the
    existing FlexAttention backend uses (``_offsets_to_doc_ids_tensor``
    on ``query_start_loc``).

    Args:
        prefix_lens:         [num_reqs] int -- decode_offset per request
        request_lookup:      [num_queries] int -- maps q_idx -> req idx
        K:                   drafts per proposal-mask block
        proposal_acc_levels: [P] int -- "num drafts accepted" semantics
        verify_len:          length of the verify segment per request.
            None -> defaults to ``K`` (no-anchor convention). Pass
            ``K + 1`` for production TiDAR (anchor + K drafts).

    Returns:
        function with the flex_attention mask_mod signature.
    """

    def mask_mod(
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        del b, h
        prefix_len = prefix_lens[request_lookup[q_idx]]
        return tidar_mask_mod(
            q_idx, kv_idx, prefix_len, K, proposal_acc_levels,
            verify_len=verify_len, no_bonus_layout=no_bonus_layout)

    return mask_mod


# ----------------------------------------------------------------------
# 2. Input layout builder
# ----------------------------------------------------------------------

def compute_position_offsets(
    K: int,
    proposal_acc_levels: list[int],
    verify_len: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.int64,
) -> torch.Tensor:
    """Position-ID offsets (rotary positions) within one request's new
    tokens.

    Per request, given prefix_len ``N``, the rotary position of new-token
    slot ``s`` is ``N + offsets[s]``. Layout:

      offsets[0 .. verify_len-1]                 = [0, 1, ..., verify_len-1]
      offsets[verify_len + p*(K+1) + j]          = p_j + (anchor_base) + j

    Each proposal contributes ``K + 1`` mask positions: one mask at the
    bonus position followed by K masks at the K subsequent draft
    positions. This mirrors two-forward TiDAR's drafter input layout
    ``[next_token, mask x K]`` -- the model needs a contiguous prefix
    of (real_or_mask)+(masks) to predict K future tokens correctly.
    Earlier SF used only K masks per proposal with a GAP at the bonus
    position; the model is trained for the K+1 layout and rejection
    cascades when the bonus position is missing. See TF Fix 4 for
    the analogous TF-side bug.

    Layout (anchor convention, ``verify_len = K + 1``):
      verify: anchor at offset 0, drafts at offsets 1..K.
      proposal p (``p_j`` drafts accepted-by-design):
        mask 0   at offset p_j + 1            (bonus position; consumed
                                               and discarded at extract).
        mask 1   at offset p_j + 2            (= next step's d_1).
        ...
        mask K   at offset p_j + K + 1        (= next step's d_K).

    Layout (no-anchor convention, ``verify_len = K``):
      verify: drafts at offsets 0..K-1.
      proposal p:
        mask 0   at offset p_j                (bonus position).
        mask 1   at offset p_j + 1            (= next step's d_1).
        ...
        mask K   at offset p_j + K            (= next step's d_K).

    Args:
        K:                   drafts per proposal-mask block
        proposal_acc_levels: list of P int acc levels (semantics =
            "num drafts accepted" for both conventions)
        verify_len:          length of verify segment. None -> ``K`` (
            no-anchor convention).

    Shape: [verify_len + P * (K + 1)].
    """
    if verify_len is None:
        verify_len = K
    has_anchor = verify_len == K + 1
    if verify_len not in (K, K + 1):
        raise ValueError(
            f"verify_len must be K ({K}) or K+1 ({K+1}); got {verify_len}")
    P = len(proposal_acc_levels)
    # Hybrid no_bonus: verify_len stays K+1 (anchor at slot 0), but
    # proposal_seg_len = K (paper-aligned, drops the bonus-position
    # mask). Default: K+1 proposal masks.
    # ZAP-ONLY variant: K+1 layout regardless of no_bonus env.
    proposal_seg_len = K + 1
    total = verify_len + P * proposal_seg_len
    offsets = torch.empty(total, dtype=dtype, device=device)
    arange_seg = torch.arange(proposal_seg_len, dtype=dtype, device=device)
    # Verify segment: 0..verify_len-1.
    offsets[:verify_len] = torch.arange(verify_len, dtype=dtype, device=device)
    # Proposal mask RoPE offsets:
    #   default (K+1):   [p_j + 1 .. p_j + K + 1]   slot 0 at the bonus
    #                    position, slots 1..K predict next step's drafts
    #   no_bonus (K):    [p_j + 1 .. p_j + K]       all K slots predict
    #                    next step's drafts (no bonus mask). Same base
    #                    offset, K positions instead of K+1.
    proposal_offset_base = 1 if has_anchor else 0
    for p_idx, p_j in enumerate(proposal_acc_levels):
        start = verify_len + p_idx * proposal_seg_len
        offsets[start:start + proposal_seg_len] = (
            p_j + proposal_offset_base + arange_seg)
    return offsets


def build_single_forward_inputs(
    verify_token_ids: torch.Tensor,
    base_positions: torch.Tensor,
    block_table: torch.Tensor,
    K: int,
    proposal_acc_levels: list[int],
    mask_token_id: int,
    block_size: int,
    max_model_len: int,
    scratch_block_table: Optional[torch.Tensor] = None,
    verify_len: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the per-request single-forward input layout.

    Args:
        verify_token_ids:      [B, verify_len] int -- the verify-segment
                               token IDs. For the production K+1 anchor
                               convention this is ``[anchor, d_1, ..., d_K]``
                               (anchor = bonus from last step). For
                               no-anchor tests this is the K drafts.
        base_positions:        [B] int -- prefix_len per request (= N = the
                               sequence position of the first verify token)
        block_table:           [B, max_blocks] int -- per-req paged-cache
                               block IDs (for verify writes)
        K:                     drafts per proposal-mask block
        proposal_acc_levels:   list of P int acc levels ("num drafts accepted"
                               semantics)
        mask_token_id:         int -- token ID for proposal mask positions
        block_size:            int -- paged cache block size
        max_model_len:         int -- clip positions exceeding this to
                               PAD_SLOT_ID
        scratch_block_table:   optional [B, P] int -- scratch block IDs
                               per proposal per request. If None, proposal
                               slot_mapping is PAD_SLOT_ID.
        verify_len:            length of the verify segment per req.
                               Defaults to ``K`` (no-anchor). Pass
                               ``K + 1`` for production TiDAR.

    Returns:
        input_ids:     [B * (verify_len + P*(K+1))] int
        positions:     [B * (verify_len + P*(K+1))] int (rotary position IDs)
        slot_mapping:  [B * (verify_len + P*(K+1))] int (paged-cache slots)
    """
    if verify_len is None:
        verify_len = K
    if verify_token_ids.dim() != 2 or \
            verify_token_ids.shape[1] != verify_len:
        raise ValueError(
            f"verify_token_ids must be [B, verify_len={verify_len}]; got "
            f"shape {tuple(verify_token_ids.shape)} verify_len={verify_len}")
    if base_positions.dim() != 1 or base_positions.shape[0] != \
            verify_token_ids.shape[0]:
        raise ValueError(
            f"base_positions must be [B] matching verify_token_ids[0]; "
            f"got shape {tuple(base_positions.shape)}")

    B = verify_token_ids.shape[0]
    P = len(proposal_acc_levels)
    # Hybrid no_bonus: verify_len stays K+1 (anchor slot 0 +
    # K drafts); proposal_seg_len = K (paper-aligned). Default:
    # verify K+1 + proposal K+1.
    # ZAP-ONLY variant: K+1 layout regardless of no_bonus env.
    proposal_seg_len = K + 1
    total_per_req = verify_len + P * proposal_seg_len
    device = verify_token_ids.device
    pos_dtype = base_positions.dtype

    if scratch_block_table is not None:
        # ZAP-ONLY: K+1 layout always; need 2 sub-blocks per proposal.
        min_sub_blocks = 2
        if scratch_block_table.dim() != 3 \
                or scratch_block_table.shape[:2] != (B, P) \
                or scratch_block_table.shape[2] < min_sub_blocks:
            raise ValueError(
                f"scratch_block_table must be [B={B}, P={P}, >={min_sub_blocks}]; "
                f"got {tuple(scratch_block_table.shape)}. "
                f"need K+1 mask slots per proposal.")

    # --- positions (rotary) --------------------------------------------
    offsets = compute_position_offsets(
        K=K, proposal_acc_levels=proposal_acc_levels,
        verify_len=verify_len, device=device, dtype=pos_dtype)
    positions = base_positions.view(B, 1) + offsets.view(1, total_per_req)
    # Sequence-length safety: clamp / detect any token that would land
    # past max_model_len so the caller can mask its slot_mapping.
    exceeds_max = positions >= max_model_len
    clamped_pos = torch.where(exceeds_max, torch.zeros_like(positions),
                              positions)

    # --- input_ids -----------------------------------------------------
    input_ids = torch.full(
        (B, total_per_req), mask_token_id,
        dtype=verify_token_ids.dtype, device=device)
    input_ids[:, :verify_len] = verify_token_ids

    # --- slot_mapping --------------------------------------------------
    # Verify portion uses real block table (writes go to AR slots).
    verify_pos = clamped_pos[:, :verify_len]  # [B, verify_len]
    verify_block_numbers = verify_pos // block_size
    verify_block_ids = block_table.gather(dim=1, index=verify_block_numbers)
    verify_slot = (verify_block_ids * block_size +
                   verify_pos % block_size)  # [B, verify_len]
    verify_slot = torch.where(exceeds_max[:, :verify_len],
                              torch.full_like(verify_slot, PAD_SLOT_ID),
                              verify_slot)

    # Proposal portion: scratch slots if provided, else PAD_SLOT_ID.
    # Each proposal has K+1 mask slots that span across 2 scratch blocks
    # (for K == block_size). Slots [0..K-1] go to scratch_block_table[i,p,0]
    # at offsets [0..K-1]; slot K goes to scratch_block_table[i,p,1] at
    # offset 0.
    if scratch_block_table is None:
        prop_slot = torch.full(
            (B, P * proposal_seg_len), PAD_SLOT_ID,
            dtype=verify_slot.dtype, device=device)
    else:
        if K > 2 * block_size:
            # 2-block scratch fits K+1 slots when K <= 2 * block_size.
            # Beyond that we'd need a 3+-block layout (currently unused).
            raise NotImplementedError(
                f"K={K} exceeds 2 * block_size={2 * block_size}; SF scratch "
                f"layout currently assumes K+1 slots fit in 2 cache blocks. "
                f"Extend scratch_block_table's last dim if larger K needed.")
        # block_ids_bp: [B, P, 2]
        block_ids_bp = scratch_block_table
        # For each of the K+1 slots: which of the 2 blocks, what offset.
        # slot j (j in 0..K): block_in_pair = j // block_size, offset = j % block_size.
        arange_Kp1 = torch.arange(proposal_seg_len,
                                  dtype=verify_slot.dtype, device=device)
        block_in_pair = arange_Kp1 // block_size  # [K+1]
        offset_in_block = arange_Kp1 % block_size  # [K+1]
        # block_id_per_slot[i, p, j] = block_ids_bp[i, p, block_in_pair[j]]
        block_id_per_slot = block_ids_bp.gather(
            dim=2,
            index=block_in_pair.to(torch.int64)
                  .view(1, 1, proposal_seg_len).expand(B, P, proposal_seg_len))
        prop_slot = (block_id_per_slot * block_size +
                     offset_in_block.view(1, 1, proposal_seg_len)).view(
            B, P * proposal_seg_len)
        # Apply max_model_len safety to the proposal portion too.
        exceeds_prop = exceeds_max[:, verify_len:]
        prop_slot = torch.where(exceeds_prop,
                                torch.full_like(prop_slot, PAD_SLOT_ID),
                                prop_slot)

    slot_mapping = torch.cat([verify_slot, prop_slot], dim=1)  # [B, total]

    # Flatten to 1-D as vLLM's runner expects.
    return (input_ids.view(-1), positions.view(-1).to(pos_dtype),
            slot_mapping.view(-1))


def extract_proposal_hidden_states(
    hidden_states: torch.Tensor,
    proposal_indices: torch.Tensor,
    K: int,
    verify_len: int,
    P_props: int,
) -> torch.Tensor:
    """Pick the K mask hidden states from the selected proposal per req.

    Per-request layout in ``hidden_states`` (the single-forward output):

        rows [0, verify_len)                       -- verify segment
        rows [verify_len, verify_len + (K+1))      -- proposal_1 (K+1 masks)
        rows [verify_len + (K+1), verify_len + 2*(K+1))  -- proposal_2
        ...
        rows [verify_len + (P-1)*(K+1), verify_len + P*(K+1)) -- proposal_P

    Each proposal contributes ``K + 1`` mask hidden states (mask 0 = bonus
    prediction, masks 1..K = K drafts). We keep only masks 1..K -- the
    bonus prediction is discarded because the rejection sampler gets the
    actual bonus from the verifier segment's logit at verify position p_j.

    Flattened across requests: req i's rows occupy
    ``[i * total_per_req, (i+1) * total_per_req)`` where
    ``total_per_req = verify_len + P_props * (K + 1)``.

    Args:
        hidden_states:     [B * total_per_req, hidden_size]
        proposal_indices:  [B] int in [0, P_props) -- selected proposal
                           index per request (output of
                           ``select_proposal_index``)
        K:                 drafts per proposal-mask block
        verify_len:        length of verify segment (K or K+1)
        P_props:           number of proposals per req

    Returns:
        [B * K, hidden_size] -- the selected proposal's K draft-mask
        hidden states per request (drops the leading bonus-mask),
        flattened. Caller feeds these to the drafter logits + sampler
        to produce next step's drafts.
    """
    if proposal_indices.dim() != 1:
        raise ValueError("proposal_indices must be 1-D")
    B = proposal_indices.shape[0]
    # ZAP-ONLY: K+1 layout always.
    no_bonus = False
    proposal_seg_len = K + 1
    total_per_req = verify_len + P_props * proposal_seg_len
    if hidden_states.shape[0] != B * total_per_req:
        raise ValueError(
            f"hidden_states[0] = {hidden_states.shape[0]} != "
            f"B * total_per_req = {B} * {total_per_req} = "
            f"{B * total_per_req}")
    H = hidden_states.shape[1]
    device = hidden_states.device

    # Reshape to [B, total_per_req, H] then slice the proposal section.
    hs_3d = hidden_states.view(B, total_per_req, H)
    # Drop the verify segment; keep only [B, P_props * (K+1), H].
    proposals_only = hs_3d[:, verify_len:, :]
    proposals_4d = proposals_only.view(B, P_props, proposal_seg_len, H)

    # Gather along the P_props axis using proposal_indices.
    # proposal_indices: [B]; result: [B, K+1, H].
    selected = proposals_4d[torch.arange(B, device=device), proposal_indices]
    if no_bonus:
        # K-mask layout: all K slots are draft predictions (slot k
        # predicts position p_j+k, which is next step's d_{k+1} at
        # base_{n+1}+k).
        return selected.reshape(B * K, H).contiguous()
    # K+1 default: drop slot 0 (the bonus-position prediction); keep
    # slots 1..K as the K draft predictions.
    drafts_only = selected[:, 1:, :]  # [B, K, H]
    return drafts_only.reshape(B * K, H).contiguous()


def select_proposal_index(
    num_accepted: torch.Tensor,
    proposal_acc_levels: torch.Tensor,
    tie_break: str = "closest_below",
) -> torch.Tensor:
    """Pick the proposal index whose acc level best matches actual acc.

    Args:
        num_accepted:        [B] int -- actual num_accepted per request
        proposal_acc_levels: [P] int -- chosen acc levels (sorted asc)
        tie_break:           "closest_below" or "closest" (abs distance).
                             Default closest_below prefers over-conditioning
                             (proposal saw fewer drafts than actual accept)
                             over under-conditioning -- empirically lower
                             quality cost when the rejected drafts are
                             only slightly off-distribution.

    Returns:
        [B] int -- chosen proposal index in [0, P-1].
    """
    if proposal_acc_levels.dim() != 1:
        raise ValueError("proposal_acc_levels must be 1-D")
    if num_accepted.dim() != 1:
        raise ValueError("num_accepted must be 1-D")

    # Distance per (req, proposal).
    diff = (num_accepted.view(-1, 1) -
            proposal_acc_levels.view(1, -1))  # [B, P]

    if tie_break == "closest_below":
        # Prefer non-positive diff (level <= actual acc). Penalize
        # negative diff slightly more than positive of the same |d|.
        score = torch.where(diff >= 0, diff.float(), -diff.float() * 1.001)
    elif tie_break == "closest":
        score = diff.abs().float()
    else:
        raise ValueError(f"unknown tie_break: {tie_break!r}")

    return score.argmin(dim=1)
