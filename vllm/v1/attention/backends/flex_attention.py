# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with FlexAttention."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Optional, Union

import torch
import torch._dynamo.decorators
import torch.nn.functional as F
from torch.nn.attention.flex_attention import (BlockMask, _mask_mod_signature,
                                               _score_mod_signature, and_masks,
                                               create_block_mask,
                                               flex_attention)

from vllm.v1.attention.backend import (AttentionBackend, AttentionImpl,
                                       AttentionMetadata, AttentionType,
                                       is_quantized_kv_cache)
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import is_torch_equal_or_newer
from vllm.v1.attention.backend import (AttentionCGSupport,
                                       AttentionMetadataBuilder,
                                       CommonAttentionMetadata)
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.spec_decode.tidar_single_forward import tidar_mask_mod

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import InputBatch

create_block_mask_compiled = torch.compile(create_block_mask,
                                           fullgraph=True,
                                           mode="reduce-overhead")
flex_attention_compiled = torch.compile(flex_attention, fullgraph=True)


def _offsets_to_doc_ids_tensor(offsets: torch.Tensor) -> torch.Tensor:
    device = offsets.device
    counts = offsets[1:] - offsets[:-1]
    return torch.repeat_interleave(
        torch.arange(len(counts), device=device, dtype=torch.int32), counts)


def pad_to_multiple(x: torch.Tensor, multiple: int, dim: int):
    difference = (multiple - (x.shape[dim] % multiple)) % multiple
    if difference == 0:
        return x

    dim = dim if dim >= 0 else x.ndim + dim
    pad_list = []

    for i in range(x.ndim - 1, dim - 1, -1):
        if i == dim:
            pad_list.extend([0, difference])
        else:
            pad_list.extend([0, 0])

    return F.pad(x, pad_list, mode="constant", value=0)


class FlexAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @classmethod
    def get_supported_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16, torch.float32]

    @classmethod
    def validate_head_size(cls, head_size: int) -> None:
        return  # FlexAttention supports any head size

    @staticmethod
    def get_name() -> str:
        return "FLEX_ATTENTION"

    @staticmethod
    def get_impl_cls() -> type["FlexAttentionImpl"]:
        return FlexAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return FlexAttentionMetadata

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_builder_cls() -> type["FlexAttentionMetadataBuilder"]:
        return FlexAttentionMetadataBuilder

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False


#@torch.compile(fullgraph=True, mode="reduce-overhead")
def physical_to_logical_mapping(block_table: torch.Tensor,
                                seq_lens: torch.Tensor, block_size: int,
                                total_blocks: int) -> torch.Tensor:
    """
    Creates an inverse mapping from physical block locations to logical indices.

    The original block_table maps from logical blocks to physical locations:

    Logical to Physical (Original block_table):
    ┌───────────────────────────────────────────┐
    │ Request 0:                                │
    │                                           │
    │ Logical Blocks:  0  1  2  3  4  5  6  7   │
    │                  │  │  │  │  │  │  │  │   │
    │                  v  v  v  v  v  v  v  v   │
    │ Physical Blocks: 3  5  1  7  4  2  0  6   │
    └───────────────────────────────────────────┘

    This function creates the inverse mapping:

    Physical to Logical (Inverse mapping):
    ┌───────────────────────────────────────────┐
    │ Request 0:                                │
    │                                           │
    │ Physical Blocks: 0  1  2  3  4  5  6  7   │
    │                  │  │  │  │  │  │  │  │   │
    │                  v  v  v  v  v  v  v  v   │
    │ Logical Blocks:  6  2  5  0  4  1  7  3   │
    └───────────────────────────────────────────┘

    If multiple logical blocks map to the same physical block,
    this function returns the first (minimum) logical block index.

    If a physical block is not mapped to by any logical block,
    its value in the result will be -1.

    IMPORTANT: Garbage Value Protection
    ────────────────────────────────────
    The block_table tensor may contain garbage values in unused positions
    (beyond the actual sequence length). For example, if a sequence only
    needs 3 blocks but the table has space for 8:

        block_table[0] = [10, 25, 7, 999, 1234, 888, ...]
                                    ^^^^^^^^^^^^^^^^^^^^
                                    garbage values

    These garbage values can cause issues because:
    1. They may map to valid physical blocks by coincidence
    2. The scatter_ operation will assign them logical indices
    3. Later attention computations may incorrectly access these blocks

    To prevent this, we use seq_lens and block_size to mask out unused
    entries, ensuring only valid block references are processed.

    Args:
        block_table: Tensor of shape [max_reqs, max_num_blocks]
            mapping logical blocks to physical locations. May contain
            garbage values in unused positions.
        seq_lens: Tensor of sequence lengths for each request. Used to
            determine how many blocks are actually needed per sequence.
        block_size: Size of each block in tokens. Used with seq_lens to
            compute the number of valid blocks per sequence.
        total_blocks: Total number of physical blocks available

    Returns:
        A tensor of shape [max_reqs, total_blocks] where each entry
        physical_to_logical[req_id, physical_block] contains the logical
        block index for that physical block, or -1 if unused.
    """
    max_reqs, max_num_blocks = block_table.shape
    device = block_table.device

    physical_to_logical = torch.full((max_reqs, total_blocks),
                                     -1,
                                     dtype=torch.long,
                                     device=device)

    # Only process valid blocks to avoid garbage values
    num_blocks_per_seq = cdiv(seq_lens, block_size)
    mask = torch.arange(max_num_blocks,
                        device=device)[None, :] < num_blocks_per_seq[:, None]

    valid_block_table = torch.where(mask, block_table, 0)
    valid_logical_indices = torch.where(
        mask,
        torch.arange(max_num_blocks, device=device)[None, :], 0)

    physical_to_logical.scatter_(-1, valid_block_table.to(torch.int64),
                                 valid_logical_indices)
    # NB - Seems like block 0 is always empty so we reset it manually
    physical_to_logical[:, 0] = -1
    return physical_to_logical


def unique_static_unsorted(
        x: torch.Tensor,
        *,
        M: int,  # maximum positive value (0 is “skip me”)
        dim: int = -1,  # axis along which to deduplicate
        ignored_val: int = 0,  # value to ignore
        pad_val: int = -1,  # sentinel for unused slots
) -> torch.Tensor:
    """
    - Keeps the first occurrence of each non-zero value while preserving order,
      then left-packs those uniques and fills the rest with `pad_val`.
    - Returns (packed, keep_mask) with the *same shape* as `x`.
    - Requires that all values be in the range [0, M]
    - Skips ignored_val

    Works on CPU or GPU, no Python loops, O(B·N) time / O(B·M) memory.

    Example:
    x =[3, 1, 0, 1, 2], M=3, ignored_val=0 => [3, 1, 2, -1, -1]
    """
    if not (-1 <= pad_val <= M):
        raise ValueError("`pad_val` must lie in [-1, M]")

    # ── move `dim` to the end so we can treat tensor as [B, N] ──────────
    dim = dim % x.ndim
    x_perm = x.movedim(dim, -1)  # shape [..., N]
    B, N = x_perm.numel() // x_perm.shape[-1], x_perm.shape[-1]
    x_flat = x_perm.reshape(B, N)  # [B, N]

    device = x.device
    idx = torch.arange(N, device=device).expand(B, N)  # per-row indices

    # ── build first-occurrence table for every v ∈ [0, M] ───────────────
    first_idx = torch.full((B, M + 1), N, device=device)  # “∞”
    # scatter_reduce_: first_idx[b, v] = min(first_idx[b, v], i) for each i
    first_idx.scatter_reduce_(1, x_flat, idx, reduce="amin")

    # ── keep mask: first occurrence *and* value ≠ 0 ─────────────────────
    keep = (x_flat != ignored_val) & (idx == first_idx.gather(1, x_flat)
                                      )  # [B, N]

    # ── left-pack uniques into a fresh tensor ───────────────────────────
    dest_pos = torch.cumsum(keep.to(torch.long), dim=1) - 1  # where to go
    packed_flat = torch.full_like(x_flat, pad_val)

    rows, src_cols = torch.nonzero(keep, as_tuple=True)
    packed_flat[rows, dest_pos[rows, src_cols]] = x_flat[rows, src_cols]

    # ── restore original layout ─────────────────────────────────────────
    packed = packed_flat.reshape(x_perm.shape).movedim(-1, dim)
    return packed


def causal_mask_mod(b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor,
                    kv_idx: torch.Tensor):
    return q_idx >= kv_idx


@dataclass
class FlexAttentionMetadata:
    causal: bool
    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: Optional[torch.Tensor]
    prefix_kv_lens: Optional[torch.Tensor]
    suffix_kv_lens: Optional[torch.Tensor]

    # Block info
    total_cache_tokens: int
    block_size: int
    max_possible_sequence_length: int
    num_reqs: int
    physical_to_logical: torch.Tensor
    decode_offset: torch.Tensor
    num_blocks_per_seq: torch.Tensor

    # For logging.
    num_input_tokens: int = 0  # Number of tokens including padding.

    # Flex Metadata
    num_blocks = 0
    block_mask: Optional[BlockMask] = None
    score_mod: Optional[_score_mod_signature] = None
    logical_mask_mod: _mask_mod_signature = causal_mask_mod
    doc_ids: Optional[torch.Tensor] = None
    direct_build: bool = True
    q_block_size: int = 16
    kv_block_size: int = 16
    transformed_score_mod: Optional[_score_mod_signature] = None
    sliding_window: Optional[int] = None
    # When True, _build_block_mask_direct fetches all blocks up to
    # max_possible_sequence_length (max_model_len) instead of
    # max_seq_len. This forces kv_indices to a static shape so the
    # caching allocator returns stable pointers across build() calls,
    # which is required for FULL cudagraph replay validity.
    use_full_cuda_graph: bool = False
    # Persistent buffers (owned by builder, refs passed through here).
    # When use_full_cuda_graph is True, _build_block_mask_direct
    # copies the per-step kv_indices / kv_num_blocks into these so the
    # captured graph's stable data_ptr stays valid across replays. The
    # alternative (fresh tensors each step) gives the allocator the
    # freedom to return different addresses, and the captured kernels
    # then read use-after-free memory at replay.
    persistent_kv_indices: Optional[torch.Tensor] = None
    persistent_kv_num_blocks: Optional[torch.Tensor] = None
    persistent_doc_ids: Optional[torch.Tensor] = None

    # TiDAR single-forward (sparse-proposal) attention mask. When all three
    # fields are set, ``get_mask_mod`` returns a structured mask that
    # implements paper Figure 3 right with sparse proposals -- verify
    # segment causal among itself + bidirectional within each proposal
    # block + each proposal attends only to verify[:p_j+1]. See
    # docs/tidar_single_forward_design_2026-05-13.md §3 +
    # scripts/_tidar_flex_smoke.py for the standalone validation.
    tidar_single_forward_verify_len: Optional[int] = None
    tidar_single_forward_K_drafts: Optional[int] = None
    tidar_single_forward_proposal_acc_levels: Optional[torch.Tensor] = None
    # CPU mirror of acc_levels for use inside the multi-call FA path
    # WITHOUT calling .cpu() during cudagraph capture (which crashes
    # with "operation not permitted when stream is capturing").
    tidar_single_forward_proposal_acc_levels_cpu: Optional[list[int]] = None
    # CPU mirror of per-request prefix lengths (= decode_offset). Cached
    # by the metadata builder per build() call so the multi-call FA
    # closure can iterate per request without .cpu() inside capture.
    tidar_single_forward_prefix_lens_cpu: Optional[list[int]] = None
    # [max_num_seqs, P] int tensor: scratch_block_ids[req, p] is the
    # physical block ID holding proposal p's K masks (K/V written via
    # slot_mapping). Used by ``get_tidar_single_forward_mask_mod`` to
    # detect scratch KV slots and remap their logical_kv_idx to the
    # proposal-segment position (overriding physical_to_logical's
    # natural mapping, which is misaligned because verify_len=K+1
    # doesn't divide block_size=K).
    tidar_single_forward_scratch_block_ids: Optional[torch.Tensor] = None

    def _convert_physical_to_logical(
        self,
        request_lookup: torch.Tensor,
        q_idx: torch.Tensor,
        physical_kv_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert physical indices to logical indices for both query and kv.

        NB is_within_lower_bound: do sequences start on block_boundaries?

        Returns:
            tuple of (is_valid, logical_q_idx, logical_kv_idx)
        """
        # Map query indices to corresponding request indices
        q_req = request_lookup[q_idx]

        # Convert physical KV indices to logical indices
        physical_kv_block = physical_kv_idx // self.block_size
        physical_kv_offset = physical_kv_idx % self.block_size
        logical_block_idx = self.physical_to_logical[q_req, physical_kv_block]
        logical_kv_idx = (logical_block_idx * self.block_size +
                          physical_kv_offset)

        # Determine valid kv indices
        live_block = logical_block_idx >= 0
        within_upper_bound = logical_kv_idx < self.seq_lens[q_req]
        within_lower_bound = logical_kv_idx >= 0
        is_valid = live_block & within_upper_bound & within_lower_bound

        # Convert physical query indices to logical indices
        local_q_idx = q_idx - self.query_start_loc[q_req]
        logical_q_idx = local_q_idx + self.decode_offset[q_req]

        return is_valid, logical_q_idx, logical_kv_idx

    def get_causal_mask_mod(self) -> _mask_mod_signature:
        """Creates the mask_mod function for FlexAttention.

        This function creates the combined mask mod function that handles:
            1. The paged attention block mapping
            2. The mapping from packed query sequences to logical query entries

        It also by defaults adds the decoding offset to the query indices.
        With this info we create the "logical" indices that are passed to
        mask_mod functions. This allows mask mod functions to be agnostic to
        layout of the query and key/value tensors.
        """
        assert self.doc_ids is not None

        def final_mask_mod(
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            physical_kv_idx: torch.Tensor,
        ) -> torch.Tensor:
            (is_valid, logical_q_idx,
             logical_kv_idx) = self._convert_physical_to_logical(
                 self.doc_ids, q_idx, physical_kv_idx)
            # Apply mask modification only for valid indices
            return torch.where(
                is_valid,
                self.logical_mask_mod(b, h, logical_q_idx, logical_kv_idx),
                False,
            )

        return final_mask_mod

    def get_bidirectional_mask_mod(self) -> _mask_mod_signature:
        """Creates the bidirectional mask_mod function.

        For DECODER attention with paged KV cache (e.g., TiDAR's two-
        forward drafter, where the K+1 input tokens attend
        bidirectionally to each other AND causally to the prefix in
        the paged cache), we apply ``_convert_physical_to_logical``
        so the prefix is visible. The is_valid check (live block,
        within seq_lens, same request via physical_to_logical[q_req])
        already enforces "same request" and "within sequence" --
        non-causal across that range is the desired behavior.

        For ENCODER_ONLY models (no paged cache), the metadata
        builder runs the original inline-K/V path; this mask_mod is
        still safe because is_valid degrades to the request_lookup
        match when physical_to_logical is identity.
        """
        assert self.doc_ids is not None

        def final_mask_mod(
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            physical_kv_idx: torch.Tensor,
        ) -> torch.Tensor:
            (is_valid, logical_q_idx,
             logical_kv_idx) = self._convert_physical_to_logical(
                 self.doc_ids, q_idx, physical_kv_idx)
            return torch.where(is_valid, True, False)

        return final_mask_mod

    def get_sliding_window_mask_mod(self) -> _mask_mod_signature:
        """Creates the sliding window mask_mod function for FlexAttention.

        Note that the sliding window mask here is bidirectional, we need
        to mask it with the bidirectional/causal mask for encoder/decoder.
        """

        if self.sliding_window is None:
            raise ValueError(
                "sliding_window must be set for sliding window attention")

        def sliding_window_mask_mod(b: torch.Tensor, h: torch.Tensor,
                                    q_idx: torch.Tensor, kv_idx: torch.Tensor):
            return torch.abs(q_idx - kv_idx) < self.sliding_window

        def final_mask_mod(
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            physical_kv_idx: torch.Tensor,
        ) -> torch.Tensor:
            (is_valid, logical_q_idx,
             logical_kv_idx) = self._convert_physical_to_logical(
                 self.doc_ids, q_idx, physical_kv_idx)
            return torch.where(
                is_valid,
                sliding_window_mask_mod(b, h, logical_q_idx, logical_kv_idx),
                False,
            )

        return final_mask_mod if self.causal else sliding_window_mask_mod

    def get_tidar_single_forward_mask_mod(self) -> _mask_mod_signature:
        """Mask mod for TiDAR single-forward (sparse-proposal design).

        Implements the structured mask from paper Figure 3 right,
        generalized to a sparse set of proposals at chosen acc levels:

          * Per-req new-token layout (in FA logical-position space):
              [verify (verify_len), prop_1 (K), ..., prop_P (K)]
          * Verify tokens: causal among themselves; nothing else
          * Proposal tokens: bidirectional within own block;
              causally attend to verify[0 .. acc_levels[p_index]]
              (= anchor + acc_levels[p_index] drafts under the standard
              K+1-input verify convention); never to other proposals
          * Prefix (cached) is always visible

        Wraps ``vllm.v1.spec_decode.tidar_single_forward.tidar_mask_mod``
        to gather per-request ``prefix_len`` (= ``decode_offset[q_req]``)
        via the ``doc_ids`` lookup from query position to request index.
        Plumbing matches the existing causal / bidirectional mask_mod
        wrappers in this class.
        """
        assert self.doc_ids is not None
        assert self.tidar_single_forward_verify_len is not None
        assert self.tidar_single_forward_K_drafts is not None
        assert self.tidar_single_forward_proposal_acc_levels is not None

        K = int(self.tidar_single_forward_K_drafts)
        verify_len = int(self.tidar_single_forward_verify_len)
        acc_levels = self.tidar_single_forward_proposal_acc_levels

        # Local handles for the inner closure -- flex_attention's
        # torch.compile path picks these up by reference.
        doc_ids = self.doc_ids
        decode_offset = self.decode_offset
        block_size = self.block_size
        scratch_block_ids = self.tidar_single_forward_scratch_block_ids
        P_props = (acc_levels.shape[0] if acc_levels.dim() == 1
                   else acc_levels.shape[-1])

        # If scratch_block_ids isn't set (e.g., warmup dummy_run before
        # _ensure_scratch_blocks is callable), fall through with no
        # scratch handling -- proposal-to-own-block attention will be
        # blocked (degraded but valid) and the captured graph won't
        # trip Dynamo on a None indexing.
        if scratch_block_ids is None:
            # Cache a sentinel zero tensor so the closure still type-checks.
            scratch_block_ids = torch.zeros(
                (1, P_props, 2), dtype=torch.int64,
                device=self.block_table.device) - 1

        def final_mask_mod(
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            physical_kv_idx: torch.Tensor,
        ) -> torch.Tensor:
            (is_valid, logical_q_idx,
             logical_kv_idx) = self._convert_physical_to_logical(
                 doc_ids, q_idx, physical_kv_idx)
            q_req = doc_ids[q_idx]
            prefix_len = decode_offset[q_req]

            # Scratch detection: proposal K/V are written to scratch
            # blocks (one per proposal per request) via slot_mapping.
            # FlexAttention's physical_to_logical_mapping covers them
            # because we appended scratch IDs to block_table at SF
            # inflate, but the resulting logical_kv_idx is misaligned
            # (verify_len=K+1 doesn't divide block_size=K). Detect
            # scratch and synthesize the correct logical_kv_idx so
            # the SF mask_mod's kv_p_index = p_idx-of-this-scratch.
            #
            # scratch_block_ids[q_req, p] is the scratch block ID for
            # proposal p of request q_req. If physical_kv_idx's block
            # matches any of these P entries, treat as scratch.
            phys_block = physical_kv_idx // block_size
            offset_in_block = physical_kv_idx % block_size
            # Layout mode by verify_len:
            #   K+1 default: scratch_block_ids is [B, P, 2], K+1 slots
            #     span 2 sub-blocks (blk 0: slots 0..K-1, blk 1: slot K).
            #     proposal_seg_len = K+1 in the LOGICAL kv space.
            #   K-mask no_bonus: scratch_block_ids is [B, P, 1] (or
            #     legacy [B, P, 2] with sub-block 1 unused); K slots
            #     fit in sub-block 0. proposal_seg_len = K in the
            #     LOGICAL kv space.
            # ZAP-ONLY: K+1 scratch layout always.
            no_bonus = False
            if no_bonus:
                scratch_row = (scratch_block_ids[q_req, :, 0]
                               if scratch_block_ids.dim() == 3
                               else scratch_block_ids[q_req])  # [P]
                is_scratch = torch.zeros_like(physical_kv_idx,
                                              dtype=torch.bool)
                kv_p_idx_scratch = torch.zeros_like(physical_kv_idx,
                                                    dtype=torch.int64)
                for p_idx in range(P_props):
                    match_pb = (phys_block == scratch_row[p_idx].to(
                        phys_block.dtype))
                    kv_p_idx_scratch = torch.where(
                        match_pb,
                        torch.full_like(kv_p_idx_scratch, p_idx),
                        kv_p_idx_scratch)
                    is_scratch = is_scratch | match_pb
                proposal_seg_len = K
                offset_within_seg = offset_in_block
            else:
                scratch_row = scratch_block_ids[q_req]  # [P, 2]
                is_scratch = torch.zeros_like(physical_kv_idx,
                                              dtype=torch.bool)
                kv_p_idx_scratch = torch.zeros_like(physical_kv_idx,
                                                    dtype=torch.int64)
                block_within = torch.zeros_like(physical_kv_idx,
                                                dtype=torch.int64)
                for p_idx in range(P_props):
                    for blk_idx in range(2):
                        match_pb = (phys_block == scratch_row[
                            p_idx, blk_idx].to(phys_block.dtype))
                        kv_p_idx_scratch = torch.where(
                            match_pb,
                            torch.full_like(kv_p_idx_scratch, p_idx),
                            kv_p_idx_scratch)
                        block_within = torch.where(
                            match_pb,
                            torch.full_like(block_within, blk_idx),
                            block_within)
                        is_scratch = is_scratch | match_pb
                proposal_seg_len = K + 1
                offset_within_seg = (block_within * block_size
                                    + offset_in_block)

            # Synthetic logical_kv_idx for scratch: kv_local = verify_len
            # + kv_p_idx_scratch * proposal_seg_len + offset_within_seg.
            logical_kv_idx_scratch = (prefix_len + verify_len
                                      + kv_p_idx_scratch * proposal_seg_len
                                      + offset_within_seg)
            logical_kv_idx = torch.where(is_scratch,
                                         logical_kv_idx_scratch,
                                         logical_kv_idx)
            is_valid = is_valid | is_scratch

            tidar_keep = tidar_mask_mod(
                logical_q_idx, logical_kv_idx,
                prefix_len, K, acc_levels,
                verify_len=verify_len,
                no_bonus_layout=no_bonus)
            return torch.where(is_valid, tidar_keep, False)

        return final_mask_mod

    def get_mask_mod(self):
        # TiDAR single-forward overrides all other mask paths -- the
        # structured mask subsumes causal / bidirectional within its own
        # combination rules.
        if self.tidar_single_forward_proposal_acc_levels is not None:
            return self.get_tidar_single_forward_mask_mod()
        # Stage-1: initialize the base mask_mod
        # (causal mask for decoder or bidirectional mask for encoder)
        if self.causal:
            mask_mod = self.get_causal_mask_mod()
        else:
            mask_mod = self.get_bidirectional_mask_mod()
        # stage-2: add external mask_mod for special attention during
        # forwarding runtime to create the combined mask_mod.
        if self.sliding_window is not None:
            # Add sliding window mask for sliding window attention
            sliding_window_mask_mod = self.get_sliding_window_mask_mod()
            mask_mod = and_masks(mask_mod, sliding_window_mask_mod)
        return mask_mod

    def get_transformed_score_mod(self) -> Optional[_score_mod_signature]:
        """Creates the transformed score_mod function for FlexAttention.

        This function wraps the user's score_mod to handle physical-to-logical
        index conversion, similar to how get_mask_mod works for mask functions.
        """
        if self.score_mod is None:
            return None

        # Create a lookup mapping from query indices -> request number
        request_lookup = _offsets_to_doc_ids_tensor(self.query_start_loc)
        user_score_mod = self.score_mod

        def transformed_score_mod(
            score: torch.Tensor,
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            physical_kv_idx: torch.Tensor,
        ) -> torch.Tensor:
            (is_valid, logical_q_idx,
             logical_kv_idx) = self._convert_physical_to_logical(
                 request_lookup, q_idx, physical_kv_idx)

            return torch.where(
                is_valid,
                user_score_mod(score,
                               b,
                               h,
                               logical_q_idx,
                               logical_kv_idx,
                               physical_q=q_idx), -float('inf'))

        return transformed_score_mod

    def _build_block_mask_direct(self) -> BlockMask:
        """Direct block mask construction for standard causal attention.

        This method constructs the block mask directly using
        BlockMask.from_kv_blocks which is much more efficient than the
        generic create_block_mask approach.

        The direct path works as follows:
        1. For each query token, fetch blocks from block_table using max_seq_len
           (this fetches more blocks than needed for shorter sequences)
        2. Group query tokens into chunks of q_block_size
        3. For each group, deduplicate the blocks using unique_static_unsorted
        4. Create BlockMask using the deduplicated block indices

        Over-estimation occurs when a group of q_block_size tokens contains
        multiple sequence IDs (doc_ids). In this case, we fetch ALL blocks for
        each sequence represented in the group, even though individual query
        tokens may only need a subset of those blocks based on causal masking
        and their position.

        """
        page_to_block_ratio = self.kv_block_size // self.block_size
        if page_to_block_ratio != 1:
            raise ValueError(
                f"FlexAttention currently requires the cache block size "
                f"({self.block_size}) to be equal to the kv_block_size "
                f"({self.kv_block_size}). Please check your model's "
                f"configuration.")

        # When FULL cudagraph is enabled, fix the block-fetch range at
        # max_model_len so kv_indices has a static shape (stable
        # caching-allocator pointer) across build() calls. is_valid in
        # _convert_physical_to_logical masks out unused blocks for
        # correctness; the extra unused work is the price for capture.
        fetch_seq_len = (self.max_possible_sequence_length
                         if self.use_full_cuda_graph else self.max_seq_len)
        used_pages = self.block_table[
            self.doc_ids, :cdiv(fetch_seq_len, self.block_size)]
        used_pages_padded = pad_to_multiple(used_pages,
                                            multiple=self.q_block_size,
                                            dim=0)
        used_pages_padded = used_pages_padded.reshape(
            used_pages_padded.shape[0] // self.q_block_size, -1)
        used_pages_padded = used_pages_padded // page_to_block_ratio
        kv_indices_fresh = unique_static_unsorted(
            (used_pages_padded.long()), M=self.num_blocks).to(torch.int32)
        kv_num_blocks_fresh = (kv_indices_fresh >= 0).sum(
            dim=-1).to(torch.int32)

        # FULL cudagraph: copy the freshly-computed values into the
        # builder's persistent buffers so the captured kernel's
        # data_ptr stays valid across replays. The view slice into
        # persistent storage shares the storage's data_ptr (starting
        # at offset 0), and slicing back to ``num_q_blocks`` at each
        # build() returns the same data_ptr/shape/stride -- which is
        # exactly what cudagraph replay requires.
        # Same conditional as doc_ids: large prefill batches that
        # exceed the persistent buffer capacity use PIECEWISE mode
        # and don't need pointer stability.
        if (self.use_full_cuda_graph
                and self.persistent_kv_indices is not None
                and kv_indices_fresh.shape[0]
                    <= self.persistent_kv_indices.shape[0]):
            num_q_blocks = kv_indices_fresh.shape[0]
            # Zero the unused rows first (the persistent buffer may
            # still carry stale data from a previous larger step).
            self.persistent_kv_indices[num_q_blocks:].fill_(-1)
            self.persistent_kv_num_blocks[num_q_blocks:].fill_(0)
            self.persistent_kv_indices[:num_q_blocks].copy_(kv_indices_fresh)
            self.persistent_kv_num_blocks[:num_q_blocks].copy_(
                kv_num_blocks_fresh)
            kv_indices = self.persistent_kv_indices[:num_q_blocks]
            kv_num_blocks = self.persistent_kv_num_blocks[:num_q_blocks]
        else:
            kv_indices = kv_indices_fresh
            kv_num_blocks = kv_num_blocks_fresh

        block_mask_kwargs = {
            "seq_lengths": (self.num_actual_tokens, self.total_cache_tokens),
            "kv_num_blocks": kv_num_blocks[None, None],
            "kv_indices": kv_indices[None, None],
            "full_kv_num_blocks": None,
            "full_kv_indices": None,
            "BLOCK_SIZE": (self.q_block_size, self.kv_block_size),
            "mask_mod": self.mask_mod,
        }

        # compute_q_blocks parameter is available in PyTorch 2.9+
        if is_torch_equal_or_newer("2.9.0.dev0"):
            block_mask_kwargs["compute_q_blocks"] = False
        return BlockMask.from_kv_blocks(**block_mask_kwargs)

    def build_block_mask(self) -> BlockMask:
        mask_mod = self.get_mask_mod()
        # DECODER paths (causal verifier, non-causal drafter, structured
        # TiDAR SF) all read from the paged kv_cache -- kv_len spans the
        # full cache. Only the ENCODER_ONLY non-causal path uses inline
        # K/V (kv_len = num_actual_tokens); since this builder serves
        # decoder-only models in our SMoE+CCA setup, the cache path
        # applies whenever the metadata has a non-zero cache.
        use_paged_cache = self.causal or self.total_cache_tokens > 0
        kv_len = (self.total_cache_tokens
                  if use_paged_cache else self.num_actual_tokens)
        return create_block_mask_compiled(
            mask_mod,
            None,
            None,
            self.num_actual_tokens,
            kv_len,
            device=self.block_table.device,
            BLOCK_SIZE=(self.q_block_size, self.kv_block_size),
        )

    def __post_init__(self):
        assert self.use_cascade is False, "Not implemented yet."
        assert self.common_prefix_len == 0, "Not implemented yet."
        assert self.cu_prefix_query_lens is None, "Not implemented yet."
        assert self.prefix_kv_lens is None, "Not implemented yet."
        assert self.suffix_kv_lens is None, "Not implemented yet."
        # Create a lookup mapping from query indices -> request number.
        # For FULL cudagraph the mask_mod closure reads this -- write
        # into the persistent buffer so its data_ptr stays stable
        # across build() calls (otherwise the captured graph reads
        # use-after-free memory from the prior step's fresh allocation).
        doc_ids_fresh = _offsets_to_doc_ids_tensor(self.query_start_loc)
        # Persistent buffer is sized to max_capture_size. Batches
        # larger than that (e.g., chunked-prefill batches) run in
        # PIECEWISE mode, not FULL cudagraph, so they don't need
        # pointer stability -- fall back to the fresh tensor.
        # VLLM_TIDAR_DISABLE_MASKMOD_PERSISTENT=1 forces the fallback
        # for A/B diagnostics.
        import os as _os
        _maskmod_persist = (
            _os.environ.get("VLLM_TIDAR_DISABLE_MASKMOD_PERSISTENT",
                            "0") != "1")
        if (_maskmod_persist
                and self.use_full_cuda_graph
                and self.persistent_doc_ids is not None
                and doc_ids_fresh.shape[0]
                    <= self.persistent_doc_ids.shape[0]):
            n = doc_ids_fresh.shape[0]
            self.persistent_doc_ids[:n].copy_(doc_ids_fresh)
            self.doc_ids = self.persistent_doc_ids[:n]
        else:
            self.doc_ids = doc_ids_fresh
        self.num_blocks = self.total_cache_tokens // self.block_size

        self.mask_mod = self.get_mask_mod()
        self.transformed_score_mod = self.get_transformed_score_mod()

        # Route the non-causal TF drafter through the direct path
        # under FULL cudagraph so kv_indices / kv_num_blocks come from
        # the builder's PERSISTENT buffers (stable data_ptr across
        # build() calls). The create_block_mask_compiled path
        # (build_block_mask) allocates fresh tensors per call, so the
        # captured graph baked the warmup-time addresses and at replay
        # reads from freed memory -> degenerate drafts (0% accept).
        # The direct path's over-estimation is fine here: mask_mod
        # handles per-position validity.
        # Exclude SF (structured mask_mod) and ENCODER_ONLY (inline
        # K/V) -- both need the create_block_mask path; SF's also hits
        # a default-stream crash if forced through direct under capture.
        _route_direct_non_causal = (
            self.use_full_cuda_graph
            and self.total_cache_tokens > 0  # paged decoder
            and self.tidar_single_forward_proposal_acc_levels is None
        )
        if self.direct_build and (self.causal or _route_direct_non_causal):
            self.block_mask = self._build_block_mask_direct()
        else:
            self.block_mask = self.build_block_mask()


class FlexAttentionMetadataBuilder(
        AttentionMetadataBuilder[FlexAttentionMetadata]):

    # UNIFORM_BATCH: supports FULL_DECODE_ONLY cudagraph for batches with
    # uniform query_len (e.g., spec-decode uniform = K+1 + P*K for TiDAR
    # single-forward). Achieved by static block_mask shapes (uses
    # max_model_len's worth of blocks, not max_seq_len's) so the caching
    # allocator returns stable pointers across build() calls.
    _cudagraph_support: ClassVar[AttentionCGSupport] = \
        AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, kv_cache_spec: AttentionSpec, layer_names: list[str],
                 vllm_config: VllmConfig, device: torch.device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.cache_config = vllm_config.cache_config
        self.compilation_config = vllm_config.compilation_config

        self.num_heads_q = self.model_config.get_num_attention_heads(
            self.parallel_config)
        self.num_heads_kv = self.model_config.get_num_kv_heads(
            self.parallel_config)
        self.headdim = self.model_config.get_head_size()
        self.block_size = kv_cache_spec.block_size
        self.kv_cache_spec = kv_cache_spec
        self.direct_build: bool = is_torch_equal_or_newer("2.9.0.dev0")
        self.q_block_size: int = 16 if is_torch_equal_or_newer(
            "2.9.0.dev0") else 128
        self.kv_block_size: int = 16 if is_torch_equal_or_newer(
            "2.9.0.dev0") else 128
        # When FULL cudagraph is enabled, freeze the block-fetch range
        # at max_model_len so kv_indices has a static shape across
        # capture and replay. Without this, _build_block_mask_direct's
        # ``cdiv(self.max_seq_len, self.block_size)`` varies per step,
        # the caching allocator returns different pointers, and the
        # captured graph reads stale memory.
        self.use_full_cuda_graph: bool = \
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        # Persistent buffers for FULL cudagraph. The captured graph
        # references several tensors by data_ptr -- they must stay
        # valid across build() calls. Without persistence we hit
        # illegal-memory-access / wrong-address-space errors inside
        # cudagraph.replay() because the per-call fresh allocations
        # get freed between capture and replay.
        #
        # Tensors read INSIDE the captured graph (via mask_mod
        # closure + block_mask):
        #   kv_indices, kv_num_blocks  -- block_mask sparse tensors
        #   doc_ids                    -- q_idx -> req lookup
        #   decode_offset              -- per-req prefix length
        #   physical_to_logical        -- per-req KV remap table
        # Tensors that are ALREADY persistent (sliced from InputBatch
        # owned buffers, so data_ptr starts at offset 0 and is
        # stable across build() calls):
        #   seq_lens, query_start_loc, block_table, slot_mapping
        # And the TiDAR acc_levels GPU tensor is allocated once by
        # the drafter (see _ensure_acc_levels_gpu).
        self._persistent_kv_indices: Optional[torch.Tensor] = None
        self._persistent_kv_num_blocks: Optional[torch.Tensor] = None
        self._persistent_doc_ids: Optional[torch.Tensor] = None
        self._persistent_decode_offset: Optional[torch.Tensor] = None
        self._persistent_physical_to_logical: Optional[torch.Tensor] = None

    def reorder_batch(self, input_batch: "InputBatch",
                      scheduler_output: "SchedulerOutput") -> bool:
        return False

    def build(self,
              common_prefix_len: int,
              common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False) -> FlexAttentionMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len

        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        num_blocks_per_seq = cdiv(seq_lens, self.block_size)

        use_cascade = common_prefix_len > 0
        cu_prefix_query_lens = None
        prefix_kv_lens = None
        suffix_kv_lens = None
        if use_cascade:
            raise NotImplementedError("Not yet my friend")

        block_size = self.kv_cache_spec.block_size
        max_possible_seq_len = self.model_config.max_model_len
        num_gpu_blocks = self.cache_config.num_gpu_blocks

        assert num_gpu_blocks is not None, \
            "FlexAttention requires num_gpu_blocks to be set"
        total_cache_tokens = (num_gpu_blocks * block_size)

        # Lazily allocate persistent buffers for FULL cudagraph.
        # Allocated on first build() because num_gpu_blocks is only
        # set after KV-cache init.
        if self.use_full_cuda_graph and self._persistent_kv_indices is None:
            max_cap = self.compilation_config.max_cudagraph_capture_size or 0
            max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs
            max_q_blocks = cdiv(max_cap, self.q_block_size)
            # unique_static_unsorted preserves input shape, so kv_indices
            # has the same dim-1 as ``used_pages_padded`` after reshape:
            # ``cdiv(max_model_len, block_size) * q_block_size``. Sizing
            # the buffer at num_gpu_blocks would mismatch this shape and
            # the copy_() asserts. (max_model_len matches the
            # ``fetch_seq_len`` static-shape bound below.)
            max_seq_blocks = cdiv(self.model_config.max_model_len, block_size)
            kv_indices_dim1 = max_seq_blocks * self.q_block_size
            if max_q_blocks > 0:
                self._persistent_kv_indices = torch.full(
                    (max_q_blocks, kv_indices_dim1),
                    -1, dtype=torch.int32, device=self.device)
                self._persistent_kv_num_blocks = torch.zeros(
                    (max_q_blocks,), dtype=torch.int32, device=self.device)
                # doc_ids: q_idx -> req lookup. Shape (max_cap,).
                self._persistent_doc_ids = torch.zeros(
                    (max_cap,), dtype=torch.int32, device=self.device)
                # decode_offset: per-req prefix length. Shape (max_num_seqs,).
                self._persistent_decode_offset = torch.zeros(
                    (max_num_seqs,), dtype=torch.int32, device=self.device)
                # physical_to_logical: per-req KV remap table.
                # Shape (max_num_seqs, num_gpu_blocks). Filled with -1
                # for unused entries (mask_mod relies on this for
                # is_valid filtering). Dtype MUST match
                # physical_to_logical_mapping's output (torch.long /
                # int64) -- mask_mod's arithmetic on logical_block_idx
                # is element-typed at compile time, and a captured
                # kernel traced with int32 storage but expecting int64
                # arithmetic reads two int32 elements per "load" and
                # produces garbage logical indices.
                self._persistent_physical_to_logical = torch.full(
                    (max_num_seqs, num_gpu_blocks),
                    -1, dtype=torch.long, device=self.device)

        inverse_block_table = physical_to_logical_mapping(
            block_table_tensor, seq_lens, block_size, num_gpu_blocks)

        offset_tensor = common_attn_metadata.num_computed_tokens_cpu.to(
            self.device, non_blocking=True)

        # FULL cudagraph: stabilize data_ptr for tensors the mask_mod
        # closure reads from. Each captured graph instance binds its
        # mask_mod's tensor pointers at capture time; replay reads
        # from those exact addresses, so write into our persistent
        # buffers (allocated once, lifetime = builder) and pass the
        # buffer views downstream.
        # Skip persistent path on prefill batches that exceed buffer
        # capacity -- those run in PIECEWISE mode and don't need
        # data_ptr stability.
        # VLLM_TIDAR_DISABLE_MASKMOD_PERSISTENT=1 disables this for
        # diagnostic A/B (kv_indices/kv_num_blocks remain persistent
        # but doc_ids/decode_offset/physical_to_logical use fresh
        # allocations like the pre-d554189 state).
        import os as _os
        _maskmod_persist = (
            _os.environ.get("VLLM_TIDAR_DISABLE_MASKMOD_PERSISTENT",
                            "0") != "1")
        if (_maskmod_persist
                and self.use_full_cuda_graph
                and self._persistent_physical_to_logical is not None
                and inverse_block_table.shape[0]
                    <= self._persistent_physical_to_logical.shape[0]
                and inverse_block_table.shape[1]
                    <= self._persistent_physical_to_logical.shape[1]
                and offset_tensor.shape[0]
                    <= self._persistent_decode_offset.shape[0]):
            self._persistent_physical_to_logical[
                :inverse_block_table.shape[0],
                :inverse_block_table.shape[1]
            ].copy_(inverse_block_table)
            inverse_block_table = self._persistent_physical_to_logical[
                :inverse_block_table.shape[0],
                :inverse_block_table.shape[1]]
            self._persistent_decode_offset[:offset_tensor.shape[0]].copy_(
                offset_tensor.to(torch.int32))
            offset_tensor = self._persistent_decode_offset[
                :offset_tensor.shape[0]]

        # TiDAR single-forward (sparse-proposal): runner sets these on
        # ``common_attn_metadata`` when an active step is single-forward.
        # When all three are present, ``get_mask_mod`` selects the
        # structured TiDAR mask (paper Fig 3 right). When None, behavior
        # is unchanged.
        sf_verify_len = getattr(
            common_attn_metadata, "tidar_single_forward_verify_len", None)
        sf_K_drafts = getattr(
            common_attn_metadata, "tidar_single_forward_K_drafts", None)
        sf_acc_levels = getattr(
            common_attn_metadata, "tidar_single_forward_proposal_acc_levels",
            None)
        sf_scratch_block_ids = getattr(
            common_attn_metadata,
            "tidar_single_forward_scratch_block_ids", None)
        # CPU mirrors precomputed here (before any potential cudagraph
        # capture) so the multi-call FA closure can run during capture
        # without triggering .cpu() syncs.
        if sf_acc_levels is not None:
            sf_acc_levels_cpu = sf_acc_levels.cpu().tolist()
        else:
            sf_acc_levels_cpu = None
        # SF prefix_lens = num_computed_tokens_cpu (the actual cached
        # prefix length per request). The earlier 'seq_lens_cpu -
        # inflate' formula is wrong for mixed batches (chunked prefill,
        # batches where SF inflation didn't fire) and silently produces
        # broken attention with ~0 acceptance for b>1. Memory:
        # project_tidar_sf_prefix_lens_bug — fix dated 2026-05-16.
        sf_prefix_lens_cpu: Optional[list[int]] = None
        if sf_verify_len is not None and sf_acc_levels is not None:
            sf_prefix_lens_cpu = (
                common_attn_metadata.num_computed_tokens_cpu[:num_reqs]
                .to(torch.int64).tolist())

        out = FlexAttentionMetadata(
            causal=common_attn_metadata.causal,
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            block_size=block_size,
            max_possible_sequence_length=max_possible_seq_len,
            num_reqs=num_reqs,
            physical_to_logical=inverse_block_table,
            total_cache_tokens=total_cache_tokens,
            decode_offset=offset_tensor,
            num_blocks_per_seq=num_blocks_per_seq,
            direct_build=self.direct_build,
            q_block_size=self.q_block_size,
            kv_block_size=self.kv_block_size,
            tidar_single_forward_verify_len=sf_verify_len,
            tidar_single_forward_K_drafts=sf_K_drafts,
            tidar_single_forward_proposal_acc_levels=sf_acc_levels,
            tidar_single_forward_proposal_acc_levels_cpu=sf_acc_levels_cpu,
            tidar_single_forward_prefix_lens_cpu=sf_prefix_lens_cpu,
            tidar_single_forward_scratch_block_ids=sf_scratch_block_ids,
            use_full_cuda_graph=self.use_full_cuda_graph,
            persistent_kv_indices=self._persistent_kv_indices,
            persistent_kv_num_blocks=self._persistent_kv_num_blocks,
            persistent_doc_ids=self._persistent_doc_ids,
        )
        return out

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        return False


class FlexAttentionImpl(AttentionImpl):
    sliding_window: Optional[int]
    alibi_slopes: Optional[torch.Tensor]
    logits_soft_cap: Optional[float]

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[list[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        logits_soft_cap: Optional[float] = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.attn_type = attn_type

        if attn_type not in (AttentionType.ENCODER_ONLY,
                             AttentionType.DECODER):
            raise NotImplementedError(
                f"FlexAttention does not support {attn_type} attention")

        if alibi_slopes is not None:
            raise NotImplementedError(
                "FlexAttention does not support alibi slopes yet.")
        else:
            self.alibi_slopes = None

        self.sliding_window = sliding_window

        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = logits_soft_cap
        if self.logits_soft_cap is not None:
            raise NotImplementedError(
                "FlexAttention does not support logits soft cap yet.")

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        if kv_sharing_target_layer_name is not None:
            raise NotImplementedError(
                "FlexAttention does not support kv sharing yet.")

        FlexAttentionBackend.validate_head_size(head_size)
        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "FlexAttention does not support quantized kv-cache. Yet")

    @staticmethod
    def view_as_4d(tensor: torch.Tensor) -> torch.Tensor:
        """View a 3d tensor as 4D."""
        if tensor.ndim == 4:
            return tensor
        assert tensor.ndim == 3
        return tensor[None, :, :, :]

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlexAttentionMetadata,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        output_block_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with FLexAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported"
                " for FlexAttentionImpl")

        enable_gqa = self.num_kv_heads != self.num_heads

        if attn_metadata is None:
            # Profiling run.
            return output
            # query = self.view_as_4d(query).permute(0, 2, 1, 3)
            # return torch.empty_like(query)

        # TiDAR single-forward FA3-verify + Triton-proposal split.
        # VLLM_TIDAR_SF_SPLIT=1 enables a two-launch path that runs the
        # verify segment on FA3 (paged-cache + causal — exactly its
        # sweet spot) and the proposal segment on the SF Triton kernel
        # with Q_START=verify_len. This avoids running the SF Triton
        # kernel over verify queries, which is its slowest portion.
        # Requires the SF Triton kernel to be wired (this method's
        # implementation reuses it).
        import os as _os
        _sf_use_split = (_os.environ.get("VLLM_TIDAR_SF_SPLIT", "0")
                         == "1")
        if (attn_metadata.tidar_single_forward_proposal_acc_levels
                is not None and _sf_use_split):
            return self._sf_split_forward(
                layer, query, key, value, output, kv_cache,
                attn_metadata)

        # TiDAR single-forward Triton paged kernel path.
        # DEFAULT: enabled. Opt out with VLLM_TIDAR_SF_TRITON=0 (which
        # falls through to the multi-call FA path below). Dispatches to
        # a fused custom Triton kernel (vllm/attention/ops/sf_attention.py)
        # that avoids FlexAttention's ~3x per-forward cost AND works
        # inside cudagraph capture (pure tensor ops, no Python loops).
        #
        # Why this is now the default (2026-05-30):
        # The multi-call FA path (_sf_multi_call_forward) was measured
        # to depress mean acceptance ~50% vs the Triton-paged path on
        # the same ckpt / K=16 / P=17 / thinking-off config (6.21 vs
        # 9.58 on AIME25; Triton matches handoff's stated 10.19, multi-
        # call does not). The performance gain is small (~1.8x at b=1)
        # but Triton is also the *correct* path — keep it as default
        # until the multi-call acceptance regression is understood.
        _sf_use_triton = (_os.environ.get("VLLM_TIDAR_SF_TRITON", "1")
                          != "0")
        if (attn_metadata.tidar_single_forward_proposal_acc_levels
                is not None and _sf_use_triton):
            return self._sf_triton_paged_forward(
                layer, query, key, value, output, kv_cache,
                attn_metadata)

        # TiDAR single-forward multi-call FA path. Splits the SF
        # combined forward into 2 flash_attn varlen calls per layer
        # (verify causal + all proposals batched bidirectional) to
        # bypass FlexAttention's ~3x per-forward cost. Supports batch>1
        # via per-request prefix gather + varlen cu_seqlens.
        # Eager-only: the per-request prefix slicing uses Python ints
        # derived from prefix_lens_cpu (which changes per step). Inside
        # cudagraph capture these ints get baked as constants -- the
        # replayed graph then uses stale prefix offsets for subsequent
        # steps and hangs. Captured runs fall through to FlexAttention
        # (still functional). Fixing captured would require dynamic
        # GPU slicing for variable prefix lengths.
        if (attn_metadata.tidar_single_forward_proposal_acc_levels
                is not None
                and not torch.cuda.is_current_stream_capturing()):
            return self._sf_multi_call_forward(
                layer, query, key, value, output, kv_cache, attn_metadata)

        num_actual_tokens = attn_metadata.num_actual_tokens

        if attn_metadata.sliding_window != self.sliding_window:
            attn_metadata.sliding_window = self.sliding_window
            if attn_metadata.direct_build:
                # TODO: Support skipping the computation of sliding window
                # in direct block mask building code path.
                logger.warning_once(
                    "Using direct block mask building with sliding window, "
                    "which is suboptimal now. Performance may be degraded.")
                # update mask mod in attention metadata
                attn_metadata.mask_mod = attn_metadata.get_mask_mod()
                attn_metadata.block_mask = (
                    attn_metadata._build_block_mask_direct())
            else:
                attn_metadata.block_mask = attn_metadata.build_block_mask()

        if (not attn_metadata.causal
                and self.attn_type == AttentionType.ENCODER_ONLY):
            # ENCODER_ONLY models: no paged prefix cache, K/V supplied
            # inline.
            query, key_tensor, value_tensor = map(
                lambda x: self.view_as_4d(x).permute(0, 2, 1, 3),
                (query, key, value),
            )

            query = query[:, :, :num_actual_tokens, :]
            if ((key_tensor.size(-2) > num_actual_tokens)
                    or (value_tensor.size(-2) > num_actual_tokens)):
                # In the encoder-only model with torch.compile,
                # qkv might be padded, which might cause exception.
                # see: https://github.com/vllm-project/vllm/pull/24872#discussion_r2353252290
                key_tensor = key_tensor[:, :, :num_actual_tokens, :]
                value_tensor = value_tensor[:, :, :num_actual_tokens, :]

        else:
            # DECODER: cache K/V (drafter writes go to scratch draft
            # block via slot_mapping; verifier writes go to AR block)
            # and attend through the paged cache. Without this, the
            # non-causal drafter sees ONLY the inline K+1 input and
            # not the prompt prefix in the cache, producing useless
            # drafts (TF acceptance 0.01 vs 4.62 with this fix).
            assert self.attn_type == AttentionType.DECODER
            key_cache, value_cache = kv_cache.unbind(0)

            torch.ops._C_cache_ops.reshape_and_cache_flash(
                key,
                value,
                key_cache,
                value_cache,
                attn_metadata.slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )

            # Reshape out the block_size dim. We use .reshape() not .view()
            # because the KV cache tensor can be non-contiguous when CCA's
            # mamba page padding has expanded the per-block stride
            # (vllm/config: "Padding mamba page size by N% to ensure mamba
            # page size and attention page size are exactly equal"). With
            # non-contiguous strides .view fails with "size is not compatible
            # with input tensor's size and stride"; .reshape transparently
            # copies if needed.
            key_cache = key_cache.reshape(
                -1, self.num_kv_heads, self.head_size)
            value_cache = value_cache.reshape(
                -1, self.num_kv_heads, self.head_size)
            query, key_tensor, value_tensor = map(
                lambda x: self.view_as_4d(x).permute(0, 2, 1, 3),
                (query, key_cache, value_cache),
            )

            query = query[:, :, :num_actual_tokens, :]

        # Doesn't work for now -> constraint violation
        # torch._dynamo.try_mark_dynamic(query, 2)

        assert attn_metadata.block_mask is not None
        block_m, block_n = attn_metadata.block_mask.BLOCK_SIZE

        kernel_options = get_kernel_options(query, block_m, block_n,
                                            attn_metadata.direct_build)
        out = flex_attention_compiled(
            query,
            key_tensor,
            value_tensor,
            attn_metadata.transformed_score_mod,
            attn_metadata.block_mask,
            self.scale,
            enable_gqa=enable_gqa,
            kernel_options=kernel_options,
        )

        # Flex doesn't have an out variant today, rely on epilogue fusion
        out = out.permute(0, 2, 1, 3).squeeze(0)
        output[:num_actual_tokens, :, :].copy_(out)
        return output

    def _sf_split_forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: Optional[torch.Tensor],
        kv_cache: torch.Tensor,
        attn_metadata: "FlexAttentionMetadata",
    ) -> torch.Tensor:
        """SF forward, split: FA3 for verify, Triton for proposals.

        The verify segment is K+1 causal queries against (paged prefix
        + just-cached verify K/V) — a textbook FA3 paged-cache call.
        FA3 outruns the Triton SF kernel for this shape because it
        skips the SF-specific mask scaffolding the Triton kernel must
        carry. The proposal segment keeps the Triton kernel (with
        Q_START=verify_len, so it skips the verify portion).

        Both launches share the same KV cache (verify K/V get written
        before either runs). Output is written in two strided regions
        of the same output tensor.
        """
        from vllm.attention.utils.fa_utils import flash_attn_varlen_func
        from vllm.attention.ops.sf_attention import (
            sf_attention_triton_paged)

        num_actual_tokens = attn_metadata.num_actual_tokens
        num_reqs = attn_metadata.num_reqs
        verify_len = int(attn_metadata.tidar_single_forward_verify_len)
        K_drafts = int(attn_metadata.tidar_single_forward_K_drafts)
        acc_levels = attn_metadata.tidar_single_forward_proposal_acc_levels
        P_props = int(acc_levels.shape[0])
        # Layout-mode by verify_len: K-mask + no-bonus when
        # verify_len == K_drafts; K+1 default otherwise.
        # ZAP-ONLY: K+1 proposal layout always.
        proposal_seg_len = K_drafts + 1
        total_per_req = verify_len + P_props * proposal_seg_len
        block_size = attn_metadata.block_size

        # Reshape Q/K/V to 3D.
        q3 = query[:num_actual_tokens].view(
            num_actual_tokens, self.num_heads, self.head_size)
        k3 = key[:num_actual_tokens].view(
            num_actual_tokens, self.num_kv_heads, self.head_size)
        v3 = value[:num_actual_tokens].view(
            num_actual_tokens, self.num_kv_heads, self.head_size)

        # Cache verify K/V to AR slots.
        key_cache, value_cache = kv_cache.unbind(0)
        slot_mapping_full = (
            attn_metadata.slot_mapping[:num_actual_tokens])
        slot_2d = slot_mapping_full.view(num_reqs, total_per_req)
        k_2d = k3.view(
            num_reqs, total_per_req, self.num_kv_heads, self.head_size)
        v_2d = v3.view(
            num_reqs, total_per_req, self.num_kv_heads, self.head_size)
        verify_slots = slot_2d[:, :verify_len].reshape(-1)
        verify_k = k_2d[:, :verify_len].reshape(
            num_reqs * verify_len, self.num_kv_heads, self.head_size)
        verify_v = v_2d[:, :verify_len].reshape(
            num_reqs * verify_len, self.num_kv_heads, self.head_size)
        torch.ops._C_cache_ops.reshape_and_cache_flash(
            verify_k, verify_v, key_cache, value_cache,
            verify_slots, self.kv_cache_dtype,
            layer._k_scale, layer._v_scale)

        # Output 3D / 4D views.
        out3 = output[:num_actual_tokens].view(
            num_actual_tokens, self.num_heads, self.head_size)
        out_4d = out3.view(
            num_reqs, total_per_req, self.num_heads, self.head_size)

        # --- FA3 verify call ---
        # Extract verify queries (strided per request) into a
        # contiguous buffer. q3 has [num_reqs*total_per_req, H, D];
        # verify queries are at [r*total_per_req + 0..verify_len-1].
        q_4d = q3.view(
            num_reqs, total_per_req, self.num_heads, self.head_size)
        verify_q = q_4d[:, :verify_len].contiguous().view(
            num_reqs * verify_len, self.num_heads, self.head_size)
        verify_out_buf = torch.empty_like(verify_q)
        # cu_seqlens_q for uniform per-req verify_len queries.
        cu_seqlens_q = torch.arange(
            0, (num_reqs + 1) * verify_len, verify_len,
            dtype=torch.int32, device=verify_q.device)
        # seqused_k per req = prefix_len + verify_len.
        decode_offset = attn_metadata.decode_offset[:num_reqs].to(
            torch.int32)
        seqused_k = decode_offset + int(verify_len)
        block_table = attn_metadata.block_table[:num_reqs]
        # Use the maximum-possible KV length from attn_metadata.
        max_seqlen_k = int(attn_metadata.max_seq_len)

        flash_attn_varlen_func(
            q=verify_q, k=key_cache, v=value_cache,
            out=verify_out_buf,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=int(verify_len),
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=True,
            block_table=block_table,
        )
        # Scatter verify output into the strided verify region of out.
        out_4d[:, :verify_len].copy_(
            verify_out_buf.view(
                num_reqs, verify_len, self.num_heads, self.head_size))

        # --- Triton proposal call (Q_START=verify_len) ---
        sf_attention_triton_paged(
            q=q3, inline_k=k3, inline_v=v3,
            kv_cache_k=key_cache, kv_cache_v=value_cache,
            block_table=block_table.to(torch.int32),
            prefix_lens=decode_offset,
            acc_levels=acc_levels.to(torch.int32),
            verify_len=verify_len,
            K_drafts=K_drafts,
            P_props=P_props,
            block_size=block_size,
            softmax_scale=self.scale,
            out=out3,
            q_start=verify_len,
        )
        return output

    def _sf_triton_paged_forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: Optional[torch.Tensor],
        kv_cache: torch.Tensor,
        attn_metadata: "FlexAttentionMetadata",
    ) -> torch.Tensor:
        """SF forward via the paged Triton kernel.

        Replaces FlexAttention's slow mask_mod path with a single fused
        Triton kernel call that internally handles:
          * inline K/V for verify + proposal segments
          * paged prefix reads via block_table + decode_offset
          * SF mask logic (verify causal, proposal bidir within block
            + causal-to-verify[<=acc_level])

        Works in both eager and cudagraph capture (no Python loops,
        no .item() calls; all per-request gather happens inside the
        Triton kernel).

        See vllm/attention/ops/sf_attention.py::sf_attention_triton_paged.
        """
        from vllm.attention.ops.sf_attention import (
            sf_attention_triton_paged)

        num_actual_tokens = attn_metadata.num_actual_tokens
        num_reqs = attn_metadata.num_reqs
        verify_len = int(attn_metadata.tidar_single_forward_verify_len)
        K_drafts = int(attn_metadata.tidar_single_forward_K_drafts)
        acc_levels = attn_metadata.tidar_single_forward_proposal_acc_levels
        P_props = int(acc_levels.shape[0])
        # Layout-mode by verify_len: K-mask + no-bonus (proposal_seg_len
        # = K) when verify_len == K_drafts; K+1 default otherwise.
        # ZAP-ONLY: K+1 proposal layout always.
        proposal_seg_len = K_drafts + 1
        total_per_req = verify_len + P_props * proposal_seg_len
        block_size = attn_metadata.block_size

        # Q/K/V are flattened [num_tokens, num_heads*head_dim] from the
        # vllm hook; reshape to 3D for the kernel.
        q3 = query[:num_actual_tokens].view(
            num_actual_tokens, self.num_heads, self.head_size)
        k3 = key[:num_actual_tokens].view(
            num_actual_tokens, self.num_kv_heads, self.head_size)
        v3 = value[:num_actual_tokens].view(
            num_actual_tokens, self.num_kv_heads, self.head_size)

        # Cache verify K/V to AR cache slots so the next step can read
        # them as prefix. Proposal K/V stay inline (the Triton kernel
        # reads them from k3/v3 directly).
        key_cache, value_cache = kv_cache.unbind(0)
        slot_mapping_full = (
            attn_metadata.slot_mapping[:num_actual_tokens])
        slot_2d = slot_mapping_full.view(num_reqs, total_per_req)
        k_2d = k3.view(
            num_reqs, total_per_req, self.num_kv_heads, self.head_size)
        v_2d = v3.view(
            num_reqs, total_per_req, self.num_kv_heads, self.head_size)
        verify_slots = slot_2d[:, :verify_len].reshape(-1)
        verify_k = k_2d[:, :verify_len].reshape(
            num_reqs * verify_len, self.num_kv_heads, self.head_size)
        verify_v = v_2d[:, :verify_len].reshape(
            num_reqs * verify_len, self.num_kv_heads, self.head_size)
        torch.ops._C_cache_ops.reshape_and_cache_flash(
            verify_k, verify_v, key_cache, value_cache,
            verify_slots, self.kv_cache_dtype,
            layer._k_scale, layer._v_scale)

        # Paged KV cache view: [num_blocks, block_size, H_kv, D].
        # vllm stores key_cache as [num_blocks, block_size, H_kv, D]
        # already; no reshape needed.
        # Block table for the active reqs.
        block_table = attn_metadata.block_table[:num_reqs].to(
            torch.int32)
        # Prefix length per req from decode_offset (= num_computed_tokens).
        prefix_lens = attn_metadata.decode_offset[:num_reqs].to(
            torch.int32)
        acc_levels_int = acc_levels.to(torch.int32)

        # Output buffer (caller-provided).
        out3 = output[:num_actual_tokens].view(
            num_actual_tokens, self.num_heads, self.head_size)

        sf_attention_triton_paged(
            q=q3, inline_k=k3, inline_v=v3,
            kv_cache_k=key_cache, kv_cache_v=value_cache,
            block_table=block_table,
            prefix_lens=prefix_lens,
            acc_levels=acc_levels_int,
            verify_len=verify_len,
            K_drafts=K_drafts,
            P_props=P_props,
            block_size=block_size,
            softmax_scale=self.scale,
            out=out3,
        )
        return output

    @torch.inference_mode()
    def _sf_multi_call_forward(
        self,
        layer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: "FlexAttentionMetadata",
    ) -> torch.Tensor:
        """Multi-call FlashAttention path for TiDAR single-forward.

        Replaces one FlexAttention call (with custom mask_mod) by P+1
        flash_attn_varlen_func calls:
          1. Verify sub-call: paged FA on prefix + verify, causal=True.
          2. For each of P proposals: inline FA on
             (gathered prefix K/V) + verify[0..p_j] K/V + own proposal
             K/V, causal=False.

        This bypasses FlexAttention's ~3x slower per-forward cost.
        Each sub-call is small (verify K+1 queries, proposal K+1
        queries) so the per-call overhead is dominated by GEMM, not
        kernel launch.

        Eager-only: the dynamic Python loop over proposals can't be
        captured by cudagraph. Forward path checks
        ``torch.cuda.is_current_stream_capturing()`` before dispatching
        here. SF performance benefit comes from avoiding FlexAttention,
        not from cudagraph.
        """
        from vllm.attention.utils.fa_utils import flash_attn_varlen_func

        num_actual_tokens = attn_metadata.num_actual_tokens
        num_reqs = attn_metadata.num_reqs

        verify_len = int(attn_metadata.tidar_single_forward_verify_len)
        K_drafts = int(attn_metadata.tidar_single_forward_K_drafts)
        # Layout-mode by verify_len: K-mask + no-bonus when
        # verify_len == K_drafts; K+1 default otherwise.
        # ZAP-ONLY: K+1 proposal layout always.
        proposal_seg_len = K_drafts + 1
        # CPU mirrors precomputed in the metadata builder (BEFORE the
        # captured forward) so this closure never touches .cpu() inside
        # cudagraph capture.
        acc_levels_cpu = attn_metadata.tidar_single_forward_proposal_acc_levels_cpu
        prefix_lens_cpu = attn_metadata.tidar_single_forward_prefix_lens_cpu
        assert acc_levels_cpu is not None and prefix_lens_cpu is not None, (
            "SF multi-call FA needs CPU mirrors set by the metadata "
            "builder; got Nones. Check FlexAttentionMetadataBuilder.build().")
        P_props = len(acc_levels_cpu)
        total_per_req = verify_len + P_props * proposal_seg_len

        block_size = attn_metadata.block_size

        # query/key/value: [num_actual_tokens, num_*_heads, head_dim]
        q = query[:num_actual_tokens].view(
            num_actual_tokens, self.num_heads, self.head_size)
        k = key[:num_actual_tokens].view(
            num_actual_tokens, self.num_kv_heads, self.head_size)
        v = value[:num_actual_tokens].view(
            num_actual_tokens, self.num_kv_heads, self.head_size)

        # 1. Cache verify K, V to AR slots. Proposal K/V are scratch
        # (we use inline K, V in the proposal sub-call). For batch>1,
        # each request's verify-len chunk in slot_mapping is at offset
        # i*total_per_req.
        key_cache, value_cache = kv_cache.unbind(0)
        # Gather verify slots across all requests: slot_mapping has
        # total_per_req entries per request; first verify_len are AR.
        slot_mapping_full = attn_metadata.slot_mapping[:num_actual_tokens]
        slot_mapping_2d = slot_mapping_full.view(num_reqs, total_per_req)
        slot_mapping_verify_all = slot_mapping_2d[:, :verify_len].reshape(-1)
        k_2d = k.view(num_reqs, total_per_req, self.num_kv_heads, self.head_size)
        v_2d = v.view(num_reqs, total_per_req, self.num_kv_heads, self.head_size)
        verify_k_all = k_2d[:, :verify_len].reshape(
            num_reqs * verify_len, self.num_kv_heads, self.head_size)
        verify_v_all = v_2d[:, :verify_len].reshape(
            num_reqs * verify_len, self.num_kv_heads, self.head_size)
        torch.ops._C_cache_ops.reshape_and_cache_flash(
            verify_k_all, verify_v_all, key_cache, value_cache,
            slot_mapping_verify_all, self.kv_cache_dtype,
            layer._k_scale, layer._v_scale)

        out_total = torch.empty_like(q)

        # 2. Gather prefix K, V per request, build verify-FA cu_seqlens.
        kc_view = key_cache.reshape(key_cache.shape[0], block_size,
                                    self.num_kv_heads, self.head_size)
        vc_view = value_cache.reshape(value_cache.shape[0], block_size,
                                      self.num_kv_heads, self.head_size)
        block_table = attn_metadata.block_table  # [num_reqs, max_blocks]

        prefix_k_per_req: list[torch.Tensor] = []
        prefix_v_per_req: list[torch.Tensor] = []
        for r in range(num_reqs):
            pl = int(prefix_lens_cpu[r])
            num_pb = (pl + block_size - 1) // block_size
            bids = block_table[r, :num_pb].to(torch.int64)
            pk = kc_view[bids].reshape(
                num_pb * block_size, self.num_kv_heads,
                self.head_size)[:pl].contiguous()
            pv = vc_view[bids].reshape(
                num_pb * block_size, self.num_kv_heads,
                self.head_size)[:pl].contiguous()
            prefix_k_per_req.append(pk)
            prefix_v_per_req.append(pv)

        # 3. Verify sub-call: all reqs' verify queries; varlen kv per req.
        verify_q_all = q.view(num_reqs, total_per_req, self.num_heads,
                              self.head_size)[:, :verify_len].reshape(
                                  num_reqs * verify_len, self.num_heads,
                                  self.head_size).contiguous()
        verify_kv_pieces_k: list[torch.Tensor] = []
        verify_kv_pieces_v: list[torch.Tensor] = []
        verify_kv_lens: list[int] = []
        for r in range(num_reqs):
            verify_kv_pieces_k.append(prefix_k_per_req[r])
            verify_kv_pieces_k.append(verify_k_all[r * verify_len:
                                                   (r + 1) * verify_len])
            verify_kv_pieces_v.append(prefix_v_per_req[r])
            verify_kv_pieces_v.append(verify_v_all[r * verify_len:
                                                   (r + 1) * verify_len])
            verify_kv_lens.append(int(prefix_lens_cpu[r]) + verify_len)
        verify_kv_k = torch.cat(verify_kv_pieces_k, dim=0)
        verify_kv_v = torch.cat(verify_kv_pieces_v, dim=0)
        cu_seqlens_q_v = torch.arange(
            num_reqs + 1, dtype=torch.int32, device=q.device) * verify_len
        # Build cu_seqlens_k_v on GPU: cumulative of (decode_offset[r]
        # + verify_len) with a leading 0. Avoids CPU->GPU copy of a
        # Python list (which can't be captured by cudagraph).
        per_req_kv_lens = (attn_metadata.decode_offset[:num_reqs].to(
            torch.int32) + verify_len)
        cu_seqlens_k_v = torch.zeros(
            num_reqs + 1, dtype=torch.int32, device=q.device)
        cu_seqlens_k_v[1:] = per_req_kv_lens.cumsum(dim=0).to(torch.int32)
        out_v = flash_attn_varlen_func(
            verify_q_all, verify_kv_k, verify_kv_v,
            max_seqlen_q=verify_len,
            cu_seqlens_q=cu_seqlens_q_v,
            max_seqlen_k=max(verify_kv_lens),
            cu_seqlens_k=cu_seqlens_k_v,
            softmax_scale=self.scale,
            causal=True,
        )
        # Scatter verify outputs back into out_total at per-request offsets.
        out_total_2d = out_total.view(num_reqs, total_per_req,
                                      self.num_heads, self.head_size)
        out_total_2d[:, :verify_len] = out_v.view(
            num_reqs, verify_len, self.num_heads, self.head_size)

        # 4. Proposal sub-calls: ONE varlen FA call across all reqs * P
        # proposals. Each "sequence" in the varlen batch is one proposal
        # of one request.
        q_props_all = q.view(num_reqs, total_per_req, self.num_heads,
                             self.head_size)[:, verify_len:].reshape(
                                 num_reqs * P_props * proposal_seg_len,
                                 self.num_heads, self.head_size).contiguous()
        prop_kv_pieces_k: list[torch.Tensor] = []
        prop_kv_pieces_v: list[torch.Tensor] = []
        prop_kv_lens: list[int] = []
        for r in range(num_reqs):
            pl = int(prefix_lens_cpu[r])
            for p_idx in range(P_props):
                p_j = int(acc_levels_cpu[p_idx])
                seg_start = verify_len + p_idx * proposal_seg_len
                seg_end = seg_start + proposal_seg_len
                prop_kv_pieces_k.append(prefix_k_per_req[r])
                prop_kv_pieces_k.append(
                    verify_k_all[r * verify_len:r * verify_len + p_j + 1])
                prop_kv_pieces_k.append(
                    k_2d[r, seg_start:seg_end])
                prop_kv_pieces_v.append(prefix_v_per_req[r])
                prop_kv_pieces_v.append(
                    verify_v_all[r * verify_len:r * verify_len + p_j + 1])
                prop_kv_pieces_v.append(
                    v_2d[r, seg_start:seg_end])
                prop_kv_lens.append(pl + (p_j + 1) + proposal_seg_len)
        prop_kv_k = torch.cat(prop_kv_pieces_k, dim=0)
        prop_kv_v = torch.cat(prop_kv_pieces_v, dim=0)
        cu_seqlens_q_props = torch.arange(
            num_reqs * P_props + 1, dtype=torch.int32, device=q.device
        ) * proposal_seg_len
        # Build cu_seqlens_k_props on GPU. Each proposal's kv_len =
        # prefix_len[r] + (p_j + 1) + proposal_seg_len.
        # Construct as a [num_reqs, P_props] tensor of per-prop kv_lens,
        # then flatten + cumsum.
        prefix_gpu = attn_metadata.decode_offset[:num_reqs].to(
            torch.int32)  # [num_reqs]
        acc_levels_gpu = (attn_metadata
                          .tidar_single_forward_proposal_acc_levels.to(
                              torch.int32))  # [P_props]
        per_prop_kv_lens = (
            prefix_gpu.unsqueeze(1)
            + (acc_levels_gpu + 1).unsqueeze(0)
            + proposal_seg_len
        ).reshape(num_reqs * P_props)
        cu_seqlens_k_props = torch.zeros(
            num_reqs * P_props + 1, dtype=torch.int32, device=q.device)
        cu_seqlens_k_props[1:] = per_prop_kv_lens.cumsum(dim=0).to(torch.int32)
        out_props = flash_attn_varlen_func(
            q_props_all, prop_kv_k, prop_kv_v,
            max_seqlen_q=proposal_seg_len,
            cu_seqlens_q=cu_seqlens_q_props,
            max_seqlen_k=max(prop_kv_lens),
            cu_seqlens_k=cu_seqlens_k_props,
            softmax_scale=self.scale,
            causal=False,
        )
        # Scatter proposal outputs back.
        out_total_2d[:, verify_len:] = out_props.view(
            num_reqs, P_props * proposal_seg_len, self.num_heads,
            self.head_size)

        # 5. Write back into output buffer in original layout.
        # output is [T, num_heads, head_dim] (3D); out_total matches.
        output[:num_actual_tokens, :, :].copy_(out_total)
        return output


def get_kernel_options(query, block_m, block_n,
                       use_direct_build: bool) -> dict[str, Union[int, bool]]:
    # FORCE_USE_FLEX_ATTENTION=True keeps the prefill-shaped flex
    # kernel (vs flex-decoding) -- the flex-decoding kernel is faster
    # for short q but doesn't currently support vllm's paged KV cache
    # + block_mask layout (raises LoweringException at compile time).
    kernel_options: dict[str, Union[int, bool]] = {
        "FORCE_USE_FLEX_ATTENTION": True,
    }
    if use_direct_build:
        kernel_options["BLOCK_M"] = block_m
        kernel_options["BLOCK_N"] = block_n
        return kernel_options
    else:
        kernel_options["BLOCK_M"] = 64
        kernel_options["BLOCK_N"] = 64
        if query.dtype == torch.float32:
            kernel_options["BLOCK_M"] = 32
            kernel_options["BLOCK_N"] = 32
        # if current_platform.is_cuda():
        if torch.cuda.is_available():
            device_props = torch.cuda.get_device_properties()
            max_shared_memory = device_props.shared_memory_per_block_optin
            if max_shared_memory < 144 * 1024:
                kernel_options["BLOCK_M"] = kernel_options["BLOCK_M"] // 2
                kernel_options["BLOCK_N"] = kernel_options["BLOCK_N"] // 2

    return kernel_options
