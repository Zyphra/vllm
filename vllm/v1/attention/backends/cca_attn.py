# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field
from typing import ClassVar, Optional

import torch

from vllm.v1.attention.backend import (AttentionBackend, AttentionCGSupport,
                                       CommonAttentionMetadata)
from vllm.config import VllmConfig
from vllm.v1.attention.backends.utils import (PAD_SLOT_ID,
                                              split_decodes_and_prefills)
from vllm.v1.attention.backends.mamba_attn import (
    BaseMambaAttentionMetadata,
    BaseMambaAttentionMetadataBuilder)
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec


class CCAAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "CCA_ATTN"

    @staticmethod
    def get_builder_cls() -> type["CCAAttentionMetadataBuilder"]:
        return CCAAttentionMetadataBuilder


@dataclass
class CCAAttentionMetadata(BaseMambaAttentionMetadata):
    # Optional override for the WRITE side of the CCA state cache. When None
    # (the default), the same tensor is used for both reads and writes
    # (verify forward semantics: read post-N-1 state, write post-K state,
    # then commit_spec_decode_state overwrites with post-acceptance state).
    # When set, ``state_indices_tensor`` is the READ slot ordering and
    # ``state_indices_tensor_write`` is the WRITE slot ordering; they
    # differ for TiDAR drafter forwards under the single-state design.
    state_indices_tensor_write: Optional[torch.Tensor] = None

    # Host-side mirrors of query_start_loc_p / has_initial_states_p. CCA's
    # prefill loop iterates over per-request slices and previously read these
    # values out of GPU 0-dim tensors, which forced an implicit
    # cudaStreamSynchronize per slice index. Both fields are None when
    # there are no prefills in the batch.
    query_start_loc_p_cpu: Optional[torch.Tensor] = None  # int64, [P+1]
    has_initial_states_p_cpu: Optional[torch.Tensor] = None  # bool, [P]

    # Host list of state-cache slot indices, in batch order. Used by CCA's
    # prefill loop and commit_spec_decode_state to index conv_states / prev_hs
    # with Python ints rather than 0-dim GPU tensors.
    state_indices_list: Optional[list] = None  # list[int], len = num_reqs

    # When True, this metadata describes a TiDAR verification step where each
    # request has K+1 query positions to be verified.
    spec_decode_mode: bool = False

    # When True, this metadata describes a TiDAR *drafter* forward.
    drafter_pass: bool = False

    # TiDAR single-forward (sparse-proposal design):
    # When ``tidar_single_forward_verify_len`` is set, the per-request input
    # layout is ``[verify (verify_len), proposal_1 (K), ..., proposal_P (K)]``
    # with total length ``verify_len + P*K`` per req.
    tidar_single_forward_verify_len: Optional[int] = None
    # [P_proposals] int64 GPU tensor; values in [0, verify_len-1].
    tidar_single_forward_proposal_acc_levels: Optional[torch.Tensor] = None


class CCAAttentionMetadataBuilder(
        BaseMambaAttentionMetadataBuilder[CCAAttentionMetadata]):
    metadata_cls = CCAAttentionMetadata
    supports_update_block_table: bool = False
    # TiDAR runs K+1 query positions per request; UNIFORM_BATCH allows the
    # FULL cudagraph dispatcher to capture this shape. Without this bump the
    # cudagraph_mode resolver downgrades spec-decode batches to PIECEWISE.
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_BATCH)

    def __init__(self, kv_cache_spec: AttentionSpec, layer_names: list[str],
                 vllm_config: VllmConfig, device: torch.device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        # Persistent GPU buffers for the per-prefill-request metadata
        # tensors that CCA's vectorized prefill kernel reads
        # (has_initial_states_p) and that downstream attn arithmetic
        # references (query_start_loc_p). These addresses are baked into
        # the captured FULL cudagraph at warmup; subsequent build() calls
        # for both verify and drafter forwards write into THESE buffers
        # in-place rather than allocating fresh tensors.
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self._has_initial_states_p_buf = torch.zeros(
            max_num_seqs, dtype=torch.bool, device=device)
        self._query_start_loc_p_buf = torch.zeros(
            max_num_seqs + 1, dtype=torch.int32, device=device)

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata,
    ) -> CCAAttentionMetadata:
        """Override base check that forbids non-decode-only capture.

        TiDAR captures K+1 query positions per request (uniform K+1 verify
        batch). We bypass the base assertion that requires num_reqs ==
        num_actual_tokens (= decode-only), instead allowing the case where
        num_actual_tokens is a uniform multiple of num_reqs.
        """
        m = common_attn_metadata
        # max_query_len computed from query_start_loc by the caller; do not
        # overwrite to 1 because that lies about the K+1 verify layout.
        return self.build(0, m)

    def build(self,
              common_prefix_len: int,
              common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False) -> CCAAttentionMetadata:
        # Defer to the base class for the common fields, then augment.
        meta = self._compute_common_metadata(common_attn_metadata)

        # v0.15 parity: respect state_indices_tensor_override (read-side
        # override). The TiDAR drafter sets this to the AR slot (col 0 of
        # CCA's block table) so the drafter reads the post-acceptance
        # state. Without this override, _compute_common_metadata uses
        # mamba_get_block_table_tensor(FA_block_table, seq_lens, ...)[:, 0]
        # which gives a DIFFERENT slot than where the verifier wrote
        # (different seq_lens → different start_indices). The drafter then
        # reads stale data and produces wrong logits, rejected 100%.
        _sit_ovr = getattr(common_attn_metadata,
                           "state_indices_tensor_override", None)
        if _sit_ovr is not None:
            meta.state_indices_tensor = _sit_ovr

        # Optional TiDAR-specific overrides plumbed via attribute injection
        # on the CommonAttentionMetadata object (see drafter forward path).
        meta.state_indices_tensor_write = getattr(
            common_attn_metadata,
            "state_indices_tensor_write_override", None)
        meta.spec_decode_mode = bool(getattr(
            common_attn_metadata, "cca_spec_decode_mode", False))
        meta.drafter_pass = bool(getattr(
            common_attn_metadata, "cca_drafter_pass", False))
        meta.tidar_single_forward_verify_len = getattr(
            common_attn_metadata,
            "tidar_single_forward_verify_len", None)
        meta.tidar_single_forward_proposal_acc_levels = getattr(
            common_attn_metadata,
            "tidar_single_forward_proposal_acc_levels", None)

        # Host-side mirrors. CCA's prefill loop and commit consume these.
        num_reqs = common_attn_metadata.num_reqs
        num_decodes, num_prefills, num_decode_tokens, _ = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold))

        if num_prefills > 0:
            # Pin host-side query_start_loc_p_cpu so cca.py can iterate
            # over prefill slices with Python ints (no per-index sync).
            if common_attn_metadata.query_start_loc_cpu is not None:
                meta.query_start_loc_p_cpu = (
                    common_attn_metadata.query_start_loc_cpu[
                        -num_prefills - 1:] - num_decode_tokens)
            # Pin host-side has_initial_states_p_cpu (bool, [P]).
            if common_attn_metadata.num_computed_tokens_cpu is not None:
                meta.has_initial_states_p_cpu = (
                    common_attn_metadata.num_computed_tokens_cpu[
                        num_reqs - num_prefills:num_reqs] > 0)
            # Pin GPU mirror into the persistent buffer so a captured
            # FULL graph (bound to the buffer's address) reads current-
            # step values on every replay.
            if meta.has_initial_states_p_cpu is not None:
                self._has_initial_states_p_buf[:num_prefills].copy_(
                    meta.has_initial_states_p_cpu, non_blocking=True)
                meta.has_initial_states_p = (
                    self._has_initial_states_p_buf[:num_prefills])

            # Pin query_start_loc_p (GPU) similarly.
            qsl_p_src = (common_attn_metadata.query_start_loc[
                -num_prefills - 1:] - num_decode_tokens)
            self._query_start_loc_p_buf[:num_prefills + 1].copy_(
                qsl_p_src, non_blocking=True)
            meta.query_start_loc_p = (
                self._query_start_loc_p_buf[:num_prefills + 1])

        # Materialize per-req slot indices as a Python list once. Used by
        # CCA's prefill loop and commit_spec_decode_state to index
        # conv_states / prev_hs with Python ints. Skip on pure-decode
        # batches (no prefill loop runs).
        if num_prefills > 0 and meta.state_indices_tensor is not None:
            meta.state_indices_list = meta.state_indices_tensor.tolist()

        return meta
