# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import os

import torch
import torch.nn as nn

from vllm.model_executor.layers.attention.attention import Attention
from vllm.compilation.monitor import set_cudagraph_capturing_enabled
from vllm.config import CUDAGraphMode, VllmConfig, get_layers_from_vllm_config
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.cca import CCA
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.v1.attention.backends.utils import (CommonAttentionMetadata,
                                              PAD_SLOT_ID)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.tidar_single_forward import (
    build_single_forward_inputs, extract_proposal_hidden_states,
    select_proposal_index)

logger = init_logger(__name__)

DEFAULT_TIDAR_MASK_TOKEN_ID = 4

# Default proposal acc levels for K=16. Acc levels are "num drafts
# accepted" semantically (matching commit_spec_decode_state). For K=16
# typical accepted_per_block is ~7 on MATH-500, ~4 on MMLU; defaults
# cover the empirical distribution.
DEFAULT_TIDAR_PROPOSAL_ACC_LEVELS_K16 = (4, 7, 10)


import os as _os


def _tidar_no_bonus() -> bool:
    """Whether to run SF in the paper-faithful K-mask + no-bonus layout.

    Opt-in via VLLM_TIDAR_NO_BONUS=1. Default off; K+1 + verifier-bonus
    behavior is unchanged.

    When on:
      * verify_len = K (no anchor slot in the verify segment)
      * proposal_seg_len = K (matches SBD training; one cache block
        per (req, proposal))
      * proposal mask RoPE positions = [p_j..p_j+K-1] (immediately
        after the assumed-accepted draft window, matching training's
        "masked block right after clean prefix" pattern)
      * verify-segment attention is bidirectional (masked-LM-style),
        not causal as in the K+1 layout
      * rejection sampler skips the all-accept bonus commit

    Reading at function-call time (not module load) so a test harness
    can toggle the env between LLM() instantiations within one process.
    """
    return _os.environ.get("VLLM_TIDAR_NO_BONUS", "0") == "1"


def _parse_proposal_acc_levels(env_val: str) -> tuple[int, ...]:
    """Parse comma-separated acc levels env var. Empty -> ()."""
    if not env_val.strip():
        return ()
    return tuple(int(x.strip()) for x in env_val.split(",") if x.strip())


class TiDARProposer(EagleProposer):
    """Two-forward TiDAR prototype using the target model and FlashAttention."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ) -> None:
        super().__init__(vllm_config, device, runner)
        self.mask_token_id = self._resolve_mask_token_id()
        self.diff_temperature: float = float(
            vllm_config.speculative_config.tidar_diff_temperature)
        # Stash the most recent draft distribution so the model runner can
        # forward it to the rejection sampler without changing the public
        # `propose()` return signature. None when diff_temperature == 0
        # (Dirac drafter — the rejection sampler's NO_DRAFT_PROBS branch
        # is exact in that regime).
        self.last_draft_probs: Optional[torch.Tensor] = None
        # Raw drafter logits (pre-softmax, pre-temperature), stashed in
        # parallel with last_draft_probs. Populated even when
        # diff_temperature == 0 (where last_draft_probs is None). The model
        # runner forwards these to spec_decode_metadata.draft_logits so the
        # mix-logit v1 sampler can construct
        #   mixed = w*target_logits + (1-w)*draft_logits
        # without relying on the (None-at-Dirac) draft_probs.
        self.last_draft_logits: Optional[torch.Tensor] = None
        # Tier 3: tracks whether the drafter-pass FULL cudagraph has been
        # captured for each (num_input_tokens) shape. The captured graph
        # has its CCA gather/scatter operands baked at *drafter* metadata
        # (read=AR, write=draft slot under path 2) — distinct from the
        # verify-shape graph captured during the standard warmup. Capture
        # happens lazily at the first propose() call for each shape;
        # subsequent calls replay. We track it here so we only re-enable
        # the global cudagraph_capturing flag once per shape, which
        # otherwise would defeat its safety-net purpose.
        self._drafter_captured_sizes: set[int] = set()

        # TiDAR single-forward (sparse-proposal) configuration.
        # SF is the DEFAULT mode now (flipped during the
        # tidar_TF + tidar_SF merge). Opt back to TF (two-forward) with
        # VLLM_TIDAR_TWO_FORWARD=1. Legacy VLLM_TIDAR_SINGLE_FORWARD=1
        # is still recognized and is now redundant (kept so existing
        # scripts/sweeps don't break); if both are set,
        # VLLM_TIDAR_TWO_FORWARD wins (TF mode).
        # Proposal acc levels: VLLM_TIDAR_PROPOSAL_ACC_LEVELS=4,7,10
        # (semantics = "num drafts accepted", matching
        # commit_spec_decode_state). Empty/unset under SF -> uses the
        # K-aware DEFAULT_TIDAR_PROPOSAL_ACC_LEVELS_K16.
        # NOTE: SF requires VLLM_ATTENTION_BACKEND=FLEX_ATTENTION at the
        # vLLM level; without it acceptance silently degrades to ~1%.
        _tf_explicit = os.environ.get(
            "VLLM_TIDAR_TWO_FORWARD", "0") == "1"
        self.single_forward_mode: bool = not _tf_explicit

        # SF mode requires the SF Triton kernel which lives in
        # flex_attention.py. With FA backend, the SF path falls through
        # to standard FA forward (no structured mask), collapsing
        # mean accept to ~1.05 (drafts unanimously rejected). Warn
        # loudly at init so users don't silently see broken SF perf.
        # TF mode is unaffected (FA is the recommended TF backend).
        if self.single_forward_mode:
            _attn_backend = os.environ.get(
                "VLLM_ATTENTION_BACKEND", "").upper()
            if _attn_backend == "FLASH_ATTN":
                logger.warning(
                    "TiDAR SF mode is running with FLASH_ATTN backend. "
                    "The SF kernel (verify-causal + proposals-bidirectional "
                    "structured mask) lives in flex_attention.py only; "
                    "FA backend falls through to standard FA forward and "
                    "produces broken drafts (mean accept ~1.05). "
                    "Set VLLM_ATTENTION_BACKEND=FLEX_ATTENTION for SF, "
                    "or VLLM_TIDAR_TWO_FORWARD=1 for TF mode (which is "
                    "FA-recommended).")
        sf_levels_env = os.environ.get("VLLM_TIDAR_PROPOSAL_ACC_LEVELS", "")
        if sf_levels_env:
            self.proposal_acc_levels: tuple[int, ...] = (
                _parse_proposal_acc_levels(sf_levels_env))
        elif self.single_forward_mode:
            self.proposal_acc_levels = DEFAULT_TIDAR_PROPOSAL_ACC_LEVELS_K16
        else:
            self.proposal_acc_levels = ()
        # Validate levels are in [0, K].
        K = self.num_speculative_tokens
        if self.single_forward_mode:
            if not self.proposal_acc_levels:
                raise ValueError(
                    "SF mode (default) requires non-empty "
                    "VLLM_TIDAR_PROPOSAL_ACC_LEVELS or a default for K. "
                    "Either set VLLM_TIDAR_PROPOSAL_ACC_LEVELS, or "
                    "switch to TF with VLLM_TIDAR_TWO_FORWARD=1.")
            for lvl in self.proposal_acc_levels:
                if not (0 <= lvl <= K):
                    raise ValueError(
                        f"Proposal acc level {lvl} out of range [0, {K}]; "
                        f"check VLLM_TIDAR_PROPOSAL_ACC_LEVELS.")
            _layout = ("hybrid no_bonus (verify K+1, proposal K, "
                       "bonus zapped)" if _tidar_no_bonus()
                       else "K+1 default (verify K+1, proposal K+1)")
            logger.info(
                "TiDAR single-forward mode ENABLED with K=%d, P=%d, "
                "acc_levels=%s. Verify_len=%d. Layout=%s.",
                K, len(self.proposal_acc_levels),
                self.proposal_acc_levels, K + 1, _layout)
        self.proposal_acc_levels_gpu: Optional[torch.Tensor] = None
        # Scratch block IDs for proposal mask K/V writes. Lazily
        # allocated on the first verify call (after the runner has
        # set up num_gpu_blocks). Shape: [max_num_seqs, P].
        # MVP hack: uses the LAST max_num_seqs * P blocks of the cache.
        # Real allocator integration is a follow-up (would reduce the
        # allocator-visible num_gpu_blocks by max_num_seqs * P).
        self.tidar_scratch_block_ids: Optional[torch.Tensor] = None
        # Cache of last step's inflated total_num_scheduled_tokens (=
        # num_reqs * (K+1 + P*K)). The runner's _preprocess reads
        # scheduler_output.total_num_scheduled_tokens for slicing
        # input_ids / positions before the forward; that scheduler-side
        # value doesn't know about the inflation. _preprocess checks
        # this attribute and uses it when set. Cleared at the end of
        # the step (or on next call into maybe_extend_verify_input).
        self.last_inflated_total: Optional[int] = None

    def _ensure_scratch_blocks(self) -> torch.Tensor:
        """Lazily allocate scratch block IDs for proposal mask K/V.

        Each (req, proposal) needs K+1 mask slots (1 bonus mask + K draft
        masks). For K=block_size this requires 2 cache blocks per
        (req, proposal). Reserved layout shape: [max_num_seqs, P, 2].

        MVP: the last ``max_num_seqs * P * 2`` block IDs of the cache.
        This is a HACK -- the allocator doesn't know these blocks are
        reserved and may use them for real requests under heavy load,
        corrupting their KV. For the MVP this is acceptable for
        b=1..16 testing; production needs proper allocator integration.
        """
        if self.tidar_scratch_block_ids is not None:
            return self.tidar_scratch_block_ids
        num_gpu_blocks = self.vllm_config.cache_config.num_gpu_blocks
        if num_gpu_blocks is None:
            raise RuntimeError(
                "num_gpu_blocks not set; call _ensure_scratch_blocks "
                "AFTER cache init.")
        max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs
        P = len(self.proposal_acc_levels)
        if P == 0:
            raise RuntimeError(
                "scratch blocks requested but proposal_acc_levels is empty")
        # K-mask layout (VLLM_TIDAR_NO_BONUS=1): 1 cache block per
        # (req, proposal) since K = block_size. Otherwise (K+1 default):
        # 2 blocks per (req, proposal) for the K+1 mask slots layout.
        blocks_per_proposal = 2  # ZAP-ONLY: K+1 layout always
        reserved = max_num_seqs * P * blocks_per_proposal
        if reserved > num_gpu_blocks // 4:
            logger.warning(
                "TiDAR scratch reservation (%d blocks = max_num_seqs %d "
                "* P %d * %d-blocks-per-proposal) is more than 25%% of "
                "num_gpu_blocks (%d); consider lowering max_num_seqs or P, "
                "or implementing proper allocator integration.",
                reserved, max_num_seqs, P, blocks_per_proposal,
                num_gpu_blocks)
        # IDs: [num_gpu_blocks - reserved, num_gpu_blocks - 1].
        # Reshape to [max_num_seqs, P, 2]: req i, proposal p, sub-block s ->
        # ID num_gpu_blocks - reserved + i*P*2 + p*2 + s.
        ids = torch.arange(
            num_gpu_blocks - reserved, num_gpu_blocks,
            dtype=torch.int32, device=self.device).view(
                max_num_seqs, P, blocks_per_proposal)
        self.tidar_scratch_block_ids = ids
        logger.info(
            "TiDAR scratch blocks allocated: ids [%d, %d), shape "
            "[max_num_seqs=%d, P=%d, blocks_per_proposal=%d]",
            num_gpu_blocks - reserved, num_gpu_blocks, max_num_seqs, P,
            blocks_per_proposal)
        return ids

    # ------------------------------------------------------------------
    # Single-forward TiDAR runner hooks
    # ------------------------------------------------------------------

    @property
    def verify_len(self) -> int:
        """Verify segment length per request = K+1 (1 anchor + K drafts).

        Under VLLM_TIDAR_NO_BONUS=1 (hybrid K-mask + no-bonus mode),
        slot 0 of verify is still the latest-accepted token (already in
        KV cache; re-processed harmlessly so the scheduler's
        ``1 + len(spec_decode_tokens) = K+1`` token budget matches the
        runner). The rejection sampler ignores slot 0's bonus output
        (zapped); slots 1..K are the K drafts. Proposal segment uses
        K (paper-aligned), not K+1.
        """
        return self.num_speculative_tokens + 1

    def _ensure_acc_levels_gpu(self) -> torch.Tensor:
        """Lazily move proposal_acc_levels to GPU as int64."""
        if self.proposal_acc_levels_gpu is None:
            self.proposal_acc_levels_gpu = torch.tensor(
                self.proposal_acc_levels,
                dtype=torch.int64, device=self.device)
        return self.proposal_acc_levels_gpu

    @torch.inference_mode()
    def maybe_extend_verify_input(
        self,
        num_reqs: int,
        num_scheduled_tokens: int,
    ) -> int:
        """Extend the runner's input_ids / positions / seq_lens /
        query_start_loc buffers in-place for single-forward TiDAR.

        Called by the runner ONCE per step, AFTER the standard
        ``_prepare_inputs`` (which builds the K+1 verify layout) and
        BEFORE the per-kv-cache-group attention metadata is built.

        Per request, the layout extends from K+1 verify tokens to
        ``verify_len + P*K`` = ``K + 1 + P * K`` tokens. The new mask
        positions get ``mask_token_id`` for input_ids, shifted rotary
        positions per proposal level, and scratch-block-backed slots
        for K/V writes (slot_mapping is written into the FA / FlexAttn
        block_table's persistent slot_mapping buffer).

        Note this does NOT call the per-group attention metadata builders
        -- those run later in the runner's normal loop and pick up the
        inflated seq_lens / query_start_loc / extended slot_mapping
        automatically. The runner must also call
        ``set_tidar_single_forward_metadata`` on each group's
        ``common_attn_metadata`` so the CCA + FA builders consume the
        TiDAR fields.

        Returns:
            new num_scheduled_tokens (= ``num_reqs * (verify_len + P*(K+1))``).
        """
        if not self.single_forward_mode:
            return num_scheduled_tokens

        runner = self.runner
        K = self.num_speculative_tokens
        verify_len = self.verify_len
        P = len(self.proposal_acc_levels)
        # K+1 masks per proposal (bonus + K drafts) -- TF Fix 4 analog.
        # K-mask: each proposal has K masks at positions [p_j..p_j+K-1].
        # K+1 default: K+1 masks at [p_j+1..p_j+K+1] with slot 0 at the
        # bonus position.
        proposal_seg_len = K + 1  # ZAP-ONLY: K+1 layout always
        total_per_req = verify_len + P * proposal_seg_len
        new_num_scheduled = num_reqs * total_per_req

        # Sanity check the incoming layout.
        if num_scheduled_tokens != num_reqs * verify_len:
            raise ValueError(
                f"Single-forward extension expects num_scheduled_tokens = "
                f"num_reqs * verify_len = {num_reqs} * {verify_len} = "
                f"{num_reqs * verify_len}; got {num_scheduled_tokens}. "
                "Check the verify input layout upstream.")
        if new_num_scheduled > runner.input_ids.gpu.shape[0]:
            raise RuntimeError(
                f"Single-forward extended num_scheduled_tokens "
                f"{new_num_scheduled} exceeds runner.input_ids buffer "
                f"size {runner.input_ids.gpu.shape[0]}. Increase "
                "max_num_tokens (runner buffer sizing) or reduce P / "
                "max_num_seqs.")

        scratch_ids = self._ensure_scratch_blocks()  # [max_num_seqs, P]
        acc_levels_gpu = self._ensure_acc_levels_gpu()
        scratch_for_step = scratch_ids[:num_reqs].contiguous()

        # FA group's block_table: lookup by first attention layer.
        # (TiDAR + SMoE has one FA group; this assumption matches the
        # current production layout. If multiple FA groups arise, this
        # needs a per-layer-name lookup.)
        _, fa_group_id = (
            self._get_metadata_builder_and_group_id_for_layer(
                self.attn_layer_names[0]))
        fa_blk_table_obj = runner.input_batch.block_table[fa_group_id]
        fa_block_table = fa_blk_table_obj.get_device_tensor(num_reqs)
        # Per-group block_size (FA's may differ from group[0]'s; the engine
        # logs "Setting attention block size to N tokens" at startup).
        block_size = (
            runner.kv_cache_config.kv_cache_groups[fa_group_id]
            .kv_cache_spec.block_size)

        # Reshape the existing verify input.
        verify_ids = runner.input_ids.gpu[:num_scheduled_tokens].view(
            num_reqs, verify_len)
        verify_positions = runner.positions.gpu[:num_scheduled_tokens].view(
            num_reqs, verify_len)
        # base_positions[i] = absolute rotary position of req i's first
        # verify token (anchor).
        base_positions = verify_positions[:, 0].contiguous()

        ext_input_ids, ext_positions, ext_slot_mapping = (
            build_single_forward_inputs(
                verify_token_ids=verify_ids,
                base_positions=base_positions,
                block_table=fa_block_table,
                K=K,
                proposal_acc_levels=list(self.proposal_acc_levels),
                mask_token_id=self.mask_token_id,
                block_size=block_size,
                max_model_len=self.max_model_len,
                scratch_block_table=scratch_for_step,
                verify_len=verify_len,
            ))

        # Write back: input_ids, positions.
        runner.input_ids.gpu[:new_num_scheduled].copy_(
            ext_input_ids.to(runner.input_ids.gpu.dtype))
        runner.positions.gpu[:new_num_scheduled].copy_(
            ext_positions.to(runner.positions.gpu.dtype))

        # Write FA group's slot_mapping (verify + scratch slots).
        fa_blk_table_obj.slot_mapping.gpu[:new_num_scheduled].copy_(
            ext_slot_mapping.to(torch.int64))
        # Other slot_mapping slots get -1 (existing convention).
        fa_blk_table_obj.slot_mapping.gpu[new_num_scheduled:].fill_(-1)

        # Append scratch block IDs to each request's block_table so
        # that FlexAttention's physical_to_logical map covers them.
        # K+1 layout: scratch_for_step is [B, P, 2] -> P*2 blocks per req.
        # Place at [N_used..N_used + P*2 - 1] where N_used = ceil((prefix
        # + verify_len) / block_size).
        n_used = ((runner.seq_lens.gpu[:num_reqs].to(torch.int64)
                   + block_size - 1) // block_size)  # [B]
        scratch_flat = scratch_for_step.reshape(num_reqs, -1).to(
            fa_block_table.dtype)
        n_scratch_cols = scratch_flat.shape[1]
        col_offsets = torch.arange(n_scratch_cols,
                                   device=fa_block_table.device,
                                   dtype=torch.int64)
        write_cols = n_used.unsqueeze(1) + col_offsets.unsqueeze(0)
        row_idx = torch.arange(num_reqs,
                               device=fa_block_table.device,
                               dtype=torch.int64).unsqueeze(1).expand(
                                   -1, n_scratch_cols)
        fa_block_table[row_idx, write_cols] = scratch_flat

        # Inflate seq_lens by P*(K+1) per req (K+1 mask positions per
        # proposal segment).
        inflate = P * proposal_seg_len
        runner.seq_lens.gpu[:num_reqs].add_(inflate)
        # CPU mirrors of seq_lens are kept in sync.
        runner.seq_lens.cpu[:num_reqs].add_(inflate)
        runner.seq_lens.np[:num_reqs] += inflate

        # Rebuild query_start_loc with new per-req length.
        device = runner.query_start_loc.gpu.device
        new_cu_gpu = torch.arange(num_reqs + 1, dtype=torch.int32,
                                  device=device) * total_per_req
        runner.query_start_loc.gpu[:num_reqs + 1].copy_(new_cu_gpu)
        new_cu_cpu = torch.arange(num_reqs + 1, dtype=torch.int32) * \
            total_per_req
        runner.query_start_loc.cpu[:num_reqs + 1].copy_(new_cu_cpu)
        runner.query_start_loc.np[:num_reqs + 1] = new_cu_cpu.numpy()

        del acc_levels_gpu  # ensure the lazy ensure is invoked
        # Cache the inflated total so runner's _preprocess uses it.
        self.last_inflated_total = new_num_scheduled
        return new_num_scheduled

    def set_tidar_single_forward_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> None:
        """Attach the 3 TiDAR single-forward fields to a per-group
        common_attn_metadata so CCA + FlexAttention metadata builders
        pick them up. Idempotent / no-op when single_forward_mode is
        off."""
        if not self.single_forward_mode:
            return
        acc_levels_gpu = self._ensure_acc_levels_gpu()
        setattr(common_attn_metadata,
                "tidar_single_forward_verify_len", self.verify_len)
        setattr(common_attn_metadata,
                "tidar_single_forward_K_drafts",
                self.num_speculative_tokens)
        setattr(common_attn_metadata,
                "tidar_single_forward_proposal_acc_levels", acc_levels_gpu)
        # Pass scratch_block_ids [B, P, 2] so the FlexAttention SF
        # mask_mod can detect when a physical KV slot lives in a
        # scratch block and synthesize the correct kv_local for
        # proposal-to-own-block attention.
        scratch_ids = self._ensure_scratch_blocks()
        setattr(common_attn_metadata,
                "tidar_single_forward_scratch_block_ids", scratch_ids)

    @torch.inference_mode()
    def extract_drafts_from_hidden(
        self,
        hidden_states: torch.Tensor,
        num_accepted_per_req: torch.Tensor,
        tie_break: str = "closest_below",
    ) -> torch.Tensor:
        """Post-rejection-sampling draft extraction for single-forward.

        Replaces the second ``propose()`` forward in single-forward mode.

        Args:
            hidden_states:        [num_scheduled_tokens, hidden_size]
                                  from the verify forward (covers verify
                                  + proposal segments).
            num_accepted_per_req: [B] int -- output of the rejection
                                  sampler.
            tie_break:            "closest_below" (default) or "closest"
                                  for ``select_proposal_index``.

        Returns:
            [B, K] int draft token IDs for next step.
        """
        if not self.single_forward_mode:
            raise RuntimeError(
                "extract_drafts_from_hidden called outside single-forward "
                "mode; runner should call propose() instead.")
        K = self.num_speculative_tokens
        verify_len = self.verify_len
        P_props = len(self.proposal_acc_levels)
        B = num_accepted_per_req.shape[0]
        acc_levels_gpu = self._ensure_acc_levels_gpu()

        # 1. Pick the closest-acc proposal per req.
        proposal_indices = select_proposal_index(
            num_accepted_per_req.to(torch.int64), acc_levels_gpu,
            tie_break=tie_break)

        # 2. Extract the K mask hidden states of the selected proposal.
        proposal_hs = extract_proposal_hidden_states(
            hidden_states, proposal_indices, K, verify_len, P_props)

        # 3. Drafter logits + sampling. Mirrors the existing propose()'s
        # logits + sampling block.
        if self._dspark_active():
            draft_token_ids = self._dspark_sample_drafts(proposal_hs, B)
            return draft_token_ids.view(B, K).to(torch.int32)

        logits = self.model.compute_logits(proposal_hs)
        if logits is None:
            raise RuntimeError(
                "TiDAR target model did not return logits during draft "
                "extraction.")
        # Stash raw logits for mix-logit v1 (consumed in
        # gpu_model_runner.py via spec_decode_metadata.draft_logits).
        # Detached to drop the autograd graph; inference-mode call
        # guarantees no graph anyway.
        self.last_draft_logits = logits.detach().contiguous()
        if self.diff_temperature == 0.0:
            self.last_draft_probs = None
            draft_token_ids = logits.argmax(dim=-1)
        else:
            scaled_logits = logits.to(torch.float32) / self.diff_temperature
            draft_probs = torch.softmax(scaled_logits, dim=-1)
            draft_token_ids = torch.multinomial(
                draft_probs, num_samples=1).squeeze(-1)
            self.last_draft_probs = draft_probs.contiguous()
        return draft_token_ids.view(B, K).to(torch.int32)

    def warmup_capture_drafter_graphs(self) -> None:
        """Approach B: warmup-time drafter graph capture.

        Called from runner.capture_model AFTER the standard verifier
        captures complete, INSIDE that method's already-active
        graph_capture(device=self.device) context. We're on a
        non-default stream and the runner's persistent buffers were
        primed by the just-finished _dummy_run.

        For each captured FULL shape, we build the drafter-specific
        CCA metadata (state_indices_tensor_write_override = draft slot,
        cca_drafter_pass=True), then call self.model with the drafter
        BatchDescriptor. The CUDAGraphWrapper sees no entry for that
        descriptor and captures the drafter graph at that moment, with
        the *drafter* metadata in scope — write→draft and skip_writes
        baked in at capture time, exactly like v0.15's lazy capture
        did. Subsequent runtime propose() calls hit a populated entry
        and replay.

        Only runs under TiDAR TF mode (single_forward_mode=False) on
        FULL-cudagraph capture sizes whose shape is a multiple of
        (K+1); other capture sizes don't correspond to a TiDAR drafter
        forward.
        """
        if self.single_forward_mode:
            # SF mode has no separate drafter pass.
            return
        if not self.cca_layer_names:
            # No CCA in this model -> no read=AR/write=draft routing.
            return
        runner = self.runner
        K_plus_1 = self.num_speculative_tokens + 1

        # Lazily resolve the CCA metadata builder + kv-cache group id.
        if self.cca_metadata_builder is None or not hasattr(
                self, "cca_kv_cache_group_id"):
            (self.cca_metadata_builder, self.cca_kv_cache_group_id) = \
                self._get_metadata_builder_and_group_id_for_layer(
                    self.cca_layer_names[0])
        if self.attn_metadata_builder is None:
            self.attn_metadata_builder = self._get_metadata_builder_for_layer(
                self.attn_layer_names[0])

        # CCA block table must have >=2 columns (col 0 = AR, col 1 = draft).
        cca_blk_obj = runner.input_batch.block_table[self.cca_kv_cache_group_id]
        cca_blk_tensor_full = cca_blk_obj.get_device_tensor(
            runner.scheduler_config.max_num_seqs)
        if cca_blk_tensor_full.shape[1] < 2:
            return

        # FlashAttn group's block table is what
        # CommonAttentionMetadata.block_table_tensor must point at --
        # the FA metadata builder consumes it for KV lookup. The CCA
        # builder also reads `common_attn_metadata.block_table_tensor`
        # to derive its default state_indices_tensor (via
        # mamba_get_block_table_tensor), and the CCA-specific draft
        # slot is plumbed separately via
        # `state_indices_tensor_write_override`.
        #
        # CRITICAL: look up FA's group_id dynamically. Hardcoding [0]
        # picked the CCA group (shape [N, K+1]) instead of FA (shape
        # [N, max_blocks_per_req]), so the captured graph traced FA
        # attention against the WRONG block table addresses and produced
        # 0% accept at runtime. (Both groups happen to live at adjacent
        # addresses 23407005301248 vs 23407005301760 with shape [1,17]
        # vs [1,128], differing by 512 bytes — the captured FA ops then
        # read garbage from the CCA-shaped buffer at replay.)
        _, _fa_group_id = (
            self._get_metadata_builder_and_group_id_for_layer(
                self.attn_layer_names[0]))
        fa_blk_obj = runner.input_batch.block_table[_fa_group_id]
        fa_blk_tensor_full = fa_blk_obj.get_device_tensor(
            runner.scheduler_config.max_num_seqs)

        # Enumerate drafter-pass capture descriptors registered with
        # the dispatcher. These have is_drafter_pass=True; the standard
        # warmup loop skips them (see CudagraphDispatcher
        # .get_capture_descs filtering is_drafter_pass=False).
        from vllm.config import CUDAGraphMode
        drafter_descs = [
            d for d in runner.cudagraph_dispatcher.cudagraph_keys[
                CUDAGraphMode.FULL]
            if getattr(d, "is_drafter_pass", False)
        ]
        if not drafter_descs:
            return

        # Capture largest first (matches main capture loop's strategy
        # for the FULL memory pool).
        drafter_descs = sorted(
            drafter_descs, key=lambda d: d.num_tokens, reverse=True)

        # Builders inside _warmup_capture_one_drafter_shape inplace-
        # mutate persistent inference tensors (e.g., FlexAttention's
        # inverse_block_table buffer). They're safe under inference_mode
        # which gpu_model_runner.capture_model does NOT wrap us in --
        # the standard captures got it via the @torch.inference_mode()
        # decorator on _dummy_run. Match that here.
        with torch.inference_mode():
            for desc in drafter_descs:
                num_input_tokens = desc.num_tokens
                if num_input_tokens % K_plus_1 != 0:
                    continue
                batch_size = num_input_tokens // K_plus_1
                if batch_size < 1:
                    continue
                if batch_size > runner.scheduler_config.max_num_seqs:
                    continue

                self._warmup_capture_one_drafter_shape(
                    runner, batch_size, K_plus_1, num_input_tokens, desc,
                    fa_blk_tensor_full, cca_blk_tensor_full, _fa_group_id)

    def _warmup_capture_one_drafter_shape(
        self, runner, batch_size: int, K_plus_1: int,
        num_input_tokens: int, drafter_desc,
        fa_blk_tensor_full, cca_blk_tensor_full, fa_group_id: int = 0):
        """Capture one drafter-pass FULL cudagraph at warmup."""
        from vllm.forward_context import set_forward_context
        from vllm.config import CUDAGraphMode

        # Zero-fill input_ids / positions in the runner buffers. The
        # captured graph reads from these buffer ADDRESSES at replay;
        # the runtime propose() will copy real drafter inputs into the
        # same buffers, so zero contents at capture time are fine.
        runner.input_ids.gpu[:num_input_tokens].fill_(0)
        runner.positions.gpu[:num_input_tokens].fill_(0)

        n_plus = batch_size + 1
        gpu_dev = runner.query_start_loc.gpu.device

        # query_start_loc: [0, K+1, 2*(K+1), ..., batch*(K+1)].
        cu_gpu = torch.arange(
            0, n_plus * K_plus_1, K_plus_1, dtype=torch.int32,
            device=gpu_dev)
        runner.query_start_loc.gpu[:n_plus].copy_(cu_gpu)
        runner.query_start_loc.gpu[n_plus:].fill_(num_input_tokens)
        query_start_loc = runner.query_start_loc.gpu[:n_plus]
        query_start_loc_cpu = torch.arange(
            0, n_plus * K_plus_1, K_plus_1, dtype=torch.int32)

        # seq_lens: fill with `max_model_len` so the captured FA
        # kernel's `max_seqlen_k` (which is a Python int read from
        # attn_metadata.max_seq_len at capture time, NOT a tensor read
        # live at replay) is large enough to cover any runtime seq
        # length. The CONTENT of the seq_lens.gpu buffer is read live
        # from the pinned address each step, so runtime propose() can
        # overwrite with the actual per-step values; only the int
        # baked into the kernel call is locked at capture.
        max_seq_int = self.max_model_len
        runner.seq_lens.gpu[:batch_size].fill_(max_seq_int)
        runner.seq_lens.gpu[batch_size:].fill_(0)
        seq_lens = runner.seq_lens.gpu[:batch_size]
        seq_lens_cpu = torch.full(
            (batch_size,), max_seq_int, dtype=torch.int32)

        # FlashAttn slot_mapping: persistent buffer; fill with valid
        # slot IDs (slot 0..num_input_tokens-1 are fine for dummy).
        # (Use the same dynamically-looked-up FA group id as above.)
        fa_blk_obj = runner.input_batch.block_table[fa_group_id]
        fa_blk_obj.slot_mapping.gpu[:num_input_tokens].copy_(
            torch.arange(num_input_tokens, dtype=torch.int64,
                         device=gpu_dev))
        fa_blk_obj.slot_mapping.gpu[num_input_tokens:].fill_(-1)
        slot_mapping_pinned = fa_blk_obj.slot_mapping.gpu[:num_input_tokens]

        # Build the drafter CommonAttentionMetadata pointing at the
        # pinned buffers above. block_table_tensor is the FA group's
        # block table (what attn_metadata_builder.build_for_drafting
        # and FA kernel consume); CCA's draft slot is plumbed
        # separately via the override below.
        fa_block_table_tensor = fa_blk_tensor_full[:batch_size]
        draft_common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=seq_lens,
            _seq_lens_cpu=seq_lens_cpu,
            _num_computed_tokens_cpu=seq_lens_cpu,
            num_reqs=batch_size,
            num_actual_tokens=num_input_tokens,
            max_query_len=K_plus_1,
            max_seq_len=max_seq_int,
            block_table_tensor=fa_block_table_tensor,
            slot_mapping=slot_mapping_pinned,
            causal=False,
        )

        # CCA-side drafter overrides: read=AR (col 0 of CCA's block
        # table), write=draft (col 1), cca_drafter_pass=True for the
        # skip_writes path.
        cca_block_table_tensor = cca_blk_tensor_full[:batch_size]
        ar_slots = cca_block_table_tensor[:, 0]
        draft_slots = cca_block_table_tensor[:, 1]
        setattr(draft_common_attn_metadata,
                "state_indices_tensor_override", ar_slots)
        setattr(draft_common_attn_metadata,
                "state_indices_tensor_write_override", draft_slots)
        setattr(draft_common_attn_metadata, "cca_drafter_pass", True)

        # Build per-layer attention metadata: FA layers via the
        # standard build_for_drafting, CCA layers via the CCA builder
        # (which honors the drafter overrides). (We're called from
        # gpu_model_runner.capture_model which wraps us in
        # inference_mode at the top.)
        attn_metadata = self.attn_metadata_builder.build_for_drafting(
            common_attn_metadata=draft_common_attn_metadata,
            draft_index=0,
        )
        per_layer_attn_metadata = {
            layer_name: attn_metadata
            for layer_name in self.attn_layer_names
        }
        cca_attn_metadata = self.cca_metadata_builder.build(
            common_prefix_len=0,
            common_attn_metadata=draft_common_attn_metadata,
            fast_build=True,
        )
        for layer_name in self.cca_layer_names:
            per_layer_attn_metadata[layer_name] = cca_attn_metadata

        # Capture: call the wrapped model under the drafter
        # BatchDescriptor's forward context. The wrapper finds no
        # entry for this descriptor and captures on the current
        # (non-default) stream — we're inside capture_model's
        # graph_capture(device=) context, which already set the stream.
        logger.info(
            "TiDAR Tier 3: warmup-capturing drafter graph at num_tokens=%d "
            "(batch_size=%d, K+1=%d)",
            num_input_tokens, batch_size, K_plus_1)
        # See propose() comment: pass slot_mapping dict so that
        # FA's unified_kv_cache_update writes drafter K/V to scratch
        # slots during warmup capture.
        _drafter_slot_map_w = {
            _ln: slot_mapping_pinned
            for _ln in self.attn_layer_names
        }
        with set_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                cudagraph_runtime_mode=CUDAGraphMode.FULL,
                batch_descriptor=drafter_desc,
                slot_mapping=_drafter_slot_map_w):
            _ = self.model(
                input_ids=runner.input_ids.gpu[:num_input_tokens],
                positions=runner.positions.gpu[:num_input_tokens],
                inputs_embeds=None,
            )
        self._drafter_captured_sizes.add(num_input_tokens)

    def _resolve_mask_token_id(self) -> Optional[int]:
        hf_config = self.vllm_config.model_config.hf_config
        candidate_attrs = (
            "tidar_mask_token_id",
            "parallel_drafting_mask_token_id",
            "draft_mask_token_id",
            "bidirectional_mask_token_id",
            "mask_token_id",
        )
        for attr in candidate_attrs:
            value = getattr(hf_config, attr, None)
            if value is not None:
                return int(value)
        logger.warning(
            "TiDAR mask token id is not set on the checkpoint config; "
            "falling back to hardcoded token id %d for drafting.",
            DEFAULT_TIDAR_MASK_TOKEN_ID,
        )
        return DEFAULT_TIDAR_MASK_TOKEN_ID

    def _dspark_active(self) -> bool:
        """DSpark draft head auto-enables when the target model carries the
        weights (config flag -> smoe.py registers them). Kill switch:
        VLLM_TIDAR_DSPARK=0 falls back to the AR-head drafter."""
        return (getattr(self.model, "dspark_markov_enabled", False)
                and os.environ.get("VLLM_TIDAR_DSPARK", "1") != "0")

    def _dspark_sample_drafts(self, hidden: torch.Tensor,
                              batch_size: int,
                              mask_positions: Optional[torch.Tensor] = None,
                              prev_token: Optional[torch.Tensor] = None,
                              ) -> torch.Tensor:
        """DSpark draft: untied head + sequential Markov logit bias.

        Per-position math (see dspark handoff doc):
            U_k    = h_k @ W_d.T                (parallel, one matmul)
            B_k    = w1[prev] @ w2              (sequential, cheap)
            q_k    = softmax((U_k + B_k) / T)
            x_k    ~ q_k                        (sample, not argmax)
        `prev` resets to the neutral id 0 at every block boundary
        (k % block_len == 0) and is chained through the sampled ids
        within a block. It is deliberately NOT seeded from the bonus /
        verified token -- the head was trained with the reset, and the
        cross-block dependency is carried by the backbone attention.

        Args:
            hidden:     [B*K, hidden] mask-position hidden states,
                        per-request contiguous.
            batch_size: B.

        Returns:
            [B*K] flat draft token ids (caller reshapes). Stashes
            last_draft_probs (exact q, fp32) and last_draft_logits
            (biased logits) exactly like the AR-head path, so the
            rejection sampler computes the exact accept ratio
            min(1, p/q).
        """
        model = self.model
        W_d = model.diffusion_output_layer.weight    # [V, H]
        w1 = model.diffusion_markov_head.w1          # [V, R]
        w2 = model.diffusion_markov_head.w2          # [R, V]
        block_len = int(getattr(model, "dspark_block_len", 16))
        K = self.num_speculative_tokens
        B = batch_size
        assert hidden.shape[0] == B * K, (
            f"dspark expected {B}*{K} rows, got {hidden.shape[0]}")

        U = torch.matmul(hidden, W_d.t()).view(B, K, -1)  # [B, K, V]
        V = U.shape[-1]
        T = self.diff_temperature
        tokens = torch.empty(B, K, dtype=torch.long, device=U.device)
        probs = (None if T == 0.0 else torch.empty(
            B, K, V, dtype=torch.float32, device=U.device))
        # Reset the Markov `prev` on the TRUE global block grid when we know
        # each mask's global position; else fall back to the draft-local grid.
        # At a non-boundary draft position 0, the correct prev is the last
        # committed token (matches training: prev = token at global g-1).
        # VLLM_DSPARK_GLOBAL_RESET=0 forces the local fallback (A/B).
        use_global = (mask_positions is not None and prev_token is not None
                      and os.environ.get("VLLM_DSPARK_GLOBAL_RESET", "1") != "0")
        if use_global:
            prev = prev_token.to(torch.long).clone()
        else:
            prev = torch.zeros(B, dtype=torch.long, device=U.device)
        for k in range(K):
            if use_global:
                at_boundary = (mask_positions[:, k] % block_len == 0)
                prev = torch.where(at_boundary,
                                   torch.zeros_like(prev), prev)
            elif k > 0 and k % block_len == 0:
                prev.zero_()
            bias = torch.matmul(w1.index_select(0, prev), w2)  # [B, V]
            if os.environ.get("VLLM_DSPARK_NO_MARKOV", "0") == "1":
                bias = torch.zeros_like(bias)   # ablation: base head only
            U[:, k].add_(bias)          # keep biased logits for the stash
            logits_k = U[:, k].to(torch.float32)
            if T == 0.0:
                x = logits_k.argmax(dim=-1)
            else:
                q = torch.softmax(logits_k / T, dim=-1)
                x = torch.multinomial(q, num_samples=1).squeeze(-1)
                probs[:, k] = q
            tokens[:, k] = x
            prev = x
        self.last_draft_logits = U.view(B * K, V).detach().contiguous()
        self.last_draft_probs = (None if probs is None else
                                 probs.view(B * K, V).contiguous())
        return tokens.view(-1)

    def load_model(self, target_model: nn.Module) -> None:
        # TiDAR reuses the target checkpoint for drafting and verification.
        self.model = target_model
        self.attn_layer_names = list(
            get_layers_from_vllm_config(self.vllm_config, Attention).keys())
        self.cca_layers = get_layers_from_vllm_config(self.vllm_config, CCA)
        self.cca_layer_names = list(self.cca_layers.keys())
        self.indexer_layer_names = []
        self.draft_indexer_metadata_builder = None
        self.attn_metadata_builder = None
        self.cca_metadata_builder = None

        if not self.attn_layer_names:
            raise RuntimeError(
                "TiDAR requires at least one attention layer in the target "
                "model.")

        logger.info(
            "Initialized TiDAR self-speculation with a two-forward "
            "FlashAttention draft pass.")

    def _get_metadata_builder_for_layer(self, chosen_layer: str):
        builder = None
        for kv_cache_group in self.runner.attn_groups:
            for attn_group in kv_cache_group:
                if chosen_layer in attn_group.layer_names:
                    builder = attn_group.get_metadata_builder()
                    break
            if builder is not None:
                break

        assert builder is not None, (
            f"Failed to find attention metadata builder for {chosen_layer}.")
        return builder

    def _get_metadata_builder_and_group_id_for_layer(self, chosen_layer: str):
        builder = None
        kv_cache_group_id = None
        for group_id, kv_cache_group in enumerate(self.runner.attn_groups):
            for attn_group in kv_cache_group:
                if chosen_layer in attn_group.layer_names:
                    builder = attn_group.get_metadata_builder()
                    kv_cache_group_id = group_id
                    break
            if builder is not None:
                break

        assert builder is not None and kv_cache_group_id is not None, (
            f"Failed to find attention metadata builder for {chosen_layer}.")
        return builder, kv_cache_group_id

    def _get_cca_block_slots(
        self,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (ar_state_indices, draft_state_indices) for CCA layers.

        Used by the drafter under the single-state CCA design: the drafter
        forward reads from the AR block (post-rejection-acceptance state)
        and writes its default-commit to the draft block (scratch — never
        read after this forward, overwritten next step). No AR -> draft
        copy is performed here; the previous _seed_cca_draft_state path
        did that copy across all CCA layers per step (~5-10ms wasted) and
        was rendered unnecessary once the captured CCA forward learned to
        scatter into a separate write slot.

        PERF: ar_slots/draft_slots are slices of the persistent CCA block
        table -- their data_ptr is stable per (num_reqs, group_id). Cache
        the slice views to skip the per-step Python iteration through
        attn_groups + validation cost (~25us saved per propose).
        """
        if self.cca_metadata_builder is None or \
                not hasattr(self, "cca_kv_cache_group_id"):
            (self.cca_metadata_builder,
             self.cca_kv_cache_group_id) = \
                self._get_metadata_builder_and_group_id_for_layer(
                    self.cca_layer_names[0])

        _num_reqs = common_attn_metadata.num_reqs
        _cache_key = (_num_reqs, self.cca_kv_cache_group_id)
        _cached = getattr(self, "_cached_cca_slots_key", None)
        if _cached == _cache_key:
            return self._cached_cca_ar_slots, self._cached_cca_draft_slots
        block_table_obj = self.runner.input_batch.block_table[
            self.cca_kv_cache_group_id]
        block_table = block_table_obj.get_device_tensor(_num_reqs)
        if block_table.shape[1] < 2:
            raise ValueError(
                "TiDAR with CCA requires a dedicated draft state block, but "
                "the current request block tables only expose one state slot.")

        ar_state_indices = block_table[:, 0]
        draft_state_indices = block_table[:, 1]
        # Validate against the CPU mirror so we don't pay a sync just to read
        # back a single bool (`torch.any` on a GPU tensor would force one).
        block_table_cpu = block_table_obj.get_cpu_tensor()[:_num_reqs]
        if (block_table_cpu[:, 0] <= 0).any().item() or \
                (block_table_cpu[:, 1] <= 0).any().item():
            raise ValueError(
                "TiDAR found invalid CCA state indices while preparing the "
                "draft cache.")

        self._cached_cca_slots_key = _cache_key
        self._cached_cca_ar_slots = ar_state_indices
        self._cached_cca_draft_slots = draft_state_indices
        return ar_state_indices, draft_state_indices

    def _build_draft_inputs(
        self,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        last_token_indices: Optional[torch.Tensor],
        common_attn_metadata: CommonAttentionMetadata,
        num_rejected_tokens_gpu: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, CommonAttentionMetadata]:
        batch_size = next_token_ids.shape[0]
        # FIX (drafter alignment): K+1 input tokens [next_token, mask×K]
        # so the K mask positions land at sequence positions P+1..P+K.
        # Under diffuser fill-in semantics, mask at seq P+i predicts
        # AR(P+i). Reading drafts at input positions 1..K thus produces
        # AR(P+1)..AR(P+K), matching the rejection sampler's expected
        # AR-style alignment. Without this, K input tokens (with K-1
        # masks) shift drafts off-by-one from rejection's expectation.
        query_len = self.num_speculative_tokens + 1
        if last_token_indices is None:
            last_token_indices = common_attn_metadata.query_start_loc[1:] - 1

        query_offsets = torch.arange(query_len,
                                     dtype=target_positions.dtype,
                                     device=target_positions.device)
        base_positions = target_positions[last_token_indices] + 1
        draft_positions = (base_positions.view(-1, 1) +
                           query_offsets.view(1, -1))

        exceeds_max_model_len = draft_positions >= self.max_model_len
        clamped_positions = torch.where(exceeds_max_model_len, 0,
                                        draft_positions)
        # v0.16: block_size lives on attn_metadata_builder.kv_cache_spec,
        # not as a cached self.block_size attribute. (eagle.py:624)
        _bs = self.attn_metadata_builder.kv_cache_spec.block_size
        block_numbers = clamped_positions // _bs
        block_ids = common_attn_metadata.block_table_tensor.gather(
            dim=1, index=block_numbers)
        slot_mapping = (block_ids * _bs +
                        clamped_positions % _bs)
        slot_mapping.masked_fill_(exceeds_max_model_len, PAD_SLOT_ID)

        # Layout: position 0 = next_token (real), positions 1..K = K masks.
        draft_input_ids = torch.full(
            (batch_size, query_len),
            fill_value=self.mask_token_id,
            dtype=next_token_ids.dtype,
            device=next_token_ids.device,
        )
        draft_input_ids[:, 0] = next_token_ids

        # ----------------------------------------------------------------
        # Pin FlashAttn-side metadata into the runner's persistent buffers
        # so the captured FULL graph (bound to runner buffer addresses
        # at warmup) reads drafter values at replay. CCA-side metadata
        # (has_initial_states_p, query_start_loc_p) is pinned by the CCA
        # metadata builder; see CCAAttentionMetadataBuilder.__init__.
        # ----------------------------------------------------------------
        runner = self.runner
        n_plus = batch_size + 1
        n_tokens = batch_size * query_len
        gpu_dev = runner.query_start_loc.gpu.device

        # query_start_loc: cumulative [0, q, 2q, ...] for batch_size reqs.
        cu_gpu = torch.arange(0, n_plus * query_len, query_len,
                              dtype=torch.int32, device=gpu_dev)
        runner.query_start_loc.gpu[:n_plus].copy_(cu_gpu)
        # FlashAttention requires non-decreasing; pad rest with last value.
        runner.query_start_loc.gpu[n_plus:].fill_(n_tokens)
        query_start_loc = runner.query_start_loc.gpu[:n_plus]
        # CPU mirror — small, rebuilt cheaply.
        query_start_loc_cpu = torch.arange(0, n_plus * query_len,
                                           query_len, dtype=torch.int32)

        # seq_lens: start from the live post-rejection sequence, not the
        # verifier's full K+1 window. Otherwise the draft masks can attend to
        # rejected suffix KV that was written by the verifier pass.
        base_seq_lens = common_attn_metadata.seq_lens
        base_seq_lens_cpu = common_attn_metadata.seq_lens_cpu
        if num_rejected_tokens_gpu is not None:
            rejected = num_rejected_tokens_gpu.to(
                device=base_seq_lens.device,
                dtype=base_seq_lens.dtype,
                non_blocking=True,
            )
            base_seq_lens = (base_seq_lens - rejected).clamp_min(0)
            rejected_cpu = num_rejected_tokens_gpu.to(
                device="cpu",
                dtype=base_seq_lens_cpu.dtype,
            )
            base_seq_lens_cpu = (
                base_seq_lens_cpu - rejected_cpu).clamp_min(0)

        new_seq = torch.clamp(base_seq_lens + query_len,
                              max=self.max_model_len).to(torch.int32)
        runner.seq_lens.gpu[:batch_size].copy_(new_seq)
        runner.seq_lens.gpu[batch_size:].fill_(0)
        seq_lens = runner.seq_lens.gpu[:batch_size]
        seq_lens_cpu = torch.clamp(base_seq_lens_cpu + query_len,
                                   max=self.max_model_len)

        # slot_mapping: write into FlashAttn group's persistent slot_mapping.
        # Use FA group id dynamically — hardcoded [0] picked the CCA group
        # on this model (CCA layers happen to register first), so the
        # captured drafter graph (which traced FA's slot_mapping at fa_group's
        # persistent buffer) read stale data at replay.
        # Cache the lookup; attn_groups is static post-init.
        if not hasattr(self, "_cached_bdi_fa_group_id"):
            _, self._cached_bdi_fa_group_id = (
                self._get_metadata_builder_and_group_id_for_layer(
                    self.attn_layer_names[0]))
        _fa_group_id_bdi = self._cached_bdi_fa_group_id
        flash_blk_table = runner.input_batch.block_table[_fa_group_id_bdi]
        flash_blk_table.slot_mapping.gpu[:n_tokens].copy_(slot_mapping.view(-1))
        flash_blk_table.slot_mapping.gpu[n_tokens:].fill_(-1)
        slot_mapping_pinned = flash_blk_table.slot_mapping.gpu[:n_tokens]

        max_seq_len = int(seq_lens_cpu.max().item())

        draft_common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=seq_lens,
            # v0.16: seq_lens_cpu is a @property; private field is
            # _seq_lens_cpu. Pass through if available.
            _seq_lens_cpu=seq_lens_cpu,
            _num_computed_tokens_cpu=base_seq_lens_cpu,
            num_reqs=batch_size,
            num_actual_tokens=n_tokens,
            max_query_len=query_len,
            max_seq_len=max_seq_len,
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=slot_mapping_pinned,
            causal=False,
        )
        return (draft_input_ids.view(-1), draft_positions.view(-1),
                draft_common_attn_metadata)

    def _coordinate_draft_forward(
        self,
        local_num_tokens: int,
    ) -> tuple[bool, Optional[torch.Tensor], int]:
        # Coordinate the nested TiDAR draft forward across DP ranks so its
        # SMoE/EP all-to-all stays lockstep. No-op unless DP>1 + MoE.
        # Ranks without real draft work advertise a small dummy draft only
        # when at least one peer has real work.
        parallel_config = self.vllm_config.parallel_config
        if (parallel_config.data_parallel_size <= 1
                or parallel_config.is_moe_model is False):
            return local_num_tokens > 0, None, local_num_tokens
        # SINGLE-BARRIER FOLD: the draft real-count already rode the outer
        # coordinate barrier (gpu_model_runner._store_tidar_draft_fold).
        # Read the stored result -- NO separate all_reduce here (that
        # separate collective was the off-by-one source). Lockstep is
        # inherited from the outer barrier (exactly once per step).
        del local_num_tokens
        runner = self.runner
        runner._tidar_coordinate_ran = True
        should_run = getattr(runner, '_tidar_draft_should_run', False)
        eff_across_dp = getattr(runner, '_tidar_draft_eff_across_dp', None)
        if not should_run or eff_across_dp is None:
            return False, eff_across_dp, 0
        eff = int(eff_across_dp[parallel_config.data_parallel_rank].item())
        return True, eff_across_dp, eff

    def propose(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        # v0.16 EagleProposer renamed last_token_indices ->
        # token_indices_to_sample; accept the new name and keep
        # the old as a compat alias.
        token_indices_to_sample: Optional[torch.Tensor] = None,
        common_attn_metadata: Optional[CommonAttentionMetadata] = None,
        sampling_metadata: Optional[SamplingMetadata] = None,
        # v0.16 renamed mm_embeds -> mm_embed_inputs
        mm_embed_inputs: Optional[list[torch.Tensor]] = None,
        # v0.16 added; TiDAR's SF eager path doesn't use these.
        num_rejected_tokens_gpu: Optional[torch.Tensor] = None,
        slot_mappings: Optional[dict[str, torch.Tensor]] = None,
        # v0.15 compat aliases
        last_token_indices: Optional[torch.Tensor] = None,
        mm_embeds: Optional[list[torch.Tensor]] = None,
    ) -> torch.Tensor:
        # Resolve v0.15/v0.16 arg name aliases.
        if last_token_indices is None:
            last_token_indices = token_indices_to_sample
        if mm_embeds is None:
            mm_embeds = mm_embed_inputs
        del target_token_ids, target_hidden_states, sampling_metadata
        del slot_mappings

        if mm_embeds is not None:
            raise NotImplementedError(
                "The TiDAR FlashAttention prototype does not support "
                "multimodal draft inputs.")

        if self.attn_metadata_builder is None:
            self.attn_metadata_builder = self._get_metadata_builder_for_layer(
                self.attn_layer_names[0])

        draft_input_ids, draft_positions, draft_common_attn_metadata = \
            self._build_draft_inputs(
                next_token_ids=next_token_ids,
                target_positions=target_positions,
                last_token_indices=last_token_indices,
                common_attn_metadata=common_attn_metadata,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            )

        # DSpark: capture the K mask tokens' TRUE global positions and the
        # committed token preceding each block, for the Markov reset. The
        # block grid follows global position (% block_len == 0), not the
        # draft-local index -- TiDAR commits a variable number of tokens per
        # step, so the mask window rarely starts on a block boundary.
        # draft_input_ids/draft_positions are flat [B*(K+1)] in
        # [next_token, mask x K] layout; mask k = slot k+1.
        self._dspark_mask_positions = None
        self._dspark_prev_token = None
        if self._dspark_active():
            _kp1 = self.num_speculative_tokens + 1
            self._dspark_mask_positions = (
                draft_positions.view(-1, _kp1)[:, 1:_kp1].contiguous())
            self._dspark_prev_token = (
                draft_input_ids.view(-1, _kp1)[:, 0].contiguous())

        attn_metadata = self.attn_metadata_builder.build_for_drafting(
            common_attn_metadata=draft_common_attn_metadata,
            draft_index=0,
        )
        # Original prototype required flash_attn; for single-forward
        # bootstrap (first decode step before any spec_decode tokens
        # exist), the runner falls back to propose() and this assertion
        # would fire under FlexAttention. The rest of propose() is
        # mostly backend-agnostic (input_ids/positions/CCA paths +
        # model.forward); model.forward dispatches to whatever backend
        # is configured. Allow FlexAttentionMetadata to flow through.
        from vllm.v1.attention.backends.flex_attention import (
            FlexAttentionMetadata)
        _allowed = [FlashAttentionMetadata, FlexAttentionMetadata]
        try:
            from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata
            _allowed.append(TritonAttentionMetadata)
        except ImportError:
            pass
        try:
            from vllm.v1.attention.backends.rocm_aiter_fa import AiterFlashAttentionMetadata
            _allowed.append(AiterFlashAttentionMetadata)
        except ImportError:
            pass
        try:
            from vllm.v1.attention.backends.rocm_attn import RocmAttentionMetadata
            _allowed.append(RocmAttentionMetadata)
        except ImportError:
            pass
        if not isinstance(attn_metadata, tuple(_allowed)):
            raise ValueError(
                "TiDAR drafter: unsupported attention metadata "
                f"{type(attn_metadata).__name__}.")

        per_layer_attn_metadata = {
            layer_name: attn_metadata
            for layer_name in self.attn_layer_names
        }
        if self.cca_layer_names:
            # Single-state CCA design: drafter READS from the AR block
            # (post-rejection-acceptance state, exactly what we want as
            # the drafter's starting state) and WRITES its default-commit
            # to the draft block (treated as scratch — never read after
            # this forward). This kills the per-step _seed_cca_draft_state
            # AR -> draft copy that previously synced the two blocks
            # before each draft forward (~5-10ms/step at 22 CCA layers).
            ar_slots, draft_slots = self._get_cca_block_slots(
                common_attn_metadata)
            setattr(draft_common_attn_metadata,
                    "state_indices_tensor_override", ar_slots)
            setattr(draft_common_attn_metadata,
                    "state_indices_tensor_write_override", draft_slots)
            # Mark as a drafter forward so the captured CCA path skips its
            # default-commit + stash writes (both pure scratch under path 2:
            # nobody reads the draft slot after this forward, and the stash
            # is consumed only post-verify by commit_spec_decode_state).
            # The drafter's separately captured graph (Tier 3) bakes the
            # no-write path; the verify graph keeps the do-write path.
            setattr(draft_common_attn_metadata, "cca_drafter_pass", True)
            cca_attn_metadata = self.cca_metadata_builder.build(
                common_prefix_len=0,
                common_attn_metadata=draft_common_attn_metadata,
                fast_build=True,
            )
            for layer_name in self.cca_layer_names:
                per_layer_attn_metadata[layer_name] = cca_attn_metadata

        num_tokens = draft_input_ids.shape[0]
        # v0.16 removed self.use_cuda_graph / self.cudagraph_batch_sizes
        # on EagleProposer; use getattr with safe defaults so the eager
        # path keeps working.
        _ucg = getattr(self, "use_cuda_graph", False)
        _cgbs = getattr(self, "cudagraph_batch_sizes", None)
        if _ucg and _cgbs and num_tokens <= _cgbs[-1]:
            num_input_tokens = self.vllm_config.pad_for_cudagraph(num_tokens)
        else:
            num_input_tokens = num_tokens

        # DP: coordinate this draft forward with peer ranks (no-op unless
        # DP>1 + MoE). Provides per-rank token counts for
        # set_forward_context so the draft SMoE/EP all-to-all stays
        # lockstep; zero-work ranks match via dummy_run().
        (_draft_should_run, _draft_num_tokens_across_dp,
         num_input_tokens) = self._coordinate_draft_forward(num_input_tokens)
        if not _draft_should_run:
            self.last_draft_probs = None
            return torch.empty(
                (0, self.num_speculative_tokens),
                dtype=torch.int32, device=next_token_ids.device)

        # Pin input_ids/positions into the runner's persistent buffers so
        # the captured FULL graph (bound to runner addresses) reads
        # drafter data at replay. Both CCA and FlashAttn metadata are
        # already pinned (see _build_draft_inputs and
        # CCAAttentionMetadataBuilder.__init__).
        runner = self.runner
        runner.input_ids.gpu[:num_tokens].copy_(
            draft_input_ids.to(runner.input_ids.gpu.dtype))
        runner.positions.gpu[:num_tokens].copy_(
            draft_positions.to(runner.positions.gpu.dtype))
        if num_input_tokens > num_tokens:
            runner.input_ids.gpu[num_tokens:num_input_tokens].fill_(0)
            runner.positions.gpu[num_tokens:num_input_tokens].fill_(0)

        # Dispatch the drafter forward through the runner's captured
        # FULL graph. Tier 3: dispatch with is_drafter_pass=True so the
        # wrapper picks the drafter-specific CUDAGraphEntry whose
        # gather/scatter operand pointers were baked at the drafter's
        # own capture — path 2's read=AR / write=draft routing only
        # reaches the captured kernels because the drafter graph is
        # captured lazily at the *first* propose() call, when this
        # function's metadata setup (state_indices_tensor_override =
        # ar_slots, state_indices_tensor_write_override = draft_slots)
        # is the metadata in scope. The verify graph captured during
        # standard warmup baked write→AR; sharing that graph would
        # corrupt AR. If TiDAR isn't active or the dispatcher hasn't
        # registered this descriptor, dispatch returns NONE and the
        # wrapper passes through to eager (no perf gain, no breakage).
        from vllm.forward_context import BatchDescriptor as _BD
        # v0.16 dispatcher: takes individual kwargs (num_tokens,
        # uniform_decode), not a BatchDescriptor. Returns
        # (CUDAGraphMode, BatchDescriptor).
        _cg_mode, _draft_desc = runner.cudagraph_dispatcher.dispatch(
            num_tokens=num_input_tokens,
            uniform_decode=True,
            is_drafter_pass=True,
        )
        # DP+EP: a FULL captured drafter graph bakes its EP all-to-all
        # routing at warmup (uniform composition). Under concurrent
        # NON-uniform DP composition the replayed draft all-to-all
        # mismatches peer/idle ranks and the post-step draft sync
        # (_get_draft_token_ids_cpu) deadlocks (captured DP + >=8 conc).
        # Force the draft eager so every rank's draft all-to-all is
        # runtime-routed and mutually consistent. Main forward stays
        # captured. Escape hatch: VLLM_TIDAR_DP_EAGER_DRAFT=0.
        _pc = self.vllm_config.parallel_config
        if (os.environ.get("VLLM_TIDAR_DP_EAGER_DRAFT", "0") == "1"
                and _pc.data_parallel_size > 1
                and _pc.is_moe_model is not False):
            _cg_mode = CUDAGraphMode.NONE

        # Tier 3 lazy capture: the first time we hit this shape, the
        # CUDAGraphWrapper has no entry for the drafter-pass descriptor
        # and tries to capture. The global cudagraph_capturing flag is
        # disabled after the standard warmup (see capture_model in the
        # runner), so without this bracket the capture would raise. We
        # re-enable for the duration of the first call, capture
        # succeeds, the entry is cached, and subsequent calls replay
        # without entering the capture branch (no validate is invoked
        # on replay). We track captured shapes locally so we don't keep
        # re-enabling the global flag on every step.
        _needs_capture = (_cg_mode != CUDAGraphMode.NONE
                          and num_input_tokens
                          not in self._drafter_captured_sizes)
        if _needs_capture:
            set_cudagraph_capturing_enabled(True)
        # v0.16 split unified_kv_cache_update off from FA's forward
        # (FlashAttentionBackend.forward_includes_kv_cache_update=False).
        # That op reads slot_mapping from forward_context.slot_mapping
        # (a {layer_name: tensor} dict). Without this dict the op
        # silently no-ops, the drafter's K/V never lands in the cache,
        # and FA reads stale slots -> 0% accept. FLEX backend has
        # forward_includes_kv_cache_update=True (default) so it
        # ignores this dict -- safe to populate unconditionally.
        # CCA layers don't go through unified_kv_cache_update (CCA
        # manages its own conv_states/prev_hs writes inside its
        # forward), so we only need entries for the FA attn layers.
        _drafter_slot_map = {
            _ln: draft_common_attn_metadata.slot_mapping
            for _ln in self.attn_layer_names
        }
        if os.environ.get("VLLM_TIDAR_DBG") == "1":
            import sys as _sys
            _r = self.vllm_config.parallel_config.data_parallel_rank
            _v = (_draft_num_tokens_across_dp.tolist()
                  if _draft_num_tokens_across_dp is not None else None)
            print(f"[TIDAR_DBG r{_r}] draft actual_num_tokens={num_tokens} "
                  f"eff_num_input_tokens={num_input_tokens} cg_mode={_cg_mode} "
                  f"ntad={_v}", flush=True, file=_sys.stderr)
        try:
            with set_forward_context(per_layer_attn_metadata,
                                     self.vllm_config,
                                     num_tokens=num_input_tokens,
                                     num_tokens_across_dp=_draft_num_tokens_across_dp,
                                     cudagraph_runtime_mode=_cg_mode,
                                     batch_descriptor=_draft_desc,
                                     slot_mapping=_drafter_slot_map):
                model_output = self.model(
                    input_ids=runner.input_ids.gpu[:num_input_tokens],
                    positions=runner.positions.gpu[:num_input_tokens],
                    inputs_embeds=None,
                )
        finally:
            if _needs_capture:
                set_cudagraph_capturing_enabled(False)
                self._drafter_captured_sizes.add(num_input_tokens)

        if isinstance(model_output, tuple):
            draft_hidden_states = model_output[0]
        else:
            draft_hidden_states = model_output
        draft_hidden_states = draft_hidden_states[:num_tokens]

        # FIX (drafter alignment): we ran K+1 inputs [next_token, mask×K]
        # per req. Drop position 0 (the non-mask next_token slot) and keep
        # the K mask positions' hidden states for logits.
        K = self.num_speculative_tokens
        batch_size_local = num_tokens // (K + 1)
        assert batch_size_local * (K + 1) == num_tokens, (
            f"drafter expected num_tokens=batch*(K+1); got num_tokens={num_tokens} K={K}")
        draft_hidden_states = draft_hidden_states.view(
            batch_size_local, K + 1, -1)[:, 1:K + 1, :].contiguous().view(
            batch_size_local * K, -1)

        if self._dspark_active():
            # DSpark: untied draft head + Markov bias, sequential within
            # the block. Stashes last_draft_probs/logits internally.
            draft_token_ids = self._dspark_sample_drafts(
                draft_hidden_states, batch_size_local,
                mask_positions=self._dspark_mask_positions,
                prev_token=self._dspark_prev_token)
            return draft_token_ids.view(
                -1, self.num_speculative_tokens).to(torch.int32)

        logits = self.model.compute_logits(draft_hidden_states)
        if logits is None:
            raise RuntimeError("TiDAR target model did not return logits.")

        # Stash raw logits for mix-logit v1 (parallel to last_draft_probs).
        self.last_draft_logits = logits.detach().contiguous()
        if self.diff_temperature == 0.0:
            # Dirac drafter (T_Diff = 0): keep the original argmax path and
            # signal "no draft distribution" downstream so the rejection
            # sampler stays on its NO_DRAFT_PROBS branch.
            self.last_draft_probs = None
            draft_token_ids = logits.argmax(dim=-1)
        else:
            # Single source of truth for the temperature scaling: scale once,
            # then both the categorical sample and the stored probs come from
            # the same distribution. Float32 for the rejection ratio.
            scaled_logits = logits.to(torch.float32) / self.diff_temperature
            draft_probs = torch.softmax(scaled_logits, dim=-1)
            draft_token_ids = torch.multinomial(
                draft_probs, num_samples=1).squeeze(-1)
            self.last_draft_probs = draft_probs.contiguous()

        return draft_token_ids.view(-1,
                                    self.num_speculative_tokens).to(torch.int32)

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings: dict[str, torch.Tensor] | None = None,
    ) -> None:
        # At profile/capture time TiDAR reuses the target forward (no
        # draft warmup needed). But under DP, a zero-work rank must run a
        # matching dummy DRAFT forward so the draft pass's SMoE/EP
        # collectives stay lockstep with active ranks (real draft in
        # propose()).
        del num_tokens, use_cudagraphs, is_graph_capturing, slot_mappings
        parallel_config = self.vllm_config.parallel_config
        if (parallel_config.data_parallel_size <= 1
                or parallel_config.is_moe_model is False):
            return
        _should_run, _ntad, _eff = self._coordinate_draft_forward(0)
        if not _should_run:
            return
        self.runner._dummy_run(
            _eff,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            force_attention=True,
            uniform_decode=True,
            allow_microbatching=False,
            skip_drafter_dummy=True,
            num_tokens_across_dp_override=_ntad,
        )

    def validate_same_kv_cache_group(self,
                                     kv_cache_config: KVCacheConfig) -> None:
        super().validate_same_kv_cache_group(kv_cache_config)

