# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import graph_capture
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.mamba.cca import CCA
from vllm.v1.attention.backend import (
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.utils.dspark import (
    enforce_dspark_target_contract,
    validate_tidar_temperature_contract,
)

logger = init_logger(__name__)

DEFAULT_TIDAR_MASK_TOKEN_ID = 4


class TiDARSpeculator:
    """V2 two-forward TiDAR self-draft path.

    The target/verify pass is handled by the normal V2 spec-decode runner.
    This speculator runs the second forward: [accepted_token, mask x K] with
    bidirectional attention inside the draft block, then returns K draft IDs.
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.vllm_config = vllm_config
        self.device = device

        self.speculative_config = vllm_config.speculative_config
        assert self.speculative_config is not None
        self.num_speculative_steps = self.speculative_config.num_speculative_tokens
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.max_model_len = vllm_config.model_config.max_model_len
        self.diff_temperature = float(
            self.speculative_config.tidar_diff_temperature)
        self.ar_temperature = float(
            self.speculative_config.tidar_ar_temperature)
        if os.environ.get(
            "VLLM_TIDAR_ASSERT_AR_LOGPROBS", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}:
            configured_ar_temperature = os.environ.get(
                "VLLM_TIDAR_AR_TEMPERATURE"
            )
            validate_tidar_temperature_contract(
                draft_temperature=self.diff_temperature,
                ar_temperature=self.ar_temperature,
                configured_ar_temperature=configured_ar_temperature,
            )
            logger.info(
                "Validated strict TiDAR temperature contract: "
                "draft=%s, AR verifier=%s, returned scores=%s.",
                self.diff_temperature,
                self.ar_temperature,
                configured_ar_temperature,
            )
        self.mask_token_id = self._resolve_mask_token_id()

        self.input_buffers = InputBuffers(
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            device=device,
        )
        self.draft_tokens = torch.zeros(
            self.max_num_reqs,
            self.num_speculative_steps,
            dtype=torch.int64,
            device=device,
        )
        self.slot_mappings: torch.Tensor | None = None
        self.draft_cudagraph_sizes = self._get_draft_cudagraph_sizes()
        self.draft_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.draft_graph_pool = torch.cuda.graph_pool_handle()
        self.draft_hidden_states: torch.Tensor | None = None
        self.last_draft_hidden_states: torch.Tensor | None = None
        self.last_draft_logits: torch.Tensor | None = None
        self.last_draft_probs: torch.Tensor | None = None
        self.last_draft_token_probs: torch.Tensor | None = None
        self.last_draft_logsumexp: torch.Tensor | None = None

    def _get_draft_cudagraph_sizes(self) -> set[int]:
        if os.environ.get("VLLM_TIDAR_V2_DISABLE_DRAFT_CUDAGRAPH") == "1":
            logger.info(
                "TiDAR drafter CUDA graph capture disabled via "
                "VLLM_TIDAR_V2_DISABLE_DRAFT_CUDAGRAPH=1.")
            return set()
        compilation_config = self.vllm_config.compilation_config
        if (compilation_config is None
                or not compilation_config.cudagraph_mode.has_full_cudagraphs()):
            return set()
        capture_sizes = compilation_config.cudagraph_capture_sizes
        if not capture_sizes:
            return set()
        query_len = self.num_speculative_steps + 1
        sizes: set[int] = set()
        for size in capture_sizes:
            if size <= 0 or size > self.max_num_tokens:
                continue
            if size % query_len != 0:
                continue
            num_reqs = size // query_len
            if 0 < num_reqs <= self.max_num_reqs:
                sizes.add(int(size))
        return sizes

    def _resolve_mask_token_id(self) -> int:
        hf_config = self.vllm_config.model_config.hf_config
        for attr in (
            "tidar_mask_token_id",
            "parallel_drafting_mask_token_id",
            "draft_mask_token_id",
            "bidirectional_mask_token_id",
            "diffusion_mask_token_id",
            "mask_token_id",
        ):
            value = getattr(hf_config, attr, None)
            if value is not None:
                return int(value)
        logger.warning(
            "TiDAR mask token id is not set on the checkpoint config; "
            "falling back to hardcoded token id %d for drafting.",
            DEFAULT_TIDAR_MASK_TOKEN_ID)
        return DEFAULT_TIDAR_MASK_TOKEN_ID

    def load_model(self, target_model: nn.Module) -> None:
        self.model = target_model
        require_ar_verifier_contract = os.environ.get(
            "VLLM_TIDAR_REQUIRE_AR_VERIFIER_CONTRACT", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        enforce_dspark_target_contract(
            target_model,
            retained_target_model=self.model,
            dspark_active=self._dspark_active(),
            required=require_ar_verifier_contract,
        )
        if require_ar_verifier_contract:
            logger.info(
                "Validated TiDAR AR-verifier contract: shared target "
                "backbone, regular tied lm_head, and distinct frozen "
                "DSpark/Markov heads."
            )
        self.attn_layer_names = list(
            get_layers_from_vllm_config(
                self.vllm_config, Attention).keys())
        self.cca_layers = get_layers_from_vllm_config(self.vllm_config, CCA)
        self.cca_layer_names = list(self.cca_layers.keys())
        if not self.attn_layer_names:
            raise RuntimeError(
                "TiDAR requires at least one attention layer in the target model.")
        logger.info(
            "Initialized V2 TiDAR two-forward self-speculator%s.",
            " with DSpark+Markov drafting" if self._dspark_active() else "",
        )

    def _dspark_active(self) -> bool:
        return (
            getattr(self.model, "dspark_markov_enabled", False)
            and os.environ.get("VLLM_TIDAR_DSPARK", "1") != "0"
        )

    def _sample_dspark_stochastic(
        self,
        logits: torch.Tensor,
        sample_position: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample DSpark logits without constructing their probability matrix.

        Gumbel-max samples exactly from softmax(logits / T). The compact
        selected-token probability and log-normalizer are all the rejection
        sampler needs: q(x) accepts a draft token, and logits plus logsumexp
        reconstruct q(v) only if that position needs residual recovery.
        """
        batch_size = logits.shape[0]
        device = logits.device
        row_indices = torch.arange(batch_size, dtype=torch.int64, device=device)
        temperatures = torch.full(
            (batch_size,), self.diff_temperature, dtype=torch.float32, device=device
        )
        # TiDAR's historical torch.multinomial path used the global CUDA RNG.
        # Keep that contract while letting the fused gumbel kernel generate one
        # independent categorical draw per row.
        seeds = torch.randint(
            torch.iinfo(torch.int32).max,
            (batch_size,),
            dtype=torch.int64,
            device=device,
        )
        positions = torch.full(
            (batch_size,), sample_position, dtype=torch.int64, device=device
        )
        sampled = gumbel_sample(
            logits,
            row_indices,
            temperatures,
            seeds,
            positions,
            apply_temperature=True,
        )
        scaled_logits = logits.to(torch.float32).div(self.diff_temperature)
        logsumexp = torch.logsumexp(scaled_logits, dim=-1)
        selected_logits = scaled_logits.gather(1, sampled.unsqueeze(1)).squeeze(1)
        selected_probs = torch.exp(selected_logits - logsumexp)
        return sampled, selected_probs, logsumexp

    def _dspark_sample_drafts(
        self,
        hidden_states: torch.Tensor,
        batch_size: int,
        mask_positions: torch.Tensor,
        prev_token: torch.Tensor,
    ) -> torch.Tensor:
        model = self.model
        draft_weight = model.diffusion_output_layer.weight
        markov_w1 = model.diffusion_markov_head.w1
        markov_w2 = model.diffusion_markov_head.w2
        block_len = int(getattr(model, "dspark_block_len", 16))
        K = self.num_speculative_steps
        assert hidden_states.shape[0] == batch_size * K

        base_logits = torch.matmul(hidden_states, draft_weight.t()).view(
            batch_size, K, -1
        )
        tokens = torch.empty(
            batch_size, K, dtype=torch.long, device=base_logits.device
        )
        token_probs = None
        logsumexp = None
        if self.diff_temperature != 0.0:
            token_probs = torch.empty(
                batch_size,
                K,
                dtype=torch.float32,
                device=base_logits.device,
            )
            logsumexp = torch.empty_like(token_probs)

        use_global_reset = os.environ.get("VLLM_DSPARK_GLOBAL_RESET", "1") != "0"
        prev = (
            prev_token.to(torch.long).clone()
            if use_global_reset
            else torch.zeros(batch_size, dtype=torch.long, device=base_logits.device)
        )
        for k in range(K):
            if use_global_reset:
                at_boundary = mask_positions[:, k] % block_len == 0
                prev = torch.where(at_boundary, torch.zeros_like(prev), prev)
            elif k > 0 and k % block_len == 0:
                prev.zero_()
            bias = torch.matmul(markov_w1.index_select(0, prev), markov_w2)
            if os.environ.get("VLLM_DSPARK_NO_MARKOV", "0") == "1":
                bias.zero_()
            base_logits[:, k].add_(bias)
            logits_k = base_logits[:, k].to(torch.float32)
            if self.diff_temperature == 0.0:
                sampled = logits_k.argmax(dim=-1)
            else:
                sampled, selected_probs, row_logsumexp = \
                    self._sample_dspark_stochastic(logits_k, k)
                assert token_probs is not None and logsumexp is not None
                token_probs[:, k] = selected_probs
                logsumexp[:, k] = row_logsumexp
            tokens[:, k] = sampled
            prev = sampled

        self.last_draft_logits = base_logits.view(
            batch_size * K, base_logits.shape[-1]
        ).detach().contiguous()
        # The standard full-probability interface stays available for other
        # drafters. DSpark's compact q(x) path intentionally leaves it empty.
        self.last_draft_probs = None
        self.last_draft_token_probs = (
            None if token_probs is None else token_probs.view(-1).contiguous()
        )
        self.last_draft_logsumexp = (
            None if logsumexp is None else logsumexp.view(-1).contiguous()
        )
        return tokens.view(-1)

    def set_attn(
        self,
        kv_cache_config: KVCacheConfig,
        attn_metadata_builders: list[AttentionMetadataBuilder],
        block_tables: BlockTables,
    ) -> None:
        self.kv_cache_config = kv_cache_config
        self.attn_metadata_builders = attn_metadata_builders
        self.block_tables = block_tables
        self._cca_group_ids = {
            i for i, group in enumerate(kv_cache_config.kv_cache_groups)
            if any(name in self.cca_layer_names for name in group.layer_names)
        }
        self.slot_mappings = torch.empty(
            len(kv_cache_config.kv_cache_groups),
            self.max_num_tokens,
            dtype=torch.int64,
            device=self.device,
        )

    def capture_model(self) -> None:
        if not self.draft_cudagraph_sizes:
            return
        if not hasattr(self, "kv_cache_config"):
            return
        logger.info("Capturing CUDA graphs for TiDAR drafter...")
        sizes = sorted(self.draft_cudagraph_sizes, reverse=True)
        with graph_capture(device=self.device):
            for num_tokens in sizes:
                self._capture_draft_graph(num_tokens)

    def _get_draft_cudagraph_size(self, num_tokens: int) -> int | None:
        if num_tokens in self.draft_graphs:
            return num_tokens
        return None

    def _prepare_draft_capture_inputs(
        self,
        num_tokens: int,
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
        query_len = self.num_speculative_steps + 1
        assert num_tokens % query_len == 0
        num_reqs = num_tokens // query_len

        self.input_buffers.input_ids[:num_tokens].fill_(self.mask_token_id)
        positions_2d = torch.arange(
            num_tokens, dtype=torch.int64, device=self.device).view(
                num_reqs, query_len)
        self.input_buffers.positions[:num_tokens].copy_(positions_2d.reshape(-1))

        query_start_loc = self.input_buffers.query_start_loc[:num_reqs + 1]
        torch.arange(
            0,
            (num_reqs + 1) * query_len,
            query_len,
            dtype=torch.int32,
            device=self.device,
            out=query_start_loc,
        )
        query_start_loc_cpu = torch.arange(
            0, (num_reqs + 1) * query_len, query_len, dtype=torch.int32)

        seq_lens = self.input_buffers.seq_lens
        seq_lens[:num_reqs].fill_(self.max_model_len)
        seq_lens[num_reqs:].fill_(0)

        block_tables = [
            block_table[:num_reqs]
            for block_table in self.block_tables.input_block_tables
        ]
        slot_mappings = self._build_slot_mappings(positions_2d, block_tables)
        return self._build_attn_metadata(
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=seq_lens,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
        )

    def _run_model_eager(
        self,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            num_tokens_across_dp=num_tokens_across_dp,
            slot_mapping=slot_mappings,
        ):
            hidden_states = self.model(
                input_ids=self.input_buffers.input_ids[:num_tokens],
                positions=self.input_buffers.positions[:num_tokens],
            )
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        return hidden_states, hidden_states

    def _capture_draft_graph(self, num_tokens: int) -> None:
        attn_metadata, slot_mappings = self._prepare_draft_capture_inputs(
            num_tokens)

        hidden_states, _ = self._run_model_eager(
            num_tokens,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp=None,
        )
        if self.draft_hidden_states is None:
            self.draft_hidden_states = torch.empty_like(hidden_states)
        elif self.draft_hidden_states.shape[0] < num_tokens:
            self.draft_hidden_states = torch.empty_like(hidden_states)

        assert num_tokens not in self.draft_graphs
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, self.draft_graph_pool):
            hidden_states, _ = self._run_model_eager(
                num_tokens,
                attn_metadata,
                slot_mappings,
                num_tokens_across_dp=None,
            )
            self.draft_hidden_states[:num_tokens] = hidden_states
        self.draft_graphs[num_tokens] = graph

    @torch.inference_mode()
    def run_model(
        self,
        num_tokens: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cudagraph_size is not None:
            assert cudagraph_size in self.draft_graphs
            self.draft_graphs[cudagraph_size].replay()
            assert self.draft_hidden_states is not None
            hidden_states = self.draft_hidden_states[:num_tokens]
            return hidden_states, hidden_states
        return self._run_model_eager(
            num_tokens,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
        )

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        num_tokens_across_dp: torch.Tensor | None,
    ) -> None:
        if num_tokens <= 0:
            return
        query_len = self.num_speculative_steps + 1
        if num_tokens % query_len != 0:
            raise RuntimeError(
                "V2 TiDAR dummy draft token count must be a multiple of "
                f"{query_len}; got {num_tokens}.")
        attn_metadata, slot_mappings = self._prepare_draft_capture_inputs(
            num_tokens)
        self.run_model(
            num_tokens,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_size=None,
        )

    def _build_slot_mappings(
        self,
        positions_2d: torch.Tensor,
        block_tables: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        assert self.slot_mappings is not None
        num_tokens = positions_2d.numel()
        exceeds_max_model_len = positions_2d >= self.max_model_len
        clamped_positions = torch.where(
            exceeds_max_model_len,
            torch.zeros((), dtype=positions_2d.dtype, device=positions_2d.device),
            positions_2d,
        )
        for group_idx, group in enumerate(self.kv_cache_config.kv_cache_groups):
            block_size = group.kv_cache_spec.block_size
            block_numbers = clamped_positions // block_size
            block_ids = block_tables[group_idx].gather(
                dim=1, index=block_numbers.to(torch.long))
            slots = block_ids.to(torch.int64) * block_size + (
                clamped_positions % block_size).to(torch.int64)
            slots = slots.masked_fill(exceeds_max_model_len, PAD_SLOT_ID)
            self.slot_mappings[group_idx, :num_tokens].copy_(slots.reshape(-1))
        return self.slot_mappings[:, :num_tokens]

    def _build_attn_metadata(
        self,
        num_reqs: int,
        num_tokens: int,
        query_start_loc: torch.Tensor,
        query_start_loc_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        block_tables: Sequence[torch.Tensor],
        slot_mappings: torch.Tensor,
    ) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
        attn_metadata: dict[str, Any] = {}
        slot_mappings_by_layer: dict[str, torch.Tensor] = {}
        query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        max_query_len = int(query_lens_cpu.max()) if query_lens_cpu.numel() else 0
        max_seq_len = int(seq_lens[:num_reqs].max().item()) if num_reqs else 0

        for group_idx, group in enumerate(self.kv_cache_config.kv_cache_groups):
            common = CommonAttentionMetadata(
                query_start_loc=query_start_loc,
                query_start_loc_cpu=query_start_loc_cpu,
                seq_lens=seq_lens[:num_reqs],
                max_seq_len=max_seq_len,
                num_reqs=num_reqs,
                num_actual_tokens=num_tokens,
                max_query_len=max_query_len,
                block_table_tensor=block_tables[group_idx],
                slot_mapping=slot_mappings[group_idx],
                causal=False,
            )
            if group_idx in self._cca_group_ids:
                ar_slots = block_tables[group_idx][:num_reqs, 0]
                setattr(common, "state_indices_tensor_override", ar_slots)
                if block_tables[group_idx].shape[1] > 1:
                    setattr(common, "state_indices_tensor_write_override",
                            block_tables[group_idx][:num_reqs, 1])
                setattr(common, "cca_drafter_pass", True)

            metadata = self.attn_metadata_builders[group_idx].build_for_drafting(
                common_attn_metadata=common,
                draft_index=0,
            )
            for layer_name in group.layer_names:
                attn_metadata[layer_name] = metadata
                slot_mappings_by_layer[layer_name] = slot_mappings[group_idx]
        return attn_metadata, slot_mappings_by_layer

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        num_tokens_across_dp: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del last_hidden_states, aux_hidden_states, next_prefill_tokens
        del temperature, seeds

        num_reqs = input_batch.num_reqs
        K = self.num_speculative_steps
        query_len = K + 1
        num_tokens = num_reqs * query_len
        if num_tokens > self.max_num_tokens:
            raise RuntimeError(
                f"TiDAR draft forward needs {num_tokens} tokens, but the "
                f"V2 input buffer only holds {self.max_num_tokens}.")

        idx_mapping = input_batch.idx_mapping[:num_reqs]
        next_token_ids = last_sampled.view(-1).gather(
            0, idx_mapping.to(torch.long)).to(torch.int32)
        live_seq_lens = (
            input_batch.seq_lens[:num_reqs].to(torch.int32)
            - num_rejected[:num_reqs].to(torch.int32)
        ).clamp_(min=0, max=self.max_model_len)

        offsets = torch.arange(query_len, dtype=torch.int64, device=self.device)
        positions_2d = live_seq_lens.to(torch.int64).view(-1, 1) + offsets.view(1, -1)
        input_ids_2d = torch.full(
            (num_reqs, query_len),
            self.mask_token_id,
            dtype=torch.int32,
            device=self.device,
        )
        input_ids_2d[:, 0] = next_token_ids

        self.input_buffers.input_ids[:num_tokens].copy_(input_ids_2d.reshape(-1))
        self.input_buffers.positions[:num_tokens].copy_(positions_2d.reshape(-1))
        query_start_loc = self.input_buffers.query_start_loc[:num_reqs + 1]
        torch.arange(
            0,
            (num_reqs + 1) * query_len,
            query_len,
            dtype=torch.int32,
            device=self.device,
            out=query_start_loc,
        )
        seq_lens = self.input_buffers.seq_lens
        seq_lens[:num_reqs].copy_(
            (live_seq_lens + query_len).clamp_(max=self.max_model_len))
        seq_lens[num_reqs:].fill_(0)

        block_tables = [
            block_table[:num_reqs]
            for block_table in self.block_tables.input_block_tables
        ]
        slot_mappings = self._build_slot_mappings(positions_2d, block_tables)
        cudagraph_size = self._get_draft_cudagraph_size(num_tokens)
        if cudagraph_size is None:
            query_start_loc_cpu = torch.arange(
                0, (num_reqs + 1) * query_len, query_len, dtype=torch.int32)
            attn_metadata, slot_mappings_by_layer = self._build_attn_metadata(
                num_reqs=num_reqs,
                num_tokens=num_tokens,
                query_start_loc=query_start_loc,
                query_start_loc_cpu=query_start_loc_cpu,
                seq_lens=seq_lens,
                block_tables=block_tables,
                slot_mappings=slot_mappings,
            )
        else:
            attn_metadata = None
            slot_mappings_by_layer = None

        hidden_states, _ = self.run_model(
            num_tokens,
            attn_metadata,
            slot_mappings_by_layer,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_size=cudagraph_size,
        )
        hidden_states = hidden_states[:num_tokens]
        draft_hidden_states = hidden_states.view(
            num_reqs, query_len, -1)[:, 1:, :].contiguous().view(num_reqs * K, -1)
        self.last_draft_hidden_states = draft_hidden_states.detach()
        if self._dspark_active():
            draft_token_ids = self._dspark_sample_drafts(
                draft_hidden_states,
                num_reqs,
                mask_positions=positions_2d[:, 1:],
                prev_token=next_token_ids,
            )
            self.draft_tokens[:num_reqs].copy_(
                draft_token_ids.view(num_reqs, K)
            )
            if torch.any(num_sampled[:num_reqs] == 0):
                inactive = num_sampled[:num_reqs] == 0
                self.draft_tokens[:num_reqs][inactive] = -1
            return self.draft_tokens[:num_reqs]

        logits = self.model.compute_logits(draft_hidden_states)
        self.last_draft_logits = logits.detach().contiguous()
        self.last_draft_token_probs = None
        self.last_draft_logsumexp = None
        if self.diff_temperature == 0.0:
            self.last_draft_probs = None
            draft_token_ids = logits.argmax(dim=-1)
        else:
            scaled_logits = logits.to(torch.float32) / self.diff_temperature
            draft_probs = torch.softmax(scaled_logits, dim=-1)
            draft_token_ids = torch.multinomial(
                draft_probs, num_samples=1).squeeze(-1)
            self.last_draft_probs = draft_probs.contiguous()

        self.draft_tokens[:num_reqs].copy_(draft_token_ids.view(num_reqs, K))
        if torch.any(num_sampled[:num_reqs] == 0):
            inactive = num_sampled[:num_reqs] == 0
            self.draft_tokens[:num_reqs][inactive] = -1
        return self.draft_tokens[:num_reqs]
