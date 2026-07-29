# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import (
    GraphCaptureContext,
    graph_capture,
    is_global_first_rank,
)
from vllm.forward_context import set_forward_context
from vllm.v1.attention.backend import AttentionMetadataBuilder
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    build_slot_mappings_by_layer,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.dp_utils import make_num_tokens_across_dp
from vllm.v1.worker.gpu.input_batch import InputBuffers


class CudaGraphBatchType(IntEnum):
    DEFAULT = 0
    TIDAR_VERIFY = 1


@dataclass(frozen=True)
class CudaGraphKey:
    num_tokens: int
    batch_type: CudaGraphBatchType = CudaGraphBatchType.DEFAULT


class CudaGraphManager:
    def __init__(self, vllm_config: VllmConfig, uses_mrope: bool, device: torch.device):
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.uses_mrope = uses_mrope
        self.device = device

        self.max_model_len = vllm_config.model_config.max_model_len
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.dp_size = vllm_config.parallel_config.data_parallel_size
        self.compilation_config = vllm_config.compilation_config
        assert self.compilation_config is not None
        self.cudagraph_mode = self.compilation_config.cudagraph_mode
        self.cudagraph_sizes = get_cudagraph_sizes(
            self.compilation_config.cudagraph_capture_sizes,
            self.max_num_reqs,
            self.max_num_tokens,
            self.cudagraph_mode,
        )
        speculative_config = vllm_config.speculative_config
        self._tidar_tokens_per_verify_req: int | None = None
        if (speculative_config is not None
                and getattr(speculative_config, "use_tidar", lambda: False)()):
            num_spec_tokens = getattr(
                speculative_config, "num_speculative_tokens", None)
            if num_spec_tokens is not None:
                self._tidar_tokens_per_verify_req = int(num_spec_tokens) + 1

        self.graphs: dict[CudaGraphKey, torch.cuda.CUDAGraph] = {}
        self.pool = torch.cuda.graph_pool_handle()
        self.hidden_states: torch.Tensor | None = None
        # Piecewise wrappers share one global graph memory pool. Keep every
        # lazy capture and replay on the same stream to preserve its ordering.
        self.piecewise_capture_context = GraphCaptureContext(
            torch.cuda.Stream(device=self.device)
        )

    def get_non_full_runtime_mode(
        self,
        *,
        model_dummy_run: bool = False,
        has_spec_decode_tokens: bool = False,
    ) -> CUDAGraphMode:
        """Select the safe runtime mode when a full graph is unavailable.

        Geometry-keyed verifier batches use full graphs when available.
        Partial verifier configurations excluded from full capture can still
        graph compiler partitions around dynamic attention and CCA metadata.
        Prompt prefills and model dummy runs stay eager.
        """
        # Piecewise graphs are lazily captured. Dummy/profile and prompt
        # prefill runs stay eager; verifier fallbacks use the runner's
        # persistent non-default capture stream.
        if model_dummy_run or not has_spec_decode_tokens:
            return CUDAGraphMode.NONE
        if self.cudagraph_mode.has_piecewise_cudagraphs():
            return CUDAGraphMode.PIECEWISE
        return CUDAGraphMode.NONE

    def needs_capture(self) -> bool:
        return bool(self._get_capture_keys())

    def piecewise_graph_capture(self):
        return graph_capture(
            device=self.device,
            graph_capture_context=self.piecewise_capture_context,
        )

    def get_cudagraph_key(
        self,
        num_tokens_after_padding: int,
        num_tokens_per_request: Iterable[int],
    ) -> CudaGraphKey | None:
        num_tokens_per_request = list(num_tokens_per_request)
        tidar_verify_len = self._tidar_tokens_per_verify_req
        if tidar_verify_len is not None:
            decode_only = all(x == 1 for x in num_tokens_per_request)
            verify_only = all(
                x == tidar_verify_len for x in num_tokens_per_request)
            if not decode_only and not verify_only:
                return None
            if verify_only:
                # DP padding currently carries token counts, not synthetic
                # verifier request rows. Keep that configuration out of full
                # graphs until its cross-rank metadata includes request
                # geometry; piecewise graphs remain available.
                if self.dp_size > 1:
                    return None
                size = self.cudagraph_sizes.get(num_tokens_after_padding)
                if size != num_tokens_after_padding:
                    return None
                return CudaGraphKey(size, CudaGraphBatchType.TIDAR_VERIFY)

        size = get_cudagraph_size(
            num_tokens_after_padding,
            num_tokens_per_request,
            self.cudagraph_sizes,
            self.cudagraph_mode,
        )
        if size is None:
            return None
        if tidar_verify_len is not None and size > self.max_num_reqs:
            # Decode batches cannot replay a verifier graph. Their padded
            # size never exceeds max_num_reqs, but retain this guard for
            # explicitly configured capture-size lists.
            return None
        return CudaGraphKey(size)

    def get_cudagraph_size(
        self,
        num_tokens_after_padding: int,
        num_tokens_per_request: Iterable[int],
    ) -> int | None:
        key = self.get_cudagraph_key(
            num_tokens_after_padding, num_tokens_per_request)
        return None if key is None else key.num_tokens

    def get_padded_cudagraph_key(
        self,
        num_tokens_after_padding: int,
        selected_key: CudaGraphKey | None,
    ) -> CudaGraphKey:
        batch_type = (
            CudaGraphBatchType.DEFAULT
            if selected_key is None else selected_key.batch_type
        )
        return CudaGraphKey(num_tokens_after_padding, batch_type)

    def _get_capture_keys(self) -> set[CudaGraphKey]:
        sizes = set(self.cudagraph_sizes.values())
        tidar_verify_len = self._tidar_tokens_per_verify_req
        if tidar_verify_len is None:
            # Ordinary AR full graphs are decode-only: one token per request.
            # Larger synthetic token counts create multi-token prefill rows,
            # which are neither a valid AR replay geometry nor graph-safe for
            # recurrent CCA's prompt-prefill state updates.
            return {
                CudaGraphKey(size) for size in sizes
                if size <= self.max_num_reqs}

        keys = {
            CudaGraphKey(size)
            for size in sizes
            if size <= self.max_num_reqs
        }
        if self.dp_size == 1:
            keys.update(
                CudaGraphKey(size, CudaGraphBatchType.TIDAR_VERIFY)
                for size in sizes
                if size % tidar_verify_len == 0
                and 0 < size // tidar_verify_len <= self.max_num_reqs
            )
        return keys

    def _get_num_reqs_for_capture(self, key: CudaGraphKey) -> int:
        if key.batch_type == CudaGraphBatchType.TIDAR_VERIFY:
            assert self._tidar_tokens_per_verify_req is not None
            return key.num_tokens // self._tidar_tokens_per_verify_req
        return min(key.num_tokens, self.max_num_reqs)

    def capture_graph(
        self,
        key: CudaGraphKey,
        model: nn.Module,
        input_buffers: InputBuffers,
        mrope_positions: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
        block_tables: BlockTables,
        attn_metadata_builders: list[AttentionMetadataBuilder],
        kv_cache_config: KVCacheConfig,
    ) -> None:
        num_tokens = key.num_tokens
        num_reqs = self._get_num_reqs_for_capture(key)
        input_ids = input_buffers.input_ids[:num_tokens]
        positions = input_buffers.positions[:num_tokens]
        if self.uses_mrope:
            assert mrope_positions is not None
            positions = mrope_positions[:, :num_tokens]
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds[:num_tokens]
        attn_metadata, slot_mappings = prepare_inputs_to_capture(
            num_reqs,
            num_tokens,
            input_buffers,
            block_tables,
            attn_metadata_builders,
            self.max_model_len,
            kv_cache_config,
        )
        num_tokens_across_dp = make_num_tokens_across_dp(self.dp_size, num_tokens)

        # Warm up.
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            num_tokens_across_dp=num_tokens_across_dp,
            slot_mapping=slot_mappings,
        ):
            hidden_states = model(
                input_ids=input_ids,
                positions=positions,
                inputs_embeds=inputs_embeds,
            )
            if self.hidden_states is None:
                self.hidden_states = torch.empty_like(hidden_states)

        # Capture the graph.
        assert key not in self.graphs
        graph = torch.cuda.CUDAGraph()
        with (
            set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                num_tokens_across_dp=num_tokens_across_dp,
                slot_mapping=slot_mappings,
            ),
            torch.cuda.graph(graph, self.pool),
        ):
            hidden_states = model(
                input_ids=input_ids,
                positions=positions,
                inputs_embeds=inputs_embeds,
            )
            self.hidden_states[:num_tokens] = hidden_states
        self.graphs[key] = graph

    @torch.inference_mode()
    def capture(
        self,
        model: nn.Module,
        input_buffers: InputBuffers,
        mrope_positions: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
        block_tables: BlockTables,
        attn_metadata_builders: list[AttentionMetadataBuilder],
        kv_cache_config: KVCacheConfig,
    ) -> None:
        keys_to_capture = sorted(
            self._get_capture_keys(),
            key=lambda key: (key.num_tokens, key.batch_type),
            reverse=True,
        )
        if is_global_first_rank():
            keys_to_capture = tqdm(
                keys_to_capture, desc="Capturing CUDA graphs")

        with graph_capture(device=self.device):
            for key in keys_to_capture:
                self.capture_graph(
                    key,
                    model=model,
                    input_buffers=input_buffers,
                    mrope_positions=mrope_positions,
                    inputs_embeds=inputs_embeds,
                    block_tables=block_tables,
                    attn_metadata_builders=attn_metadata_builders,
                    kv_cache_config=kv_cache_config,
                )

    def run(self, key: CudaGraphKey) -> torch.Tensor:
        assert key in self.graphs
        self.graphs[key].replay()
        assert self.hidden_states is not None
        return self.hidden_states[:key.num_tokens]


def get_cudagraph_sizes(
    capture_sizes: list[int] | None,
    max_num_reqs: int,
    max_num_tokens: int,
    cudagraph_mode: CUDAGraphMode,
) -> dict[int, int]:
    if not cudagraph_mode.has_full_cudagraphs():
        return {}
    if not capture_sizes:
        return {}

    capture_sizes = sorted(capture_sizes)
    # Limit the capture sizes to the max number of requests or tokens.
    upper_bound = (
        max_num_reqs
        if cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY
        else max_num_tokens
    )
    capture_sizes = [x for x in capture_sizes if x <= upper_bound]
    if not capture_sizes:
        return {}

    cudagraph_sizes: dict[int, int] = {}
    for i in range(1, capture_sizes[-1] + 1):
        for x in capture_sizes:
            if i <= x:
                cudagraph_sizes[i] = x
                break
    return cudagraph_sizes


def get_cudagraph_size(
    num_tokens_after_dp_padding: int,
    num_tokens_per_request: Iterable[int],
    cudagraph_sizes: dict[int, int],
    cudagraph_mode: CUDAGraphMode,
) -> int | None:
    if not cudagraph_mode.has_full_cudagraphs():
        # No full CUDA graph is used.
        return None

    size = cudagraph_sizes.get(num_tokens_after_dp_padding)
    if size is None:
        # No CUDA graph for this size.
        return None

    is_mixed = any(x > 1 for x in num_tokens_per_request)
    if is_mixed and cudagraph_mode.mixed_mode() != CUDAGraphMode.FULL:
        # Prefill is included, and this mode doesn't use CUDA graph for it.
        return None
    return size


def capture_graphs(
    cudagraph_sizes: dict[int, int],
    device: torch.device,
    capture_fn: Callable,
    **capture_kwargs,
) -> None:
    # Capture larger graphs first.
    sizes_to_capture = sorted(set(cudagraph_sizes.values()), reverse=True)
    if is_global_first_rank():
        sizes_to_capture = tqdm(sizes_to_capture, desc="Capturing CUDA graphs")

    with graph_capture(device=device):
        for size in sizes_to_capture:
            capture_fn(size, **capture_kwargs)


def prepare_inputs_to_capture(
    num_reqs: int,
    num_tokens: int,
    input_buffers: InputBuffers,
    block_tables: BlockTables,
    attn_metadata_builders: list[AttentionMetadataBuilder],
    max_model_len: int,
    kv_cache_config: KVCacheConfig,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    num_tokens_per_req = num_tokens // num_reqs

    query_start_loc_np = np.arange(num_reqs + 1, dtype=np.int32) * num_tokens_per_req
    query_start_loc_np[-1] = num_tokens
    query_start_loc_cpu = torch.from_numpy(query_start_loc_np)
    input_buffers.query_start_loc[: num_reqs + 1] = query_start_loc_cpu
    input_buffers.query_start_loc[num_reqs + 1 :] = num_tokens
    query_start_loc = input_buffers.query_start_loc[: num_reqs + 1]

    # HACK(woosuk): For faster warmup, we set seq_lens (GPU) to num_tokens
    # rather than max_model_len.
    input_buffers.seq_lens[:num_reqs] = num_tokens
    input_buffers.seq_lens[num_reqs:] = 0

    input_block_tables = [x[:num_reqs] for x in block_tables.input_block_tables]
    slot_mappings = block_tables.slot_mappings[:, :num_tokens]
    slot_mappings_by_layer = build_slot_mappings_by_layer(
        slot_mappings, kv_cache_config
    )

    attn_metadata = build_attn_metadata(
        attn_metadata_builders=attn_metadata_builders,
        num_reqs=num_reqs,
        num_tokens=num_tokens,
        query_start_loc_gpu=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=input_buffers.seq_lens,
        max_seq_len=max_model_len,
        block_tables=input_block_tables,
        slot_mappings=slot_mappings,
        kv_cache_config=kv_cache_config,
        for_cudagraph_capture=True,
    )
    return attn_metadata, slot_mappings_by_layer
