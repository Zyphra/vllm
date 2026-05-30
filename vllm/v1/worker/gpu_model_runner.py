# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import itertools
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from typing import TYPE_CHECKING, Any, NamedTuple, Optional, Union, cast

import numpy as np
import torch
import torch.distributed
import torch.nn as nn
from tqdm import tqdm
from typing_extensions import TypeAlias

import vllm.envs as envs
from vllm.attention import Attention, AttentionType
from vllm.attention.backends.abstract import AttentionBackend
from vllm.model_executor.layers.attention.chunked_local_attention import ChunkedLocalAttention
from vllm.compilation.counter import compilation_counter
from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.compilation.monitor import set_cudagraph_capturing_enabled
from vllm.config import (CompilationLevel, CUDAGraphMode, VllmConfig,
                         get_layers_from_vllm_config, update_config)
from vllm.distributed.eplb.eplb_state import EplbState
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group)
from vllm.distributed.kv_transfer.kv_connector.utils import copy_kv_blocks
from vllm.distributed.parallel_state import (
    get_pp_group, get_tp_group, graph_capture, is_global_first_rank,
    prepare_communication_buffer_for_model)
from vllm.forward_context import (BatchDescriptor, DPMetadata,
                                  set_forward_context)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.model_executor.model_loader import TensorizerLoader, get_model_loader
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
# yapf conflicts with isort for this block
# yapf: disable
from vllm.model_executor.models.interfaces import (SupportsMultiModal,
                                                   is_mixture_of_experts,
                                                   supports_eagle3,
                                                   supports_mrope,
                                                   supports_multimodal_pruning,
                                                   supports_transcription)
# yapf: enable
from vllm.model_executor.models.interfaces_base import (
    VllmModelForPooling, is_pooling_model, is_text_generation_model)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (BatchedTensorInputs, MultiModalKwargsItem,
                                    PlaceholderRange)
from vllm.multimodal.utils import group_mm_kwargs_by_modality
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingType
from vllm.sequence import IntermediateTensors
from vllm.tasks import GenerationTask, PoolingTask, SupportedTask
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, DeviceMemoryProfiler,
                        GiB_bytes, cdiv, check_use_alibi, get_dtype_size,
                        is_pin_memory_available,
                        length_from_prompt_token_ids_or_embeds, round_up,
                        supports_dynamo)
from vllm.utils.jsontree import json_map_leaves
from vllm.v1.attention.backends.flash_attn import AttentionMetadata
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.attention.backends.utils import (
    AttentionCGSupport, AttentionMetadataBuilder, CommonAttentionMetadata,
    create_fast_prefill_custom_backend,
    reorder_batch_to_split_decodes_and_prefills, split_attn_metadata)
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
# yapf conflicts with isort for this block
# yapf: disable
from vllm.v1.kv_cache_interface import (AttentionSpec,
                                        ChunkedLocalAttentionSpec,
                                        CrossAttentionSpec,
                                        EncoderOnlyAttentionSpec,
                                        FullAttentionSpec, KVCacheConfig,
                                        KVCacheGroupSpec, KVCacheSpec,
                                        MambaSpec, MLAAttentionSpec,
                                        SlidingWindowSpec,
                                        UniformTypeKVCacheSpecs)
# yapf: enable
from vllm.v1.outputs import (EMPTY_MODEL_RUNNER_OUTPUT, AsyncModelRunnerOutput,
                             DraftTokenIds, LogprobsLists, LogprobsTensors,
                             ModelRunnerOutput, PoolerOutput, SamplerOutput)
from vllm.v1.pool.metadata import PoolingMetadata
from vllm.v1.sample.logits_processor import LogitsProcessors, build_logitsprocs
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.medusa import MedusaProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.ngram_proposer import NgramProposer
from vllm.v1.spec_decode.tidar import TiDARProposer
from vllm.v1.structured_output.utils import apply_grammar_bitmask
from vllm.v1.utils import CpuGpuBuffer, record_function_or_nullcontext
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper
from vllm.v1.worker.kv_connector_model_runner_mixin import (
    KVConnectorModelRunnerMixin)
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin
from vllm.v1.worker.ubatch_splitting import (check_ubatch_thresholds,
                                             ubatch_split)
from vllm.v1.worker.ubatch_utils import UBatchSlice, UBatchSlices
from vllm.v1.worker.utils import is_residual_scattered_for_sp

from .utils import (AttentionGroup, MultiModalBudget,
                    add_kv_sharing_layers_to_kv_cache_groups, bind_kv_cache,
                    gather_mm_placeholders, sanity_check_mm_encoder_outputs,
                    scatter_mm_placeholders)

if TYPE_CHECKING:
    from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)

AttnMetadataDict: TypeAlias = dict[str, AttentionMetadata]
# list when ubatching is enabled
PerLayerAttnMetadata: TypeAlias = Union[list[AttnMetadataDict],
                                        AttnMetadataDict]


# Wrapper for ModelRunnerOutput to support overlapped execution.
class AsyncGPUModelRunnerOutput(AsyncModelRunnerOutput):

    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        sampled_token_ids: torch.Tensor,
        invalid_req_indices: list[int],
        async_output_copy_stream: torch.cuda.Stream,
    ):
        self._model_runner_output = model_runner_output
        self._invalid_req_indices = invalid_req_indices

        # Event on the copy stream so we can synchronize the non-blocking copy.
        self._async_copy_ready_event = torch.cuda.Event()

        # Keep a reference to the device tensor to avoid it being
        # deallocated until we finish copying it to the host.
        self._sampled_token_ids = sampled_token_ids

        # Initiate the copy on a separate stream, but do not synchronize it.
        default_stream = torch.cuda.current_stream()
        with torch.cuda.stream(async_output_copy_stream):
            async_output_copy_stream.wait_stream(default_stream)
            self._sampled_token_ids_cpu = self._sampled_token_ids.to(
                'cpu', non_blocking=True)
            self._async_copy_ready_event.record()

    def get_output(self) -> ModelRunnerOutput:
        """Copy the device tensors to the host and return a ModelRunnerOutput.
        
        This function blocks until the copy is finished.
        """
        self._async_copy_ready_event.synchronize()

        # Release the device tensor once the copy has completed
        del self._sampled_token_ids

        valid_sampled_token_ids = self._sampled_token_ids_cpu.tolist()
        for i in self._invalid_req_indices:
            valid_sampled_token_ids[i].clear()

        output = self._model_runner_output
        output.sampled_token_ids = valid_sampled_token_ids
        return output


class GPUModelRunner(LoRAModelRunnerMixin, KVConnectorModelRunnerMixin):

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.compilation_config = vllm_config.compilation_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config

        from vllm.model_executor.models.utils import set_cpu_offload_max_bytes
        set_cpu_offload_max_bytes(
            int(self.cache_config.cpu_offload_gb * 1024**3))

        model_config = self.model_config
        cache_config = self.cache_config
        scheduler_config = self.scheduler_config
        parallel_config = self.parallel_config
        self.device = device
        self.pin_memory = is_pin_memory_available()
        self.dtype = self.model_config.dtype
        if cache_config.cache_dtype == "auto":
            self.kv_cache_dtype = self.dtype
        else:
            self.kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
                cache_config.cache_dtype]

        self.is_pooling_model = (model_config.runner_type == 'pooling')
        self.enable_prompt_embeds = model_config.enable_prompt_embeds
        self.is_multimodal_raw_input_only_model = (
            model_config.is_multimodal_raw_input_only_model)
        # This will be overridden in load_model()
        self.is_multimodal_pruning_enabled = False
        self.max_model_len = model_config.max_model_len
        self.dcp_world_size = self.parallel_config.decode_context_parallel_size
        self.max_num_tokens = scheduler_config.max_num_batched_tokens
        self.max_num_reqs = scheduler_config.max_num_seqs

        # Broadcast PP output for external_launcher (torchrun)
        # to make sure we are synced across pp ranks
        # TODO: Support overlapping mirco-batches
        # https://github.com/vllm-project/vllm/issues/18019
        self.broadcast_pp_output = (
            self.parallel_config.distributed_executor_backend
            == "external_launcher" and len(get_pp_group().ranks) > 0)

        # Model-related.
        self.num_query_heads = model_config.get_num_attention_heads(
            parallel_config)
        self.hidden_size = model_config.get_hidden_size()
        self.attention_chunk_size = model_config.attention_chunk_size
        # Only relevant for models using ALiBi (e.g, MPT)
        self.use_alibi = check_use_alibi(model_config)

        self.cascade_attn_enabled = not self.model_config.disable_cascade_attn

        # Multi-modal data support
        self.mm_registry = MULTIMODAL_REGISTRY
        self.uses_mrope = model_config.uses_mrope
        self.supports_mm_inputs = self.mm_registry.supports_multimodal_inputs(
            model_config)

        if self.model_config.is_encoder_decoder:
            # Maximum length of the encoder input, only for encoder-decoder
            # models.
            self.max_encoder_len = scheduler_config.\
                            max_num_encoder_input_tokens
        else:
            self.max_encoder_len = 0

        # Sampler
        self.sampler = Sampler(logprobs_mode=self.model_config.logprobs_mode)

        self.eplb_state: Optional[EplbState] = None
        """
        State of the expert parallelism load balancer.

        Will be lazily initialized when the model is loaded.
        """

        # Lazy initializations
        # self.model: nn.Module  # Set after load_model
        # Initialize in initialize_kv_cache
        self.kv_caches: list[torch.Tensor] = []
        # indexes: [kv_cache_group_id][attn_group]
        self.attn_groups: list[list[AttentionGroup]] = []
        # self.kv_cache_config: KVCacheConfig

        # mm_hash ->  encoder_output
        self.encoder_cache: dict[str, torch.Tensor] = {}

        self.use_aux_hidden_state_outputs = False
        # Set up speculative decoding.
        # NOTE(Jiayi): currently we put the entire draft model on
        # the last PP rank. This is not ideal if there are many
        # layers in the draft model.
        if self.speculative_config and get_pp_group().is_last_rank:
            if self.speculative_config.method == "ngram":
                self.drafter = NgramProposer(self.vllm_config)
            elif self.speculative_config.use_tidar():
                self.drafter = TiDARProposer(self.vllm_config, self.device,
                                             self)  # type: ignore
            elif self.speculative_config.use_eagle():
                self.drafter = EagleProposer(self.vllm_config, self.device,
                                             self)  # type: ignore
                if self.speculative_config.uses_aux_hidden_state_outputs():
                    self.use_aux_hidden_state_outputs = True
            elif self.speculative_config.method == "medusa":
                self.drafter = MedusaProposer(
                    vllm_config=self.vllm_config,
                    device=self.device)  # type: ignore
            else:
                raise ValueError("Unknown speculative decoding method: "
                                 f"{self.speculative_config.method}")
            self.rejection_sampler = RejectionSampler()

        # Request states.
        self.requests: dict[str, CachedRequestState] = {}
        self.comm_stream = torch.cuda.Stream()

        # Input Batch
        # NOTE(Chen): Ideally, we should initialize the input batch inside
        # `initialize_kv_cache` based on the kv cache config. However, as in
        # https://github.com/vllm-project/vllm/pull/18298, due to some unknown
        # reasons, we have to initialize the input batch before `load_model`,
        # quantization + weight offloading will fail otherwise. As a temporary
        # solution, we initialize the input batch here, and re-initialize it
        # in `initialize_kv_cache` if the block_sizes here is different from
        # the block_sizes in the kv cache config.
        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            # We need to use the encoder length for encoder-decoer
            # because of KV cache for cross-attention.
            max_model_len=max(self.max_model_len, self.max_encoder_len),
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
            block_sizes=[self.cache_config.block_size],
            is_spec_decode=bool(self.vllm_config.speculative_config),
            logitsprocs=build_logitsprocs(
                self.vllm_config, self.device, self.pin_memory,
                self.is_pooling_model,
                self.vllm_config.model_config.logits_processors),
            is_pooling_model=self.is_pooling_model,
        )

        self.use_async_scheduling = self.scheduler_config.async_scheduling
        self.async_output_copy_stream = torch.cuda.Stream() if \
            self.use_async_scheduling else None

        # TODO(woosuk): Provide an option to tune the max cudagraph batch size.
        # The convention is different.
        # self.cudagraph_batch_sizes sorts in ascending order.
        # The batch sizes in the config are in descending order.
        if self.compilation_config.cudagraph_capture_sizes and \
                self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
            self.cudagraph_batch_sizes = list(
                reversed(self.compilation_config.cudagraph_capture_sizes))

        # Cache the device properties.
        self._init_device_properties()

        # Persistent buffers for CUDA graphs.
        self.input_ids = self._make_buffer(self.max_num_tokens,
                                           dtype=torch.int32)
        self.positions = self._make_buffer(self.max_num_tokens,
                                           dtype=torch.int64)
        self.query_start_loc = self._make_buffer(self.max_num_reqs + 1,
                                                 dtype=torch.int32)
        self.seq_lens = self._make_buffer(self.max_num_reqs, dtype=torch.int32)
        # Because inputs_embeds may be bfloat16 and we don't need a numpy
        # version of this tensor, avoid a RuntimeError by not creating a
        # numpy buffer.
        self.inputs_embeds = self._make_buffer(self.max_num_tokens,
                                               self.hidden_size,
                                               dtype=self.dtype,
                                               numpy=False)
        self.is_token_ids = self._make_buffer(self.max_num_tokens,
                                              dtype=torch.bool)
        self.discard_request_indices = self._make_buffer(self.max_num_reqs,
                                                         dtype=torch.int64)
        self.num_discarded_requests = 0

        self.num_decode_draft_tokens = self._make_buffer(self.max_num_reqs,
                                                         dtype=torch.int32)
        self.num_accepted_tokens = self._make_buffer(self.max_num_reqs,
                                                     dtype=torch.int64)

        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            # NOTE: `mrope_positions` is implemented with one additional dummy
            # position on purpose to make it non-contiguous so that it can work
            # with torch compile.
            # See detailed explanation in https://github.com/vllm-project/vllm/pull/12128#discussion_r1926431923

            # NOTE: When M-RoPE is enabled, position ids are 3D regardless of
            # the modality of inputs. For text-only inputs, each dimension has
            # identical position IDs, making M-RoPE functionally equivalent to
            # 1D-RoPE.
            # See page 5 of https://arxiv.org/abs/2409.12191
            self.mrope_positions = self._make_buffer(
                (3, self.max_num_tokens + 1), dtype=torch.int64)

        # CUDA event to synchronize use of reused CPU tensors between steps
        # when async scheduling is enabled.
        self.prepare_inputs_event: Optional[torch.cuda.Event] = None
        if self.use_async_scheduling:
            self.prepare_inputs_event = torch.cuda.Event()
            # Start in a completed state.
            self.prepare_inputs_event.record(torch.cuda.default_stream())

        # None in the first PP rank. The rest are set after load_model.
        self.intermediate_tensors: Optional[IntermediateTensors] = None

        # OPTIMIZATION: Cache the tensors rather than creating them every step.
        # Keep in int64 to avoid overflow with long context
        self.arange_np = np.arange(max(self.max_num_reqs + 1,
                                       self.max_model_len,
                                       self.max_num_tokens),
                                   dtype=np.int64)

        # Layer pairings for cross-layer KV sharing.
        # If an Attention layer `layer_name` is in the keys of this dict, it
        # means this layer will perform attention using the keys and values
        # from the KV cache of `shared_kv_cache_layers[layer_name]`.
        self.shared_kv_cache_layers: dict[str, str] = {}
        self.kv_sharing_fast_prefill_eligible_layers: set[str] = set()

        self.kv_sharing_fast_prefill_logits_indices = None
        if self.cache_config.kv_sharing_fast_prefill:
            self.kv_sharing_fast_prefill_logits_indices = torch.zeros(
                self.max_num_tokens, dtype=torch.int32, device=self.device)

        self.uniform_decode_query_len = 1 if not self.speculative_config else \
            1 + self.speculative_config.num_speculative_tokens
        # TiDAR single-forward inflates per-req tokens from K+1 to
        # K+1 + P*K (verify segment + P pre-draft proposals of K tokens
        # each). For FULL_DECODE_ONLY cudagraph capture to hit the
        # inflated layout, override the uniform-decode query length to
        # the inflated value -- but ONLY when FULL cudagraph is active.
        # Under PIECEWISE the override changes the uniform_decode
        # dispatch check at _prepare_inputs (line ~1077) and breaks
        # captured-chunk dispatching for the K+1 spec-decode-step
        # batches; we hit garbage output (all-special-tokens). The
        # standard spec-decode contract (rejection sampler, etc.)
        # still keys off ``num_speculative_tokens`` = K, not this value.
        _cg_mode = getattr(self.compilation_config, "cudagraph_mode", None)
        _full_active = (_cg_mode is not None
                        and hasattr(_cg_mode, "has_full_cudagraphs")
                        and _cg_mode.has_full_cudagraphs())
        if (_full_active
                and self.speculative_config is not None
                and self.speculative_config.use_tidar()
                and getattr(self, "drafter", None) is not None
                and getattr(self.drafter, "single_forward_mode", False)):
            K = self.speculative_config.num_speculative_tokens
            P = len(getattr(self.drafter, "proposal_acc_levels", ()))
            if P > 0:
                # ZAP-ONLY: K+1 layout always. uniform_decode_query_len
                # = K+1 + P*(K+1).
                verify_len = self.drafter.verify_len
                self.uniform_decode_query_len = (
                    verify_len + P * (K + 1))

        # Cudagraph dispatcher for runtime cudagraph dispatching.
        self.cudagraph_dispatcher = CudagraphDispatcher(self.vllm_config)

        self.mm_budget = MultiModalBudget(
            self.model_config,
            self.scheduler_config,
            self.mm_registry,
        ) if self.supports_mm_inputs else None

        self.reorder_batch_threshold: Optional[int] = None

        # Attention layers that are only in the KVCacheConfig of the runner
        # (e.g., KV sharing, encoder-only attention), but not in the
        # KVCacheConfig of the scheduler.
        self.runner_only_attn_layers: set[str] = set()

        # Cached outputs.
        self._draft_token_ids: Optional[Union[list[list[int]],
                                              torch.Tensor]] = None
        # Per-request draft probability distributions stashed after each
        # drafter call (TiDAR with T_Diff > 0 only). Keyed by req_id; each
        # value is a [num_speculative_tokens, vocab_size] float32 tensor.
        # Reassembled in input-batch order at the next step's input prep
        # and threaded through SpecDecodeMetadata.draft_probs.
        self._draft_probs_by_req_id: dict[str, torch.Tensor] = {}
        # Raw drafter logits stashed in parallel with draft_probs. Always
        # populated when the TiDAR drafter runs (even at T_Diff == 0 where
        # draft_probs is None). Threaded through
        # SpecDecodeMetadata.draft_logits so the mix-logit v1 sampler can
        # construct mixed_logits = w*target + (1-w)*draft without depending
        # on the (None-at-Dirac) draft_probs.
        self._draft_logits_by_req_id: dict[str, torch.Tensor] = {}
        self.transfer_event = torch.cuda.Event()
        self.sampled_token_ids_pinned_cpu = torch.empty(
            (self.max_model_len, 1),
            dtype=torch.int64,
            device="cpu",
            pin_memory=self.pin_memory)

    def _make_buffer(self,
                     *size: Union[int, torch.SymInt],
                     dtype: torch.dtype,
                     numpy: bool = True) -> CpuGpuBuffer:
        return CpuGpuBuffer(*size,
                            dtype=dtype,
                            device=self.device,
                            pin_memory=self.pin_memory,
                            with_numpy=numpy)

    def _init_model_kwargs(self, num_tokens: int):
        model_kwargs = dict[str, Any]()

        if not self.is_pooling_model:
            return model_kwargs

        num_reqs = self.input_batch.num_reqs
        pooling_params = self.input_batch.get_pooling_params()

        token_type_id_requests = dict[int, Any]()
        for i, param in enumerate(pooling_params):
            if param.extra_kwargs is not None and \
            (token_types := param.extra_kwargs.get(
                "compressed_token_type_ids")) is not None:
                token_type_id_requests[i] = token_types

        if len(token_type_id_requests) == 0:
            return model_kwargs

        seq_lens = self.seq_lens.gpu[:num_reqs]
        token_type_ids = []

        for i in range(num_reqs):
            pos = token_type_id_requests.get(i, seq_lens[i])
            ids = (torch.arange(seq_lens[i]) >= pos).int()
            token_type_ids.append(ids)

        model_kwargs["token_type_ids"] = torch.concat(token_type_ids).to(
            device=self.device)
        return model_kwargs

    def _may_reorder_batch(self, scheduler_output: "SchedulerOutput") -> None:
        """
        Update the order of requests in the batch based on the attention
        backend's needs. For example, some attention backends (namely MLA) may
        want to separate requests based on if the attention computation will be
        compute-bound or memory-bound.

        Args:
            scheduler_output: The scheduler output.
        """
        # Attention free models have zero kv_cache_goups, however models
        # like Mamba are also attention free but use the kv_cache for
        # keeping its internal state. This is why we check the number
        # of kv_cache groups instead of solely checking
        # for self.model_config.is_attention_free.
        if len(self.kv_cache_config.kv_cache_groups) == 0:
            return

        if self.reorder_batch_threshold is not None:
            # NOTE(lucas): currently no backend supports the custom masking
            #  required for DCP with q_len > 1, so we assert here. Remove this
            #  assert once the custom mask is support is added to FA3.
            if self.dcp_world_size > 1:
                assert self.reorder_batch_threshold == 1, \
                    "DCP not support reorder_batch_threshold > 1 now."
            reorder_batch_to_split_decodes_and_prefills(
                self.input_batch,
                scheduler_output,
                decode_threshold=self.reorder_batch_threshold)

    # Note: used for model runner override.
    def _init_device_properties(self) -> None:
        """Initialize attributes from torch.cuda.get_device_properties
        """
        self.device_properties = torch.cuda.get_device_properties(self.device)
        self.num_sms = self.device_properties.multi_processor_count

    # Note: used for model runner override.
    def _sync_device(self) -> None:
        torch.cuda.synchronize()

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        """Update the cached states and the persistent batch with the scheduler
        output.

        The updated states are used by the `_prepare_inputs` function to create
        the input GPU tensors for the model.

        The SamplingMetadata is updated and copied to the GPU if there is a
        new/resumed/paused/finished request in the batch.
        """
        # Remove finished requests from the cached states.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
        # Remove the finished requests from the persistent batch.
        # NOTE(woosuk): There could be an edge case where finished_req_ids and
        # scheduled_req_ids overlap. This happens when a request is aborted and
        # then resubmitted with the same ID. In this case, we treat them as two
        # distinct requests - clearing the cached states for the first request
        # and handling the second as a new request.
        for req_id in scheduler_output.finished_req_ids:
            self.input_batch.remove_request(req_id)

        # Free the cached encoder outputs.
        for mm_hash in scheduler_output.free_encoder_mm_hashes:
            self.encoder_cache.pop(mm_hash, None)

        # Remove the unscheduled requests from the persistent batch.
        # NOTE(woosuk): The unscheduled requests are either preempted requests
        # or running requests that are not scheduled in this step. We remove
        # them from the persistent batch but keep their cached states since
        # they will be scheduled again sometime in the future.
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        unscheduled_req_ids = cached_req_ids - scheduled_req_ids
        # NOTE(woosuk): The persistent batch optimization assumes that
        # consecutive batches contain mostly the same requests. If batches
        # have low request overlap (e.g., alternating between two distinct
        # sets of requests), this optimization becomes very inefficient.
        for req_id in unscheduled_req_ids:
            self.input_batch.remove_request(req_id)

        reqs_to_add: list[CachedRequestState] = []
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            sampling_params = new_req_data.sampling_params
            pooling_params = new_req_data.pooling_params

            if sampling_params and \
                sampling_params.sampling_type == SamplingType.RANDOM_SEED:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None

            if self.is_pooling_model:
                assert pooling_params is not None
                task = pooling_params.task
                assert task is not None, "You did not set `task` in the API"

                model = cast(VllmModelForPooling, self.get_model())
                to_update = model.pooler.get_pooling_updates(task)
                to_update.apply(pooling_params)

            req_state = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                prompt_embeds=new_req_data.prompt_embeds,
                mm_features=new_req_data.mm_features,
                sampling_params=sampling_params,
                pooling_params=pooling_params,
                generator=generator,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
            )
            self.requests[req_id] = req_state

            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            if self.uses_mrope:
                self._init_mrope_positions(req_state)

            reqs_to_add.append(req_state)

        # Update the states of the running/resumed requests.
        is_last_rank = get_pp_group().is_last_rank
        req_data = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(req_data.req_ids):
            req_state = self.requests[req_id]
            num_computed_tokens = req_data.num_computed_tokens[i]
            new_block_ids = req_data.new_block_ids[i]
            resumed_from_preemption = req_data.resumed_from_preemption[i]

            # Update the cached states.
            req_state.num_computed_tokens = num_computed_tokens

            if not is_last_rank:
                # When using PP, the scheduler sends the sampled tokens back,
                # because there's no direct communication between the first-
                # stage worker and the last-stage worker.
                new_token_ids = req_data.new_token_ids[i]
                # Add the sampled token(s) from the previous step (if any).
                # This doesn't include "unverified" tokens like spec tokens.
                num_new_tokens = (num_computed_tokens + len(new_token_ids) -
                                  req_state.num_tokens)
                if num_new_tokens == 1:
                    # Avoid slicing list in most common case.
                    req_state.output_token_ids.append(new_token_ids[-1])
                elif num_new_tokens > 0:
                    req_state.output_token_ids.extend(
                        new_token_ids[-num_new_tokens:])

            # Update the block IDs.
            if not resumed_from_preemption:
                if new_block_ids is not None:
                    # Append the new blocks to the existing block IDs.
                    for block_ids, new_ids in zip(req_state.block_ids,
                                                  new_block_ids):
                        block_ids.extend(new_ids)
            else:
                assert new_block_ids is not None
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                req_state.block_ids = new_block_ids

            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is None:
                # The request is not in the persistent batch.
                # The request was either preempted and resumed later, or was not
                # scheduled in the previous step and needs to be added again.
                reqs_to_add.append(req_state)
                continue

            # Update the persistent batch.
            self.input_batch.num_computed_tokens_cpu[req_index] = (
                num_computed_tokens)
            if new_block_ids is not None:
                self.input_batch.block_table.append_row(
                    new_block_ids, req_index)

            # For the last rank, we don't need to update the token_ids_cpu
            # because the sampled tokens are already cached.
            if not is_last_rank:
                # Add new_token_ids to token_ids_cpu.
                start_token_index = num_computed_tokens
                end_token_index = num_computed_tokens + len(new_token_ids)
                self.input_batch.token_ids_cpu[
                    req_index,
                    start_token_index:end_token_index] = new_token_ids
                self.input_batch.num_tokens_no_spec[
                    req_index] = end_token_index
                self.input_batch.num_tokens[req_index] = end_token_index

            # Add spec_token_ids to token_ids_cpu.
            spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id, ()))
            if spec_token_ids:
                num_spec_tokens = len(spec_token_ids)
                start_index = self.input_batch.num_tokens_no_spec[req_index]
                end_token_index = start_index + num_spec_tokens
                self.input_batch.token_ids_cpu[
                    req_index, start_index:end_token_index] = spec_token_ids
                # NOTE(woosuk): `num_tokens` here may include spec tokens.
                self.input_batch.num_tokens[req_index] += num_spec_tokens

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        for request in reqs_to_add:
            self.input_batch.add_request(request)

        # Condense the batched states if there are gaps left by removed requests
        self.input_batch.condense()
        # Allow attention backend to reorder the batch, potentially
        self._may_reorder_batch(scheduler_output)
        # Refresh batch metadata with any pending updates.
        self.input_batch.refresh_metadata()

    def _update_states_after_model_execute(
            self, output_token_ids: torch.Tensor) -> None:
        """Update the cached states after model execution.

        This is used for MTP/EAGLE for hybrid models, as in linear attention,
        only the last token's state is kept. In MTP/EAGLE, for draft tokens
        the state are kept util we decide how many tokens are accepted for
        each sequence, and a shifting is done during the next iteration
        based on the number of accepted tokens.
        """
        if not self.model_config.is_hybrid or not self.speculative_config:
            return

        # Find the number of accepted tokens for each sequence.
        num_accepted_tokens = (torch.cat(
            [
                output_token_ids,
                torch.full((output_token_ids.size(0), 1),
                           -1,
                           device=output_token_ids.device),
            ],
            dim=1) == -1).int().argmax(-1).cpu().numpy()
        for i, num_tokens in enumerate(num_accepted_tokens):
            self.input_batch.num_accepted_tokens_cpu[i] = num_tokens

    def _init_mrope_positions(self, req_state: CachedRequestState):
        image_grid_thw = []
        video_grid_thw = []
        second_per_grid_ts = []
        audio_feature_lengths = []
        use_audio_in_video = False
        for mm_feature in req_state.mm_features:
            mm_item = mm_feature.data
            if mm_item is None:
                continue
            mm_input = mm_item.get_data()
            if (t := mm_input.get("image_grid_thw")) is not None:
                image_grid_thw.append(t.tolist())
            if (t := mm_input.get("video_grid_thw")) is not None:
                video_grid_thw.append(t.tolist())
            if (t := mm_input.get("second_per_grid_ts")) is not None:
                second_per_grid_ts.append(t)
            if (t := mm_input.get("audio_feature_lengths")) is not None:
                audio_feature_lengths.append(t)
            if mm_input.get("use_audio_in_video") is True:
                use_audio_in_video = True

        if supports_mrope(self.model):
            req_state.mrope_positions, req_state.mrope_position_delta = \
                self.model.get_mrope_input_positions(
                    req_state.prompt_token_ids,
                    hf_config=self.model_config.hf_config,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    audio_feature_lengths=audio_feature_lengths,
                    use_audio_in_video=use_audio_in_video,
                )
        else:
            req_state.mrope_positions, req_state.mrope_position_delta = \
                MRotaryEmbedding.get_input_positions_tensor(
                    req_state.prompt_token_ids,
                    hf_config=self.model_config.hf_config,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    audio_feature_lengths=audio_feature_lengths,
                    use_audio_in_video=use_audio_in_video,
                )

    def _extract_mm_kwargs(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> BatchedTensorInputs:
        if not scheduler_output or not self.is_multimodal_raw_input_only_model:
            return {}

        mm_kwargs = list[MultiModalKwargsItem]()
        for req in scheduler_output.scheduled_new_reqs:
            for feature in req.mm_features:
                if feature.data is not None:
                    mm_kwargs.append(feature.data)

        # Input all modalities at once
        model = cast(SupportsMultiModal, self.model)
        mm_kwargs_combined: BatchedTensorInputs = {}
        for _, _, mm_kwargs_group in group_mm_kwargs_by_modality(
                mm_kwargs,
                device=self.device,
                pin_memory=self.pin_memory,
                merge_by_field_config=model.merge_by_field_config,
        ):
            mm_kwargs_combined.update(mm_kwargs_group)

        return mm_kwargs_combined

    def _dummy_mm_kwargs(self, num_seqs: int) -> BatchedTensorInputs:
        if not self.is_multimodal_raw_input_only_model:
            return {}

        mm_budget = self.mm_budget
        assert mm_budget is not None

        dummy_modality = mm_budget.get_modality_with_max_tokens()
        return self._get_mm_dummy_batch(dummy_modality, num_seqs)

    def _get_cumsum_and_arange(
        self,
        num_tokens: np.ndarray,
        cumsum_dtype: Optional[np.dtype] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the cumulative sum and batched arange of the given array.
        # E.g., [2, 5, 3] -> ([2, 7, 10], [0, 1, 0, 1, 2, 3, 4, 0, 1, 2])
        # Equivalent to but faster than:
        # np.concatenate([np.arange(n) for n in num_tokens])
        """
        # Step 1. [2, 5, 3] -> [2, 7, 10]
        cu_num_tokens = np.cumsum(num_tokens, dtype=cumsum_dtype)
        total_num_tokens = cu_num_tokens[-1]
        # Step 2. [2, 7, 10] -> [0, 0, 2, 2, 2, 2, 2, 7, 7, 7]
        cumsums_offsets = np.repeat(cu_num_tokens - num_tokens, num_tokens)
        # Step 3. [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        arange = self.arange_np[:total_num_tokens] - cumsums_offsets

        return cu_num_tokens, arange

    def _prepare_input_ids(self, total_num_scheduled_tokens: int,
                           cu_num_tokens: np.ndarray) -> None:
        """Prepare the input IDs for the current batch.
        
        Carefully handles the `prev_sampled_token_ids` which can be cached
        from the previous engine iteration, in which case those tokens on the
        GPU need to be copied into the corresponding slots into input_ids."""

        if self.input_batch.prev_sampled_token_ids is None:
            # Normal scheduling case
            self.input_ids.copy_to_gpu(total_num_scheduled_tokens)
            if self.enable_prompt_embeds:
                self.inputs_embeds.copy_to_gpu(total_num_scheduled_tokens)
                self.is_token_ids.copy_to_gpu(total_num_scheduled_tokens)
            return

        # Async scheduling case, where some decode requests from the previous
        # iteration won't have entries in input_ids_cpu and need to be copied
        # on the GPU from prev_sampled_token_ids.
        prev_req_id_to_index = self.input_batch.prev_req_id_to_index
        assert prev_req_id_to_index is not None
        flattened_indices = []
        prev_common_req_indices = []
        indices_match = True
        max_flattened_index = -1
        for req_id, cur_index in self.input_batch.req_id_to_index.items():
            if (prev_index := prev_req_id_to_index.get(req_id)) is not None:
                prev_common_req_indices.append(prev_index)
                # We need to compute the flattened input_ids index of the
                # last token in each common request.
                flattened_index = cu_num_tokens[cur_index].item() - 1
                flattened_indices.append(flattened_index)
                indices_match &= (prev_index == flattened_index)
                max_flattened_index = max(max_flattened_index, flattened_index)
        num_commmon_tokens = len(flattened_indices)
        if num_commmon_tokens < total_num_scheduled_tokens:
            # If not all requests are decodes from the last iteration,
            # We need to copy the input_ids_cpu to the GPU first.
            self.input_ids.copy_to_gpu(total_num_scheduled_tokens)
            if self.enable_prompt_embeds:
                self.inputs_embeds.copy_to_gpu(total_num_scheduled_tokens)
                self.is_token_ids.copy_to_gpu(total_num_scheduled_tokens)
        if num_commmon_tokens == 0:
            # No requests in common with the previous iteration
            # So input_ids_cpu will have all the input ids.
            return
        if indices_match and max_flattened_index == (num_commmon_tokens - 1):
            # Common-case optimization: the batch is unchanged
            # and no reordering happened.
            # The indices are both the same permutation of 0..N-1 so
            # we can copy directly using a single slice.
            self.input_ids.gpu[:num_commmon_tokens].copy_(
                self.input_batch.prev_sampled_token_ids[:num_commmon_tokens,
                                                        0],
                non_blocking=True)
            if self.enable_prompt_embeds:
                self.is_token_ids.gpu[:num_commmon_tokens] = True
            return
        # Upload the index tensors asynchronously
        # so the scatter can be non-blocking.
        input_ids_index_tensor = torch.tensor(flattened_indices,
                                              dtype=torch.int64,
                                              pin_memory=self.pin_memory).to(
                                                  self.device,
                                                  non_blocking=True)
        prev_common_req_indices_tensor = torch.tensor(
            prev_common_req_indices,
            dtype=torch.int64,
            pin_memory=self.pin_memory).to(self.device, non_blocking=True)
        self.input_ids.gpu.scatter_(
            dim=0,
            index=input_ids_index_tensor,
            src=self.input_batch.prev_sampled_token_ids[
                prev_common_req_indices_tensor, 0])

    def _get_encoder_seq_lens(
        self,
        scheduler_output: "SchedulerOutput",
        kv_cache_spec: KVCacheSpec,
        num_reqs: int,
    ) -> Optional[np.ndarray]:
        if not isinstance(kv_cache_spec, CrossAttentionSpec):
            return None

        # Build encoder_seq_lens array mapping request indices to
        # encoder lengths for inputs scheduled in this batch
        encoder_seq_lens = np.zeros(num_reqs, dtype=np.int32)
        for req_id in scheduler_output.scheduled_encoder_inputs:
            req_index = self.input_batch.req_id_to_index[req_id]
            encoder_seq_lens[req_index] = self.max_encoder_len

        return encoder_seq_lens

    def _prepare_inputs(
        self, scheduler_output: "SchedulerOutput"
    ) -> tuple[PerLayerAttnMetadata, torch.Tensor,
               Optional[SpecDecodeMetadata], np.ndarray,
               Optional[CommonAttentionMetadata], int, Optional[UBatchSlices],
               Optional[torch.Tensor]]:
        """
        :return: tuple[
            attn_metadata: layer-to-attention_metadata mapping,
            logits_indices, spec_decode_metadata
        ]
        """
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        self.input_batch.block_table.commit_block_table(num_reqs)

        # Get the number of scheduled tokens for each request.
        req_ids = self.input_batch.req_ids
        tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
        num_scheduled_tokens = np.array(tokens, dtype=np.int32)
        max_num_scheduled_tokens = max(tokens)

        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        req_indices = np.repeat(self.arange_np[:num_reqs],
                                num_scheduled_tokens)

        # cu_num_tokens: [2, 5, 3] -> [2, 7, 10]
        # arange: [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        cu_num_tokens, arange = self._get_cumsum_and_arange(
            num_scheduled_tokens)

        # Get positions.
        positions_np = self.positions.np[:total_num_scheduled_tokens]
        np.add(self.input_batch.num_computed_tokens_cpu[req_indices],
               arange,
               out=positions_np)

        # Calculate M-RoPE positions.
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            self._calc_mrope_positions(scheduler_output)

        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        token_indices = (positions_np +
                         req_indices * self.input_batch.token_ids_cpu.shape[1])
        token_indices_tensor = torch.from_numpy(token_indices)

        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        torch.index_select(self.input_batch.token_ids_cpu_tensor.flatten(),
                           0,
                           token_indices_tensor,
                           out=self.input_ids.cpu[:total_num_scheduled_tokens])
        if self.enable_prompt_embeds:
            is_token_ids = self.input_batch.is_token_ids.flatten()
            torch.index_select(
                is_token_ids,
                0,
                token_indices_tensor,
                out=self.is_token_ids.cpu[:total_num_scheduled_tokens])

        # Because we did not pre-allocate a massive prompt_embeds CPU tensor on
        # the InputBatch, we need to fill in the prompt embeds into the expected
        # spots in the GpuModelRunner's pre-allocated prompt_embeds tensor.
        if self.input_batch.req_prompt_embeds:
            output_idx = 0
            for req_idx in range(num_reqs):
                num_sched = num_scheduled_tokens[req_idx]

                # Skip if this request doesn't have embeddings
                if req_idx not in self.input_batch.req_prompt_embeds:
                    output_idx += num_sched
                    continue

                # Skip if no tokens scheduled
                if num_sched <= 0:
                    output_idx += num_sched
                    continue

                req_embeds = self.input_batch.req_prompt_embeds[req_idx]
                start_pos = self.input_batch.num_computed_tokens_cpu[req_idx]

                # Skip if trying to read beyond available embeddings
                if start_pos >= req_embeds.shape[0]:
                    output_idx += num_sched
                    continue

                # Copy available embeddings
                end_pos = start_pos + num_sched
                actual_end = min(end_pos, req_embeds.shape[0])
                actual_num_sched = actual_end - start_pos

                if actual_num_sched > 0:
                    self.inputs_embeds.cpu[output_idx:output_idx +
                                           actual_num_sched].copy_(
                                               req_embeds[start_pos:actual_end]
                                           )

                output_idx += num_sched

        self.input_batch.block_table.compute_slot_mapping(
            req_indices, positions_np)
        self.input_batch.block_table.commit_slot_mapping(
            total_num_scheduled_tokens)

        # Prepare the attention metadata.
        self.query_start_loc.np[0] = 0
        self.query_start_loc.np[1:num_reqs + 1] = cu_num_tokens
        # Note: pad query_start_loc to be non-decreasing, as kernels
        # like FlashAttention requires that
        self.query_start_loc.np[num_reqs + 1:].fill(cu_num_tokens[-1])
        self.query_start_loc.copy_to_gpu()
        query_start_loc = self.query_start_loc.gpu[:num_reqs + 1]

        num_tokens_unpadded = scheduler_output.total_num_scheduled_tokens
        num_tokens_padded = num_tokens_unpadded + self.get_local_padding(
            num_tokens_unpadded)
        uniform_decode = \
            (max_num_scheduled_tokens == self.uniform_decode_query_len) and \
            (total_num_scheduled_tokens == num_reqs * max_num_scheduled_tokens)
        ubatch_slices, num_tokens_after_padding = \
            ubatch_split(num_scheduled_tokens,
                         num_tokens_unpadded,
                         num_tokens_padded,
                         uniform_decode=uniform_decode,
                         vllm_config=self.vllm_config)

        self.seq_lens.np[:num_reqs] = (
            self.input_batch.num_computed_tokens_cpu[:num_reqs] +
            num_scheduled_tokens)
        # Fill unused with 0 for full cuda graph mode.
        self.seq_lens.np[num_reqs:].fill(0)
        self.seq_lens.copy_to_gpu()
        seq_lens = self.seq_lens.gpu[:num_reqs]
        max_seq_len = self.seq_lens.np[:num_reqs].max().item()

        num_tokens = [
            self.requests[r].num_tokens for r in self.input_batch.req_ids
        ]
        num_tokens_np = np.array(num_tokens, dtype=np.int32)

        # Record the index of requests that should not be sampled,
        # so that we could clear the sampled tokens before returning
        discard_requests_mask = self.seq_lens.np[:num_reqs] < num_tokens_np
        discard_request_indices = np.nonzero(discard_requests_mask)[0]
        self.num_discarded_requests = len(discard_request_indices)
        self.discard_request_indices.np[:self.num_discarded_requests] = (
            discard_request_indices)

        self.discard_request_indices.copy_to_gpu(self.num_discarded_requests)

        # Copy the tensors to the GPU.
        self._prepare_input_ids(total_num_scheduled_tokens, cu_num_tokens)

        if self.uses_mrope:
            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            self.mrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
                self.mrope_positions.cpu[:, :total_num_scheduled_tokens],
                non_blocking=True)
        else:
            # Common case (1D positions)
            self.positions.copy_to_gpu(total_num_scheduled_tokens)

        use_spec_decode = len(
            scheduler_output.scheduled_spec_decode_tokens) > 0
        if not use_spec_decode:
            # NOTE(woosuk): Due to chunked prefills, the batch may contain
            # partial requests. While we should not sample any token
            # from these partial requests, we do so for simplicity.
            # We will ignore the sampled tokens from the partial requests.
            # TODO: Support prompt logprobs.
            logits_indices = query_start_loc[1:] - 1
            num_draft_tokens = None
            spec_decode_metadata = None
        else:
            # Get the number of draft tokens for each request.
            # Iterate over the dictionary rather than all requests since not all
            # requests have draft tokens.
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
            # For chunked prefills, use -1 as mask rather than 0, as guided
            # decoding may rollback speculative tokens.
            num_decode_draft_tokens = np.full(num_reqs, -1, dtype=np.int32)
            for req_id, draft_token_ids in (
                    scheduler_output.scheduled_spec_decode_tokens.items()):
                req_idx = self.input_batch.req_id_to_index[req_id]
                num_draft_tokens[req_idx] = len(draft_token_ids)
                num_decode_draft_tokens[req_idx] = (len(draft_token_ids) if (
                    self.input_batch.num_computed_tokens_cpu[req_idx]
                    >= self.input_batch.num_prompt_tokens[req_idx]) else -1)
            spec_decode_metadata = self._calc_spec_decode_metadata(
                num_draft_tokens, cu_num_tokens)
            spec_decode_metadata.draft_probs = self._gather_draft_probs(
                num_draft_tokens)
            spec_decode_metadata.draft_logits = self._gather_draft_logits(
                num_draft_tokens)
            logits_indices = spec_decode_metadata.logits_indices

            # For DECODE only cuda graph of some attention backends (e.g., GDN).
            self.num_decode_draft_tokens.np[:
                                            num_reqs] = num_decode_draft_tokens
            self.num_decode_draft_tokens.np[num_reqs:].fill(-1)
            self.num_decode_draft_tokens.copy_to_gpu()

        logits_indices_padded = None
        if self.cache_config.kv_sharing_fast_prefill:
            logits_indices_padded = self._prepare_kv_sharing_fast_prefill(
                logits_indices)

        attn_metadata: PerLayerAttnMetadata = {}
        if ubatch_slices is not None:
            attn_metadata = [dict() for _ in range(len(ubatch_slices))]

        # Used in the below loop.
        query_start_loc_cpu = self.query_start_loc.cpu[:num_reqs + 1]
        seq_lens_cpu = self.seq_lens.cpu[:num_reqs]
        num_computed_tokens_cpu = (
            self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs])
        spec_decode_common_attn_metadata = None
        if use_spec_decode:
            self.num_accepted_tokens.np[:num_reqs] = (
                self.input_batch.num_accepted_tokens_cpu[:num_reqs])
            self.num_accepted_tokens.np[num_reqs:].fill(1)
            self.num_accepted_tokens.copy_to_gpu()

        # TiDAR single-forward (sparse-proposal) input extension. Called
        # ONCE per step, BEFORE the per-group attention metadata is built.
        # When ``single_forward_mode`` is off (default), this is a no-op
        # and ``total_num_scheduled_tokens`` is unchanged. When on:
        #   - Per-req input layout extends from K+1 -> K+1 + P*K
        #   - input_ids / positions / seq_lens / query_start_loc and the
        #     FA group's slot_mapping are rewritten in place
        # The new ``total_num_scheduled_tokens`` reflects the inflated
        # per-req length and propagates to every per-group metadata
        # built below. See docs/tidar_single_forward_design_2026-05-13.md
        # §6 (CCA path) and §3 (FlexAttention path).
        # TiDAR single-forward: clear any stale inflated-total cache
        # from the previous step before deciding whether to fire the
        # inflation. Bootstrap steps (no spec_decode_tokens) leave
        # this at None so ``_preprocess`` falls back to the scheduler's
        # value.
        if (self.speculative_config and self.speculative_config.use_tidar()
                and getattr(self.drafter, "single_forward_mode", False)):
            self.drafter.last_inflated_total = None
        # Inflation precondition: all reqs must be in the K+1 verify
        # state. Mixed batches (e.g., chunked prefill + spec-decode in
        # the same step) violate this, so fall back to the two-forward
        # propose() path for that step. Note we use verify_len (= K+1)
        # rather than self.uniform_decode_query_len here -- the latter
        # is overridden to K+1+P*K above for cudagraph capture sizing
        # and would never match the pre-inflation layout.
        _sf_verify_len = (
            self.drafter.verify_len
            if (self.speculative_config and self.speculative_config.use_tidar()
                and getattr(self.drafter, "single_forward_mode", False))
            else None)
        _sf_inflation_fired = False
        if (use_spec_decode and _sf_verify_len is not None
                and max_num_scheduled_tokens == _sf_verify_len
                and total_num_scheduled_tokens == num_reqs * _sf_verify_len):
            _sf_inflation_fired = True
            new_total = self.drafter.maybe_extend_verify_input(
                num_reqs=num_reqs,
                num_scheduled_tokens=total_num_scheduled_tokens,
            )
            total_num_scheduled_tokens = new_total
            # Re-read the now-inflated seq_lens for downstream metadata.
            seq_lens = self.seq_lens.gpu[:num_reqs]
            seq_lens_cpu = self.seq_lens.cpu[:num_reqs]
            query_start_loc = self.query_start_loc.gpu[:num_reqs + 1]
            query_start_loc_cpu = self.query_start_loc.cpu[:num_reqs + 1]
            max_seq_len = int(seq_lens_cpu.max().item())
            verify_len = self.drafter.verify_len
            P_props = len(self.drafter.proposal_acc_levels)
            K_drafts = self.drafter.num_speculative_tokens
            # ZAP-ONLY: K+1 proposal layout always.
            proposal_seg_len = K_drafts + 1
            new_stride = verify_len + P_props * proposal_seg_len
            max_num_scheduled_tokens = new_stride
            # Rewrite spec_decode_metadata.logits_indices: it points
            # into the FULL hidden_states tensor (which inflated from
            # B*verify_len to B*new_stride rows). Each OLD index ``j``
            # (in req ``j // verify_len``) shifts to
            # ``(j // verify_len) * new_stride + (j % verify_len)``.
            #
            # target_logits_indices and bonus_logits_indices are NOT
            # remapped: they point into the post-slicing logits tensor
            # ``logits = hidden_states[logits_indices]``, whose shape
            # (B*verify_len, V) is invariant under inflation. Remapping
            # them too caused illegal-memory-access at
            # ``logits[bonus_logits_indices]`` because the bonus
            # indices became larger than the post-slicing row count.
            if spec_decode_metadata is not None:
                old_idx = spec_decode_metadata.logits_indices
                req_idx = old_idx // verify_len
                within = old_idx % verify_len
                spec_decode_metadata.logits_indices = (
                    req_idx * new_stride + within)
                # Re-bind the runner-level logits_indices used to
                # gather sampling logits from the model output.
                logits_indices = spec_decode_metadata.logits_indices

        # Prepare the attention metadata for each KV cache group and make layers
        # in the same group share the same metadata.
        for kv_cache_group_id, kv_cache_group_spec in enumerate(
                self.kv_cache_config.kv_cache_groups):
            encoder_seq_lens = self._get_encoder_seq_lens(
                scheduler_output, kv_cache_group_spec.kv_cache_spec, num_reqs)

            if isinstance(kv_cache_group_spec.kv_cache_spec,
                          EncoderOnlyAttentionSpec):
                # Encoder-only layers do not have KV cache, so we need to
                # create a dummy block table and slot mapping for them.
                blk_table_tensor = torch.zeros(
                    (num_reqs, 1),
                    dtype=torch.int32,
                    device=self.device,
                )
                slot_mapping = torch.zeros(
                    (total_num_scheduled_tokens, ),
                    dtype=torch.int64,
                    device=self.device,
                )
                num_common_prefix_blocks = 0
            else:
                blk_table = self.input_batch.block_table[kv_cache_group_id]
                blk_table_tensor = blk_table.get_device_tensor(num_reqs)
                slot_mapping = blk_table.slot_mapping.gpu[:
                                                          total_num_scheduled_tokens]

                # Fill unused with -1. Needed for reshape_and_cache in full cuda
                # graph mode.
                blk_table.slot_mapping.gpu[total_num_scheduled_tokens:].fill_(
                    -1)
                num_common_prefix_blocks = (
                    scheduler_output.
                    num_common_prefix_blocks[kv_cache_group_id])

            common_attn_metadata = CommonAttentionMetadata(
                query_start_loc=query_start_loc,
                query_start_loc_cpu=query_start_loc_cpu,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu,
                num_computed_tokens_cpu=num_computed_tokens_cpu,
                num_reqs=num_reqs,
                num_actual_tokens=total_num_scheduled_tokens,
                max_query_len=max_num_scheduled_tokens,
                max_seq_len=max_seq_len,
                block_table_tensor=blk_table_tensor,
                slot_mapping=slot_mapping,
                logits_indices_padded=logits_indices_padded,
                num_logits_indices=logits_indices.size(0),
                causal=True,
                encoder_seq_lens=encoder_seq_lens,
            )
            # Signal CCA's metadata builder that this is a TiDAR verification
            # forward so its prefill loop defers conv_states / prev_hs writes
            # until rejection sampling tells us num_accepted per req. See
            # docs/diffusion_vllm_handoff.md.
            if (use_spec_decode and self.speculative_config
                    and self.speculative_config.use_tidar()):
                setattr(common_attn_metadata, "cca_spec_decode_mode", True)
                # Single-forward TiDAR: attach the three metadata fields
                # so the CCA + FlexAttention builders pick them up via
                # getattr. Only when inflation actually fired upstream --
                # otherwise the inputs are NOT in the K+1+P*K layout the
                # TiDAR mask_mod expects, and the kernel asserts OOB on
                # the proposal-index bound.
                if _sf_inflation_fired:
                    self.drafter.set_tidar_single_forward_metadata(
                        common_attn_metadata)

            if (self.speculative_config
                    and spec_decode_common_attn_metadata is None):
                if isinstance(self.drafter, EagleProposer):
                    if (self.drafter.attn_layer_names[0]
                            in kv_cache_group_spec.layer_names):
                        spec_decode_common_attn_metadata = common_attn_metadata
                else:
                    spec_decode_common_attn_metadata = common_attn_metadata

            for attn_group in self.attn_groups[kv_cache_group_id]:
                # Prepare for cascade attention if enabled & beneficial.
                common_prefix_len = 0
                builder = attn_group.get_metadata_builder()
                if self.cascade_attn_enabled:
                    common_prefix_len = self._compute_cascade_attn_prefix_len(
                        num_scheduled_tokens,
                        num_common_prefix_blocks,
                        attn_group.kv_cache_spec,
                        builder,
                    )

                extra_attn_metadata_args = {}
                if use_spec_decode and isinstance(builder,
                                                  GDNAttentionMetadataBuilder):
                    extra_attn_metadata_args = dict(
                        num_accepted_tokens=self.num_accepted_tokens.
                        gpu[:num_reqs],
                        num_decode_draft_tokens_cpu=self.
                        num_decode_draft_tokens.cpu[:num_reqs],
                    )

                if ubatch_slices is not None:
                    common_attn_metadata_list = split_attn_metadata(
                        ubatch_slices, common_attn_metadata)
                    for ubid, common_attn_metadata in enumerate(
                            common_attn_metadata_list):
                        attn_metadata_i = (attn_group.get_metadata_builder(
                            ubatch_id=ubid).build(
                                common_prefix_len=common_prefix_len,
                                common_attn_metadata=common_attn_metadata))
                        for layer_name in kv_cache_group_spec.layer_names:
                            assert type(attn_metadata) is list
                            attn_metadata[ubid][layer_name] = attn_metadata_i
                else:
                    assert isinstance(attn_metadata, dict)
                    attn_metadata_i = builder.build(
                        common_prefix_len=common_prefix_len,
                        common_attn_metadata=common_attn_metadata,
                        **extra_attn_metadata_args)
                    for layer_name in attn_group.layer_names:
                        attn_metadata[layer_name] = attn_metadata_i

        # Hot-Swap lora model
        if self.lora_config:
            self.set_active_loras(self.input_batch, num_scheduled_tokens)

        return (attn_metadata, logits_indices, spec_decode_metadata,
                num_scheduled_tokens, spec_decode_common_attn_metadata,
                max_num_scheduled_tokens, ubatch_slices,
                num_tokens_after_padding)

    def _compute_cascade_attn_prefix_len(
        self,
        num_scheduled_tokens: np.ndarray,
        num_common_prefix_blocks: int,
        kv_cache_spec: KVCacheSpec,
        attn_metadata_builder: AttentionMetadataBuilder,
    ) -> int:
        """Compute the length of the common prefix for cascade attention.

        NOTE(woosuk): The common prefix length returned by this function
        represents the length used specifically for cascade attention, not the
        actual number of tokens shared between requests. When cascade attention
        is disabled (use_cascade=False), this function returns 0 even if
        requests share common tokens. Additionally, the common prefix length is
        truncated to a multiple of the block size and may be further truncated
        due to implementation details explained below.

        Args:
            num_scheduled_tokens: Number of tokens scheduled per request.
            num_common_prefix_blocks: Number of shared KV cache blocks.

        Returns:
            int: Length of common prefix in tokens.
        """
        common_prefix_len = num_common_prefix_blocks * kv_cache_spec.block_size
        if common_prefix_len == 0:
            # Common case.
            return 0

        # NOTE(woosuk): Cascade attention uses two attention kernels: one
        # for the common prefix and the other for the rest. For the first
        # kernel, we concatenate all the query tokens (possibly from
        # different requests) and treat them as if they are from the same
        # request. Then, we use bi-directional attention to process the
        # common prefix in the KV cache. Importantly, this means that the
        # first kernel does not do any masking.

        # Consider the following example:
        # Request 1's input query: [D, E, X]
        # Request 1's kv cache: [A, B, C, D, E, X]
        # Request 1's num_computed_tokens: 3 (i.e., [A, B, C])
        # Request 2's input query: [E, Y]
        # Request 2's kv cache: [A, B, C, D, E, Y]
        # Request 2's num_computed_tokens: 4 (i.e., [A, B, C, D])

        # If we use [A, B, C, D, E] as the common prefix, then the
        # first kernel will compute the bi-directional attention between
        # input query [D, E, X, E, Y] and common prefix [A, B, C, D, E].
        # However, this is wrong because D in Request 1 should not attend to
        # E in the common prefix (i.e., we need masking).
        # To avoid this, [A, B, C, D] should be the common prefix.
        # That is, the common prefix should be capped by the minimum
        # num_computed_tokens among the requests, and plus one to include
        # the first token of the query.

        # In practice, we use [A, B, C] as the common prefix, instead of
        # [A, B, C, D] (i.e., the common prefix is capped by the minimum
        # num_computed_tokens, without plus one).
        # This is because of an implementation detail: We want to always
        # use two kernels for cascade attention. Let's imagine:
        # Request 3's input query: [D]
        # Request 3's kv cache: [A, B, C, D]
        # Request 3's num_computed_tokens: 3 (i.e., [A, B, C])
        # If we use [A, B, C, D] as the common prefix for Request 1-3,
        # then Request 3 will be processed only by the first kernel,
        # and the second kernel will get an empty input. While this is not
        # a fundamental problem, our current implementation does not support
        # this case.
        num_reqs = len(num_scheduled_tokens)
        common_prefix_len = min(
            common_prefix_len,
            self.input_batch.num_computed_tokens_cpu[:num_reqs].min())
        # common_prefix_len should be a multiple of the block size.
        common_prefix_len = (common_prefix_len // kv_cache_spec.block_size *
                             kv_cache_spec.block_size)
        use_sliding_window = (isinstance(kv_cache_spec, SlidingWindowSpec) or
                              (isinstance(kv_cache_spec, FullAttentionSpec)
                               and kv_cache_spec.sliding_window is not None))
        use_local_attention = (
            isinstance(kv_cache_spec, ChunkedLocalAttentionSpec)
            or (isinstance(kv_cache_spec, FullAttentionSpec)
                and kv_cache_spec.attention_chunk_size is not None))
        assert isinstance(kv_cache_spec, AttentionSpec)
        use_cascade = attn_metadata_builder.use_cascade_attention(
            common_prefix_len=common_prefix_len,
            query_lens=num_scheduled_tokens,
            num_query_heads=self.num_query_heads,
            num_kv_heads=kv_cache_spec.num_kv_heads,
            use_alibi=self.use_alibi,
            use_sliding_window=use_sliding_window,
            use_local_attention=use_local_attention,
            num_sms=self.num_sms,
        )
        return common_prefix_len if use_cascade else 0

    def _calc_mrope_positions(self, scheduler_output: "SchedulerOutput"):
        mrope_pos_ptr = 0
        for index, req_id in enumerate(self.input_batch.req_ids):
            req = self.requests[req_id]
            assert req.mrope_positions is not None

            num_computed_tokens = \
                self.input_batch.num_computed_tokens_cpu[index]
            num_scheduled_tokens = \
                scheduler_output.num_scheduled_tokens[req_id]
            num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
                req.prompt_token_ids, req.prompt_embeds)

            if num_computed_tokens + num_scheduled_tokens > num_prompt_tokens:
                prompt_part_len = max(0,
                                      num_prompt_tokens - num_computed_tokens)
                completion_part_len = max(
                    0, num_scheduled_tokens - prompt_part_len)
            else:
                prompt_part_len = num_scheduled_tokens
                completion_part_len = 0

            assert num_scheduled_tokens == prompt_part_len + completion_part_len

            if prompt_part_len > 0:
                # prompt's mrope_positions are pre-computed
                dst_start = mrope_pos_ptr
                dst_end = mrope_pos_ptr + prompt_part_len
                src_start = num_computed_tokens
                src_end = num_computed_tokens + prompt_part_len

                self.mrope_positions.cpu[:, dst_start:dst_end] = (
                    req.mrope_positions[:, src_start:src_end])
                mrope_pos_ptr += prompt_part_len

            if completion_part_len > 0:
                # compute completion's mrope_positions on-the-fly
                dst_start = mrope_pos_ptr
                dst_end = mrope_pos_ptr + completion_part_len

                MRotaryEmbedding.get_next_input_positions_tensor(
                    out=self.mrope_positions.np,
                    out_offset=dst_start,
                    mrope_position_delta=req.mrope_position_delta,
                    context_len=num_computed_tokens + prompt_part_len,
                    num_new_tokens=completion_part_len,
                )

                mrope_pos_ptr += completion_part_len

    def _calc_spec_decode_metadata(
        self,
        num_draft_tokens: np.ndarray,
        cu_num_scheduled_tokens: np.ndarray,
    ) -> SpecDecodeMetadata:
        # Inputs:
        # cu_num_scheduled_tokens:  [  4, 104, 107, 207, 209]
        # num_draft_tokens:         [  3,   0,   2,   0,   1]
        # Outputs:
        # cu_num_draft_tokens:      [  3,   3,   5,   5,   6]
        # logits_indices:           [  0,   1,   2,   3, 103, 104, 105, 106,
        #                            206, 207, 208]
        # target_logits_indices:    [  0,   1,   2,   5,   6,   9]
        # bonus_logits_indices:     [  3,   4,   7,   8,  10]

        # Compute the logits indices.
        # [4, 1, 3, 1, 2]
        num_sampled_tokens = num_draft_tokens + 1

        # Step 1. cu_num_sampled_tokens: [4, 5, 8, 9, 11]
        # arange: [0, 1, 2, 3, 0, 0, 1, 2, 0, 0, 1]
        cu_num_sampled_tokens, arange = self._get_cumsum_and_arange(
            num_sampled_tokens, cumsum_dtype=np.int32)
        # Step 2. [0, 0, 0, 0, 103, 104, 104, 104, 206, 207, 207]
        logits_indices = np.repeat(
            cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens)
        # Step 3. [0, 1, 2, 3, 103, 104, 105, 106, 206, 207, 208]
        logits_indices += arange

        # Compute the bonus logits indices.
        bonus_logits_indices = cu_num_sampled_tokens - 1

        # Compute the draft logits indices.
        # cu_num_draft_tokens: [3, 3, 5, 5, 6]
        # arange: [0, 1, 2, 0, 1, 0]
        cu_num_draft_tokens, arange = self._get_cumsum_and_arange(
            num_draft_tokens, cumsum_dtype=np.int32)
        # [0, 0, 0, 5, 5, 9]
        target_logits_indices = np.repeat(
            cu_num_sampled_tokens - num_sampled_tokens, num_draft_tokens)
        # [0, 1, 2, 5, 6, 9]
        target_logits_indices += arange

        # TODO: Optimize the CPU -> GPU copy.
        cu_num_draft_tokens = torch.from_numpy(cu_num_draft_tokens).to(
            self.device, non_blocking=True)
        logits_indices = torch.from_numpy(logits_indices).to(self.device,
                                                             non_blocking=True)
        target_logits_indices = torch.from_numpy(target_logits_indices).to(
            self.device, non_blocking=True)
        bonus_logits_indices = torch.from_numpy(bonus_logits_indices).to(
            self.device, non_blocking=True)

        # Compute the draft token ids.
        # draft_token_indices:      [  1,   2,   3, 105, 106, 208]
        draft_token_ids = self.input_ids.gpu[logits_indices]
        draft_token_ids = draft_token_ids[target_logits_indices + 1]

        metadata = SpecDecodeMetadata(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens.tolist(),
            cu_num_draft_tokens=cu_num_draft_tokens,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
        )
        return metadata

    def _gather_draft_logits(
        self,
        num_draft_tokens: np.ndarray,
    ) -> Optional[torch.Tensor]:
        """Parallel to _gather_draft_probs but for raw drafter logits."""
        if not hasattr(self, '_gather_dbg_steps'):
            self._gather_dbg_steps = 0
        if self._gather_dbg_steps < 3:
            self._gather_dbg_steps += 1
            print(f"[GATHER_DBG step={self._gather_dbg_steps}] dict_keys={len(self._draft_logits_by_req_id)} num_draft={list(num_draft_tokens)[:4]}", flush=True)
        if not self._draft_logits_by_req_id:
            return None
        chunks: list[torch.Tensor] = []
        for i, req_id in enumerate(self.input_batch.req_ids):
            n = int(num_draft_tokens[i])
            if n == 0:
                continue
            logits_i = self._draft_logits_by_req_id.get(req_id)
            if logits_i is None:
                return None
            chunks.append(logits_i[:n])
        if not chunks:
            return None
        return torch.cat(chunks, dim=0).contiguous()

    def _gather_draft_probs(
        self,
        num_draft_tokens: np.ndarray,
    ) -> Optional[torch.Tensor]:
        # Concatenate per-req draft_probs (stashed by the previous step's
        # drafter call) in current input-batch order, sliced to the actual
        # num_draft_tokens for this step. Returns None if no probs were
        # stashed (e.g., T_Diff == 0 on TiDAR, or any non-TiDAR drafter).
        if not self._draft_probs_by_req_id:
            return None
        chunks: list[torch.Tensor] = []
        for i, req_id in enumerate(self.input_batch.req_ids):
            n = int(num_draft_tokens[i])
            if n == 0:
                continue
            probs = self._draft_probs_by_req_id.get(req_id)
            if probs is None:
                # Defensive: a req scheduled for spec-decode verification
                # without a stash entry indicates a state mismatch. Bail to
                # NO_DRAFT_PROBS rather than silently zero-fill.
                return None
            chunks.append(probs[:n])
        if not chunks:
            return None
        return torch.cat(chunks, dim=0).contiguous()

    def _gather_draft_logits(
        self,
        num_draft_tokens: np.ndarray,
    ) -> Optional[torch.Tensor]:
        # Parallel to _gather_draft_probs but for raw drafter logits
        # (pre-softmax, pre-temperature). Used by mix-logit v1 to construct
        # mixed_logits = w*target + (1-w)*draft without depending on
        # draft_probs (which is None at T_Diff == 0).
        if not self._draft_logits_by_req_id:
            return None
        chunks: list[torch.Tensor] = []
        for i, req_id in enumerate(self.input_batch.req_ids):
            n = int(num_draft_tokens[i])
            if n == 0:
                continue
            logits_i = self._draft_logits_by_req_id.get(req_id)
            if logits_i is None:
                return None
            chunks.append(logits_i[:n])
        if not chunks:
            return None
        return torch.cat(chunks, dim=0).contiguous()

    def _commit_tidar_cca_state(
        self,
        valid_sampled_token_ids: list[list[int]],
    ) -> None:
        """Tell each CCA layer which post-acceptance candidate state to write.

        For each TiDAR-verified request, the rejection sampler emitted
        `num_accepted + 1` tokens (accepted drafts + bonus/recovered). The
        CCA layer's prefill loop appended its deferred-stash entries in batch
        order; here we just compute num_accepted per req in the same order
        and pass it in. Each CCA layer carries its own slot_idx (kept as a
        0-dim tensor — no CPU-GPU sync per req).
        """
        from vllm.config import get_layers_from_vllm_config
        from vllm.model_executor.layers.mamba.cca import CCA
        if not hasattr(self, "_cca_layers_cache"):
            self._cca_layers_cache = get_layers_from_vllm_config(
                self.vllm_config, CCA)
        if not self._cca_layers_cache:
            return
        # Pass num_accepted in input-batch order; CCA layers index into this
        # by the batch_idx they stashed during the deferred prefill loop.
        # For non-deferring reqs the entry is unused (len-1 still defined,
        # just ignored).
        num_accepted_per_batch_idx = [
            max(0, len(tokens) - 1) for tokens in valid_sampled_token_ids
        ]
        if not num_accepted_per_batch_idx:
            return
        for cca_layer in self._cca_layers_cache.values():
            cca_layer.commit_spec_decode_state(num_accepted_per_batch_idx)

    def _prepare_kv_sharing_fast_prefill(
        self,
        logits_indices: torch.Tensor,
    ) -> torch.Tensor:
        assert self.kv_sharing_fast_prefill_logits_indices is not None
        num_logits = logits_indices.shape[0]
        assert num_logits > 0
        self.kv_sharing_fast_prefill_logits_indices[:num_logits].copy_(
            logits_indices)
        # There might have leftover indices in logits_indices[num_logits:]
        # from previous iterations, whose values may be greater than the
        # batch size in the current iteration. To ensure indices are always
        # valid, we fill the padded indices with the last index.
        self.kv_sharing_fast_prefill_logits_indices[num_logits:].fill_(
            logits_indices[-1].item())
        if (self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
                and num_logits <= self.cudagraph_batch_sizes[-1]):
            # Use piecewise CUDA graphs.
            # Add padding to the batch size.
            num_logits_padded = self.vllm_config.pad_for_cudagraph(num_logits)
        else:
            num_logits_padded = num_logits
        logits_indices_padded = (
            self.kv_sharing_fast_prefill_logits_indices[:num_logits_padded])
        return logits_indices_padded

    def _batch_mm_kwargs_from_scheduler(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> tuple[list[MultiModalKwargsItem], list[tuple[str, PlaceholderRange]]]:
        """Batch multimodal kwargs from scheduled encoder inputs.

        Args:
            scheduler_output: The scheduler output containing scheduled encoder
                inputs.

        Returns:
            A tuple of (mm_kwargs, req_ids_pos) where:
            - mm_kwargs: List of multimodal kwargs items to be batched
            - mm_hashes_pos: List of (mm_hash, position_info) tuples
        """
        scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs
        if not scheduled_encoder_inputs:
            return [], []
        # Batch the multi-modal inputs.
        mm_kwargs = list[MultiModalKwargsItem]()
        # list of tuple (mm_hash, position_info)
        mm_hashes_pos = list[tuple[str, PlaceholderRange]]()
        for req_id, encoder_input_ids in scheduled_encoder_inputs.items():
            req_state = self.requests[req_id]

            for mm_input_id in encoder_input_ids:
                mm_feature = req_state.mm_features[mm_input_id]
                mm_hash = mm_feature.identifier
                mm_kwargs.append(mm_feature.data)
                mm_hashes_pos.append((mm_hash, mm_feature.mm_position))

        return mm_kwargs, mm_hashes_pos

    def _execute_mm_encoder(self, scheduler_output: "SchedulerOutput"):
        # Batch the multi-modal inputs using the helper method.
        mm_kwargs, mm_hashes_pos = self._batch_mm_kwargs_from_scheduler(
            scheduler_output)

        if not mm_kwargs:
            return

        # Batch mm inputs as much as we can: if a request in the batch has
        # multiple modalities or a different modality than the previous one,
        # we process it separately to preserve item order.
        # FIXME(ywang96): This is a hacky way to deal with multiple modalities
        # in the same batch while still being able to benefit from batching
        # multimodal inputs. The proper solution should be reordering the
        # encoder outputs.
        model = cast(SupportsMultiModal, self.model)
        encoder_outputs = []
        for modality, num_items, mm_kwargs_group in group_mm_kwargs_by_modality(
                mm_kwargs,
                device=self.device,
                pin_memory=self.pin_memory,
                merge_by_field_config=model.merge_by_field_config,
        ):
            # (ekhvedchenia): Temporary hack to limit peak memory usage when
            # processing multimodal data.This solves the issue with scheduler
            # putting too many video samples into a single batch. Scheduler
            # uses pruned vision tokens count to compare it versus compute
            # budget which is incorrect (Either input media size or non-pruned
            # output vision tokens count should be considered)
            curr_group_outputs = []

            if self.is_multimodal_pruning_enabled and modality == "video":
                micro_batch_size = 1
                for i in range(0, num_items, micro_batch_size):
                    micro_batch_mm_inputs = dict(
                        (k, v[i:i + micro_batch_size])
                        for k, v in mm_kwargs_group.items())

                    micro_batch_outputs = model.get_multimodal_embeddings(
                        **micro_batch_mm_inputs)

                    curr_group_outputs.extend(micro_batch_outputs)
            else:
                # Run the encoder.
                # `curr_group_outputs` is either of the following:
                # 1. A tensor of shape (num_items, feature_size, hidden_size)
                # in case feature_size is fixed across all multimodal items.
                # 2. A list or tuple (length: num_items) of tensors,
                # each of shape (feature_size, hidden_size) in case the feature
                # size is dynamic depending on the input multimodal items.
                curr_group_outputs = model.get_multimodal_embeddings(
                    **mm_kwargs_group)

            sanity_check_mm_encoder_outputs(
                curr_group_outputs,
                expected_num_items=num_items,
            )
            encoder_outputs.extend(curr_group_outputs)

        # Cache the encoder outputs by mm_hash
        for (mm_hash, pos_info), output in zip(mm_hashes_pos, encoder_outputs):
            self.encoder_cache[mm_hash] = scatter_mm_placeholders(
                output,
                is_embed=pos_info.is_embed,
            )

    def _gather_mm_embeddings(
        self,
        scheduler_output: "SchedulerOutput",
        shift_computed_tokens: int = 0,
    ) -> list[torch.Tensor]:
        should_sync_mrope_positions = False
        mm_embeds: list[torch.Tensor] = []
        for req_id in self.input_batch.req_ids:
            mm_embeds_req: list[torch.Tensor] = []

            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[
                req_id]
            req_state = self.requests[req_id]
            num_computed_tokens = \
                req_state.num_computed_tokens + shift_computed_tokens
            for mm_feature in req_state.mm_features:
                pos_info = mm_feature.mm_position
                start_pos = pos_info.offset
                num_encoder_tokens = pos_info.length

                # The encoder output is needed if the two ranges overlap:
                # [num_computed_tokens,
                #  num_computed_tokens + num_scheduled_tokens) and
                # [start_pos, start_pos + num_encoder_tokens)
                if start_pos >= num_computed_tokens + num_scheduled_tokens:
                    # The encoder output is not needed in this step.
                    break
                if start_pos + num_encoder_tokens <= num_computed_tokens:
                    # The encoder output is already processed and stored
                    # in the decoder's KV cache.
                    continue

                start_idx = max(num_computed_tokens - start_pos, 0)
                end_idx = min(
                    num_computed_tokens - start_pos + num_scheduled_tokens,
                    num_encoder_tokens,
                )
                assert start_idx < end_idx

                mm_hash = mm_feature.identifier
                encoder_output = self.encoder_cache.get(mm_hash, None)
                assert encoder_output is not None,\
                    f"Encoder cache miss for {mm_hash}."

                if (is_embed := pos_info.is_embed) is not None:
                    is_embed = is_embed[start_idx:end_idx]

                mm_embeds_item = gather_mm_placeholders(
                    encoder_output[start_idx:end_idx],
                    is_embed=is_embed,
                )
                mm_embeds_req.append(mm_embeds_item)

            if self.is_multimodal_pruning_enabled and self.uses_mrope:
                should_sync_mrope_positions = True
                mm_embeds_req, new_mrope_positions, new_delta = (
                    self.model.recompute_mrope_positions(
                        input_ids=req_state.prompt_token_ids,
                        multimodal_embeddings=mm_embeds_req,
                        mrope_positions=req_state.mrope_positions,
                        num_computed_tokens=req_state.num_computed_tokens,
                    ))
                assert req_state.mrope_positions is not None
                req_state.mrope_positions.copy_(new_mrope_positions)
                req_state.mrope_position_delta = new_delta

            mm_embeds.extend(mm_embeds_req)

        if should_sync_mrope_positions:
            self._calc_mrope_positions(scheduler_output)
            self.mrope_positions.copy_to_gpu(
                scheduler_output.total_num_scheduled_tokens)

        return mm_embeds

    def _extract_encoder_inputs(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> dict[str, torch.Tensor]:
        """Extract encoder inputs for encoder-decoder models.

        This method extracts multimodal input features from scheduled encoder
        inputs and formats them for the encoder-decoder model forward pass.
        """
        # Batch the multi-modal inputs using the helper method.
        mm_kwargs, _ = self._batch_mm_kwargs_from_scheduler(scheduler_output)

        if not mm_kwargs:
            return {}

        # Group MM kwargs by modality and extract features
        model = cast(SupportsMultiModal, self.model)
        encoder_features = {}
        for _, _, mm_kwargs_group in group_mm_kwargs_by_modality(
                mm_kwargs,
                device=self.device,
                pin_memory=self.pin_memory,
                merge_by_field_config=model.merge_by_field_config,
        ):
            # Add the grouped features to encoder_features dict
            # This allows the model to receive them as kwargs (e.g.,
            # input_features=...)
            encoder_features.update(mm_kwargs_group)

        return encoder_features

    def get_model(self) -> nn.Module:
        # get raw model out of the cudagraph wrapper.
        if isinstance(self.model, (CUDAGraphWrapper, UBatchWrapper)):
            return self.model.unwrap()
        return self.model

    def get_supported_generation_tasks(self) -> list[GenerationTask]:
        model = self.get_model()
        supported_tasks = list[GenerationTask]()

        if is_text_generation_model(model):
            supported_tasks.append("generate")

        if supports_transcription(model):
            if model.supports_transcription_only:
                return ["transcription"]

            supported_tasks.append("transcription")

        return supported_tasks

    def get_supported_pooling_tasks(self) -> list[PoolingTask]:
        model = self.get_model()
        if not is_pooling_model(model):
            return []

        supported_tasks = list(model.pooler.get_supported_tasks())

        if (self.scheduler_config.chunked_prefill_enabled
                and "encode" in supported_tasks):
            supported_tasks.remove("encode")

            logger.debug_once("Chunked prefill is not supported with "
                              "encode task which using ALL pooling. "
                              "Please turn off chunked prefill by "
                              "`--no-enable-chunked-prefill` before using it.")

        if "score" in supported_tasks:
            num_labels = getattr(self.model_config.hf_config, "num_labels", 0)
            if num_labels != 1:
                supported_tasks.remove("score")
                logger.debug_once(
                    "Score API is only enabled for num_labels == 1.")

        return supported_tasks

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        tasks = list[SupportedTask]()

        if self.model_config.runner_type == "generate":
            tasks.extend(self.get_supported_generation_tasks())
        if self.model_config.runner_type == "pooling":
            tasks.extend(self.get_supported_pooling_tasks())

        return tuple(tasks)

    def sync_and_slice_intermediate_tensors(
            self, num_tokens: int, intermediate_tensors: IntermediateTensors,
            sync_self: bool) -> IntermediateTensors:

        assert self.intermediate_tensors is not None

        tp = self.vllm_config.parallel_config.tensor_parallel_size
        is_rs = is_residual_scattered_for_sp(self.vllm_config, num_tokens)

        # When sequence parallelism is enabled, the "residual" tensor is sharded
        # across tensor parallel ranks, so each rank only needs its own slice.
        if sync_self:
            assert intermediate_tensors is not None
            for k, v in intermediate_tensors.items():
                is_scattered = k == "residual" and is_rs
                copy_len = num_tokens // tp if is_scattered else \
                    num_tokens
                self.intermediate_tensors[k][:copy_len].copy_(
                    v[:copy_len], non_blocking=True)

        return IntermediateTensors({
            k:
            v[:num_tokens //
              tp] if k == "residual" and is_rs else v[:num_tokens]
            for k, v in self.intermediate_tensors.items()
        })

    def eplb_step(self,
                  is_dummy: bool = False,
                  is_profile: bool = False) -> None:
        """
        Step for the EPLB (Expert Parallelism Load Balancing) state.
        """
        if not self.parallel_config.enable_eplb:
            return

        assert self.eplb_state is not None
        model = self.get_model()
        assert is_mixture_of_experts(model)
        self.eplb_state.step(
            model,
            is_dummy,
            is_profile,
            log_stats=self.parallel_config.eplb_config.log_balancedness,
        )

    def get_dp_padding(self,
                       num_tokens: int) -> tuple[int, Optional[torch.Tensor]]:
        """
        Determines the total number of tokens that each rank will run.
        All ranks will be padded out so that they run with the same number
        of tokens

        Returns: tuple[
            num_pad_tokens: The number of tokens that will be added to the batch
            num_tokens_after_padding: A tensor containing the total number of
            tokens for each DP rank including padding.
        ]
        """
        dp_size = self.vllm_config.parallel_config.data_parallel_size
        dp_rank = self.vllm_config.parallel_config.data_parallel_rank

        # For DP: Don't pad when setting enforce_eager.
        # This lets us set enforce_eager on the prefiller in a P/D setup and
        # still use CUDA graphs (enabled by this padding) on the decoder.
        #
        # TODO(tms) : There are many cases where padding is enabled for
        # prefills, causing unnecessary and excessive padding of activations.

        if dp_size == 1 or self.vllm_config.model_config.enforce_eager:
            # Early exit.
            return 0, None

        num_tokens_across_dp = DPMetadata.num_tokens_across_dp(
            num_tokens, dp_size, dp_rank)
        max_tokens_across_dp_cpu = torch.max(num_tokens_across_dp).item()
        num_tokens_after_padding = torch.tensor([max_tokens_across_dp_cpu] *
                                                dp_size,
                                                device="cpu",
                                                dtype=torch.int32)
        return max_tokens_across_dp_cpu - num_tokens, num_tokens_after_padding

    def get_local_padding(self, num_tokens_unpadded: int) -> int:

        num_tokens_padded = num_tokens_unpadded

        if (self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
                and num_tokens_unpadded <= self.cudagraph_batch_sizes[-1]):
            # Use piecewise CUDA graphs.
            # Add padding to the batch size.
            num_tokens_padded = self.vllm_config.pad_for_cudagraph(
                num_tokens_unpadded)
        else:
            # Eager mode.
            # Pad tokens to multiple of tensor_parallel_size when
            # enabled collective fusion for SP
            tp_size = self.vllm_config.parallel_config.tensor_parallel_size
            if self.vllm_config.compilation_config.pass_config. \
                enable_sequence_parallelism and tp_size > 1:
                num_tokens_padded = round_up(num_tokens_unpadded, tp_size)

        num_pad_tokens = num_tokens_padded - num_tokens_unpadded
        return num_pad_tokens

    # This is where the second ubatch is adjusted to account for the padding.
    # Should be called after attention metadata creation. This just pads
    # the second ubatch slice out to the total number of tokens
    # (num_tokens + padding)
    def pad_out_ubatch_slice(self, ubatch_slices: UBatchSlices,
                             num_total_tokens: int):
        padded_second_ubatch_slice = slice(ubatch_slices[1].token_slice.start,
                                           num_total_tokens)
        ubatch_slices[1] = UBatchSlice(padded_second_ubatch_slice,
                                       padded_second_ubatch_slice)

    def _pool(
        self,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: int,
        num_scheduled_tokens_np: np.ndarray,
    ) -> ModelRunnerOutput:
        assert self.input_batch.num_reqs ==\
            len(self.input_batch.pooling_params), \
        "Either all or none of the requests in" \
        " a batch must be pooling request"

        hidden_states = hidden_states[:num_scheduled_tokens]
        pooling_metadata = self.input_batch.get_pooling_metadata()
        pooling_metadata.build_pooling_cursor(num_scheduled_tokens_np.tolist(),
                                              device=hidden_states.device)
        seq_lens_cpu = self.seq_lens.cpu[:self.input_batch.num_reqs]

        model = cast(VllmModelForPooling, self.model)
        raw_pooler_output: PoolerOutput = model.pooler(
            hidden_states=hidden_states,
            pooling_metadata=pooling_metadata,
        )
        raw_pooler_output = json_map_leaves(
            lambda x: x.to("cpu", non_blocking=True),
            raw_pooler_output,
        )
        self._sync_device()

        pooler_output: list[Optional[torch.Tensor]] = []
        for raw_output, seq_len, prompt_len in zip(
                raw_pooler_output, seq_lens_cpu, pooling_metadata.prompt_lens):

            output = raw_output if seq_len == prompt_len else None
            pooler_output.append(output)

        return ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=[],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=pooler_output,
        )

    def _get_num_input_tokens(self, num_scheduled_tokens: int) -> int:
        if (self.compilation_config.cudagraph_mode != CUDAGraphMode.NONE
                and not envs.VLLM_DISABLE_PAD_FOR_CUDAGRAPH
                and hasattr(self, "cudagraph_batch_sizes")
                and self.cudagraph_batch_sizes
                and num_scheduled_tokens <= self.cudagraph_batch_sizes[-1]):
            # Use CUDA graphs.
            # Add padding to the batch size.
            return self.vllm_config.pad_for_cudagraph(num_scheduled_tokens)

        # Eager mode.
        # Pad tokens to multiple of tensor_parallel_size when
        # enabled collective fusion for SP
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        if (self.compilation_config.pass_config.enable_sequence_parallelism
                and tp_size > 1):
            return round_up(num_scheduled_tokens, tp_size)
        return num_scheduled_tokens

    def _preprocess(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
        ubatch_slices: Optional[UBatchSlices] = None,
        num_tokens_after_padding: Optional[torch.Tensor] = None,
    ) -> tuple[int, int, Optional[torch.Tensor], Optional[torch.Tensor],
               Optional[torch.Tensor], torch.Tensor,
               Optional[IntermediateTensors], dict[str, Any]]:

        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        # TiDAR single-forward: ``maybe_extend_verify_input`` inflates
        # the verify layout by P*K per request and caches the new
        # total on ``self.drafter.last_inflated_total``. The forward
        # below needs to slice input_ids / positions to the inflated
        # size (else flex_attention's block_mask shape mismatches).
        if (self.speculative_config
                and self.speculative_config.use_tidar()
                and getattr(self.drafter, "single_forward_mode", False)
                and getattr(self.drafter, "last_inflated_total", None)
                is not None):
            num_scheduled_tokens = self.drafter.last_inflated_total
        if ubatch_slices:
            assert num_tokens_after_padding is not None
            num_input_tokens = int(num_tokens_after_padding[0].item() * 2)
            self.pad_out_ubatch_slice(ubatch_slices, num_input_tokens)
        elif ubatch_slices is None:
            num_input_tokens = self._get_num_input_tokens(num_scheduled_tokens)
            num_pad, num_tokens_after_padding = self.get_dp_padding(
                num_input_tokens)
            num_input_tokens += num_pad

        # _prepare_inputs may reorder the batch, so we must gather multi
        # modal outputs after that to ensure the correct order
        if (self.supports_mm_inputs and get_pp_group().is_first_rank
                and not self.model_config.is_encoder_decoder):
            # Run the multimodal encoder if any.
            self._execute_mm_encoder(scheduler_output)
            mm_embeds = self._gather_mm_embeddings(scheduler_output)

            # NOTE(woosuk): To unify token ids and soft tokens (vision
            # embeddings), we always use embeddings (rather than token ids)
            # as input to the multimodal model, even when the input is text.
            inputs_embeds_scheduled = self.model.get_input_embeddings(
                input_ids=self.input_ids.gpu[:num_scheduled_tokens],
                multimodal_embeddings=mm_embeds or None,
            )

            # TODO(woosuk): Avoid the copy. Optimize.
            self.inputs_embeds.gpu[:num_scheduled_tokens].copy_(
                inputs_embeds_scheduled)

            input_ids = None
            inputs_embeds = self.inputs_embeds.gpu[:num_input_tokens]
            model_kwargs = {
                **self._init_model_kwargs(num_scheduled_tokens),
                **self._extract_mm_kwargs(scheduler_output),
            }
        elif self.enable_prompt_embeds and get_pp_group().is_first_rank:
            # Get the input embeddings for the tokens that are not input embeds,
            # then put them into the appropriate positions.
            # TODO(qthequartermasterman): Since even when prompt embeds are
            # enabled, (a) not all requests will use prompt embeds, and (b)
            # after the initial prompt is processed, the rest of the generated
            # tokens will be token ids, it is not desirable to have the
            # embedding layer outside of the CUDA graph all the time. The v0
            # engine avoids this by "double compiling" the CUDA graph, once
            # with input_ids and again with inputs_embeds, for all num_tokens.
            # If a batch only has token ids, then including the embedding layer
            # in the CUDA graph will be more performant (like in the else case
            # below).
            token_ids_idx = self.is_token_ids.gpu[:num_scheduled_tokens] \
                .nonzero(as_tuple=False) \
                .squeeze(1)
            # Some tokens ids may need to become embeds
            if token_ids_idx.numel() > 0:
                token_ids = self.input_ids.gpu[token_ids_idx]
                tokens_to_embeds = self.model.get_input_embeddings(
                    input_ids=token_ids)
                self.inputs_embeds.gpu[token_ids_idx] = tokens_to_embeds

            inputs_embeds = self.inputs_embeds.gpu[:num_input_tokens]
            model_kwargs = self._init_model_kwargs(num_input_tokens)
            input_ids = None
        else:
            # For text-only models, we use token ids as input.
            # While it is possible to use embeddings as input just like the
            # multimodal models, it is not desirable for performance since
            # then the embedding layer is not included in the CUDA graph.
            input_ids = self.input_ids.gpu[:num_input_tokens]
            inputs_embeds = None
            model_kwargs = self._init_model_kwargs(num_input_tokens)
        if self.uses_mrope:
            positions = self.mrope_positions.gpu[:, :num_input_tokens]
        else:
            positions = self.positions.gpu[:num_input_tokens]

        if get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                num_input_tokens, intermediate_tensors, True)

        if (self.model_config.is_encoder_decoder
                and scheduler_output.scheduled_encoder_inputs):
            encoder_inputs = self._extract_encoder_inputs(scheduler_output)
            model_kwargs.update(encoder_inputs)

        return (
            num_scheduled_tokens,
            num_input_tokens,
            num_tokens_after_padding,
            input_ids,
            inputs_embeds,
            positions,
            intermediate_tensors,
            model_kwargs,
        )

    def _sample(
            self, logits: Optional[torch.Tensor],
            spec_decode_metadata: Optional[SpecDecodeMetadata]
    ) -> SamplerOutput:
        # Sample the next token and get logprobs if needed.
        sampling_metadata = self.input_batch.sampling_metadata
        if spec_decode_metadata is None:
            sampler_output = self.sampler(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )
        else:
            # When indexing with a tensor (bonus_logits_indices), PyTorch
            # creates a new tensor with separate storage from the original
            # logits tensor. This means any in-place operations on bonus_logits
            # won't affect the original logits tensor.
            assert logits is not None
            # Per-TOKEN broadening samplers (apply to ALL spec positions
            # before bonus-sampling and rejection-sampling).
            #   TIDAR_MIN_ENTROPY_THRESHOLD: for each logit row, compute
            #     h_min = -log2(max(softmax(logits, T=1))). Broaden if
            #     h_min > threshold (default 1.0 bit, i.e. max_p < 0.5).
            #   TIDAR_ENTROPY_THRESHOLD: for each logit row, compute Shannon
            #     entropy H = -sum(p * log_2 p) (in BITS) of
            #     softmax(logits, T=1). Broaden if H > threshold
            #     (default 1.0 bit).
            # When a row is "broadened", scale that row's logits by T_AR so
            # the subsequent T_AR division yields a T=1 distribution. Affects
            # BOTH the bonus token and the K target tokens (rejection sampler
            # input). Only one of these two env vars should be set at a time.
            import os, dataclasses
            _met = os.environ.get("TIDAR_MIN_ENTROPY_THRESHOLD")
            _et = os.environ.get("TIDAR_ENTROPY_THRESHOLD")
            # v2 variants: per-token broadening on TARGET rows only; bonus
            # forced to T=1 unconditionally. Implemented via the existing
            # scaling trick on target rows + a bonus-temperature override
            # branch below.
            _met_v2 = os.environ.get("TIDAR_MIN_ENTROPY_V2_THRESHOLD")
            _et_v2 = os.environ.get("TIDAR_ENTROPY_V2_THRESHOLD")
            # Top-p v2 sampler:
            #   TIDAR_TOP_P_V2=p: nucleus filter applied per-token to target
            #   rows at T_AR + bonus rows at T=1. Tokens outside the smallest
            #   set with cumulative prob >= p have their logits set to -inf,
            #   so the rejection sampler treats them as p_target=0 (any
            #   draft token outside the nucleus is auto-rejected -> redraft).
            #   Bonus T=1 forcing reuses the v2 branch below.
            _topp_v2 = os.environ.get("TIDAR_TOP_P_V2")
            _v2_active = ((_met_v2 is not None) or (_et_v2 is not None)
                          or (_topp_v2 is not None))
            _met_eff = _met_v2 if _met_v2 is not None else _met
            _et_eff = _et_v2 if _et_v2 is not None else _et
            _temp = sampling_metadata.temperature
            if (_met_eff is not None or _et_eff is not None) and _temp is not None:
                _probs_t1 = logits.softmax(dim=-1, dtype=torch.float32)
                if _met_eff is not None:
                    _thr = float(_met_eff) if _met_eff.strip() else 1.0
                    _h_per_row = -(_probs_t1.amax(dim=-1)
                                   .clamp_min(1e-12)).log2()
                else:
                    _thr = float(_et_eff) if _et_eff.strip() else 1.0
                    _h_per_row = -(_probs_t1
                                   * (_probs_t1.clamp_min(1e-12)).log2()
                                   ).sum(dim=-1)
                _broaden_mask = _h_per_row > _thr
                # Optional per-token entropy logger (writes JSONL with both
                # Shannon entropy H and min-entropy Hm per row, keyed by
                # req_id + spec-step + pos in {0..K} where pos=K is bonus).
                _pertok_log = os.environ.get("TIDAR_PERTOK_ENTROPY_LOG")
                if _pertok_log:
                    try:
                        import json as _json
                        # Always compute both metrics for logging (regardless
                        # of which one drives broadening).
                        _Hm_all = -(_probs_t1.amax(dim=-1)
                                    .clamp_min(1e-12)).log2()
                        _H_all = -(_probs_t1
                                   * (_probs_t1.clamp_min(1e-12)).log2()
                                   ).sum(dim=-1)
                        _Hm_cpu = _Hm_all.detach().cpu().tolist()
                        _H_cpu = _H_all.detach().cpu().tolist()
                        _b_cpu = (spec_decode_metadata.bonus_logits_indices
                                  .detach().cpu().tolist())
                        _t_cpu = (spec_decode_metadata.target_logits_indices
                                  .detach().cpu().tolist())
                        _K_pl = len(_t_cpu) // max(len(_b_cpu), 1)
                        _rids = list(self.input_batch.req_ids)
                        if not hasattr(self, "_pertok_log_step"):
                            self._pertok_log_step = 0
                        if not hasattr(self, "_pertok_log_fh"):
                            self._pertok_log_fh = open(_pertok_log, "a",
                                                       buffering=64 * 1024)
                        for _r_pl in range(len(_b_cpu)):
                            _rid_pl = (_rids[_r_pl]
                                       if _r_pl < len(_rids)
                                       else "req_" + str(_r_pl))
                            for _k_pl in range(_K_pl):
                                _row_pl = _t_cpu[_r_pl * _K_pl + _k_pl]
                                self._pertok_log_fh.write(_json.dumps({
                                    "rid": _rid_pl,
                                    "step": self._pertok_log_step,
                                    "pos": _k_pl,
                                    "H": _H_cpu[_row_pl],
                                    "Hm": _Hm_cpu[_row_pl],
                                }) + chr(10))
                            _rb = _b_cpu[_r_pl]
                            self._pertok_log_fh.write(_json.dumps({
                                "rid": _rid_pl,
                                "step": self._pertok_log_step,
                                "pos": _K_pl,
                                "H": _H_cpu[_rb],
                                "Hm": _Hm_cpu[_rb],
                            }) + chr(10))
                        self._pertok_log_step += 1
                    except Exception as _e_pl:
                        # never crash inference on logger error
                        if not hasattr(self, "_pertok_warned"):
                            print("[pertok_log] disabled after error:",
                                  type(_e_pl).__name__, _e_pl, flush=True)
                            self._pertok_warned = True
                if _v2_active:
                    # v2: don't scale bonus rows here; bonus is forced to T=1
                    # below via the bonus-temperature override branch.
                    _b_idx_v2 = spec_decode_metadata.bonus_logits_indices
                    if _b_idx_v2 is not None:
                        _broaden_mask[_b_idx_v2] = False
                if _broaden_mask.any():
                    # Build per-row T_AR. _temp is per-request; spec decode
                    # has 1 bonus row + K target rows per request. Map via
                    # the *_logits_indices to populate per-row temperatures.
                    _t_per_row = torch.ones(logits.shape[0],
                                            device=logits.device,
                                            dtype=logits.dtype)
                    _b_idx = spec_decode_metadata.bonus_logits_indices
                    _t_idx = spec_decode_metadata.target_logits_indices
                    if _b_idx is not None:
                        _t_per_row[_b_idx] = _temp.to(logits.dtype)
                    if _t_idx is not None and _b_idx is not None:
                        _K = _t_idx.numel() // _b_idx.numel()
                        _t_per_row[_t_idx] = (
                            _temp.to(logits.dtype).repeat_interleave(_K))
                    _scale = torch.where(_broaden_mask,
                                         _t_per_row.clamp_min(1e-3),
                                         torch.ones_like(_t_per_row))
                    logits = logits * _scale.unsqueeze(-1)
            if _topp_v2 is not None and _temp is not None:
                # Nucleus mask in vocab space: keep the smallest set of
                # tokens whose cumulative probability under the effective
                # per-row temperature reaches _p_thr. Set everything else
                # to -inf so the downstream rejection sampler and bonus
                # sampler both see zero mass on dropped tokens.
                _p_thr = float(_topp_v2)
                _b_idx_p = spec_decode_metadata.bonus_logits_indices
                _t_idx_p = spec_decode_metadata.target_logits_indices
                _t_per_row_p = torch.ones(logits.shape[0],
                                          device=logits.device,
                                          dtype=logits.dtype)
                if _t_idx_p is not None and _b_idx_p is not None:
                    _K_p = _t_idx_p.numel() // max(_b_idx_p.numel(), 1)
                    _t_per_row_p[_t_idx_p] = (
                        _temp.to(logits.dtype).repeat_interleave(_K_p))
                _scaled = (logits.float()
                           / _t_per_row_p.clamp_min(1e-3)
                                          .unsqueeze(-1).float())
                _probs_p = _scaled.softmax(dim=-1)
                _sorted_p, _sorted_idx_p = _probs_p.sort(dim=-1,
                                                        descending=True)
                _cum_p = _sorted_p.cumsum(dim=-1)
                # Drop tokens whose *prefix* cumsum already >= threshold
                # (keeps the first token that crosses, drops the rest).
                _drop_sorted = (_cum_p - _sorted_p) >= _p_thr
                _drop_mask = torch.zeros_like(_drop_sorted).scatter_(
                    -1, _sorted_idx_p, _drop_sorted)
                logits = logits.masked_fill(_drop_mask, float("-inf"))
            bonus_logits = logits[spec_decode_metadata.bonus_logits_indices]
            # Bonus-token sampling overrides via env vars (apply only to the
            # K+1-th "bonus" token, AFTER any min-entropy scaling above):
            #   TIDAR_BONUS_TEMPERATURE: sample bonus at this fixed temperature
            #     for ALL requests (uniform override).
            #   TIDAR_BONUS_BROAD_THRESHOLD: per-request conditional override.
            #     For each request, compute max(softmax(bonus_logits, T=1)).
            #     If < threshold (broad distribution), sample THAT request's
            #     bonus at T=1; otherwise keep T_AR.
            #   TIDAR_BONUS_ENTROPY_THRESHOLD: per-request conditional override
            #     ("Entropy-Sampler"). For each request, compute Shannon
            #     entropy H = -sum(p * log_2 p) (in BITS) of softmax(bonus_logits, T=1).
            #     If H > threshold (high-uncertainty distribution), sample
            #     THAT request's bonus at T=1; otherwise keep T_AR.
            #     Default h = 1.0 bit.
            _bt = os.environ.get("TIDAR_BONUS_TEMPERATURE")
            _bbt = os.environ.get("TIDAR_BONUS_BROAD_THRESHOLD")
            _bet = os.environ.get("TIDAR_BONUS_ENTROPY_THRESHOLD")
            if _v2_active and _temp is not None:
                # v2: force bonus to T=1 unconditionally (regardless of any
                # per-token entropy on the bonus row, which we already
                # excluded from broadening above so bonus_logits is unscaled).
                _bonus_temp_v2 = torch.ones_like(_temp)
                _bonus_meta = dataclasses.replace(
                    sampling_metadata,
                    temperature=_bonus_temp_v2,
                    all_greedy=False,
                    all_random=True,
                )
                sampler_output = self.sampler(
                    logits=bonus_logits,
                    sampling_metadata=_bonus_meta,
                )
            elif _bbt is not None and _temp is not None:
                _bbt_val = float(_bbt)
                _bonus_probs_t1 = bonus_logits.softmax(dim=-1, dtype=torch.float32)
                _max_probs_t1 = _bonus_probs_t1.amax(dim=-1)
                _broad_mask = _max_probs_t1 < _bbt_val
                _new_temp = torch.where(
                    _broad_mask,
                    torch.ones_like(_temp),
                    _temp,
                )
                _bonus_meta = dataclasses.replace(
                    sampling_metadata,
                    temperature=_new_temp,
                    all_greedy=False,
                    all_random=True,
                )
                sampler_output = self.sampler(
                    logits=bonus_logits,
                    sampling_metadata=_bonus_meta,
                )
            elif _bet is not None and _temp is not None:
                # Entropy-Sampler: broaden bonus to T=1 when its T=1 softmax
                # has Shannon entropy (in BITS, log_2) above the threshold.
                _bet_val = float(_bet) if _bet.strip() else 1.0
                _bonus_probs_t1 = bonus_logits.softmax(dim=-1, dtype=torch.float32)
                _entropy = -(_bonus_probs_t1
                             * (_bonus_probs_t1.clamp_min(1e-12)).log2()
                             ).sum(dim=-1)
                _broad_mask = _entropy > _bet_val
                _new_temp = torch.where(
                    _broad_mask,
                    torch.ones_like(_temp),
                    _temp,
                )
                _bonus_meta = dataclasses.replace(
                    sampling_metadata,
                    temperature=_new_temp,
                    all_greedy=False,
                    all_random=True,
                )
                sampler_output = self.sampler(
                    logits=bonus_logits,
                    sampling_metadata=_bonus_meta,
                )
            elif _bt is not None and _temp is not None:
                _bt_val = float(_bt)
                _bonus_temp = torch.full_like(_temp, _bt_val)
                _bonus_meta = dataclasses.replace(
                    sampling_metadata,
                    temperature=_bonus_temp,
                    all_greedy=(_bt_val < 1e-5),
                    all_random=(_bt_val >= 1e-5),
                )
                sampler_output = self.sampler(
                    logits=bonus_logits,
                    sampling_metadata=_bonus_meta,
                )
            else:
                sampler_output = self.sampler(
                    logits=bonus_logits,
                    sampling_metadata=sampling_metadata,
                )
            bonus_token_ids = sampler_output.sampled_token_ids

            # Just like `bonus_logits`, `target_logits` is a new tensor with
            # separate storage from the original `logits` tensor. Therefore,
            # it is safe to update `target_logits` in place.
            target_logits = logits[spec_decode_metadata.target_logits_indices]
            # Mix Draft+Target sampler (direct-sample, NO rejection):
            #   TIDAR_MIX_DRAFT_TARGET=1: for each K target row, build
            #   mixed_logits = 0.5 * (target_logits + log(draft_probs)),
            #   sample one token at per-row T_AR, and concatenate the K
            #   mix-sampled tokens with the bonus token. Bypasses the
            #   rejection sampler entirely (every position is emitted; no
            #   accept/reject events). No-op at T_AR=0 because
            #   `draft_probs=None` on the greedy path.
            # Mix Draft+Target sampler (v1, rejection-based):
            #   TIDAR_MIX_DRAFT_TARGET_V1=1: build mixed_logits the same way
            #   as v2, but pass them as the rejection-sampler target so the
            #   draft is still tested against mixed_probs. Preserves a
            #   meaningful acceptance metric. draft_probs is passed UNCHANGED
            #   to the sampler — at T_diff=0 (Dirac) that means None, which
            #   routes to the rejection_sampler's NO_DRAFT_PROBS branch.
            #   DO NOT synthesize a soft q from draft_logits here — see
            #   commit 583479480 on jinzhao/tidar_TF (fixes the over-accept
            #   short-circuit that biases accepts toward the drafter argmax
            #   independently of TIDAR_MIX_LOGIT_TARGET_WEIGHT).
            # Drafts-only sampler (no verification):
            #   TIDAR_DRAFTS_ONLY=1: commit all K draft tokens as the
            #   output, plus 1 bonus from AR. Ignores target_logits at
            #   the K positions entirely. Useful as a "drafter alone"
            #   eval. Requires t_diff>0 for sample diversity.
            _drafts_only = os.environ.get("TIDAR_DRAFTS_ONLY")
            # AR-only sampler (no verification):
            #   TIDAR_AR_ONLY=1: drafter still runs (KV prefill), but
            #   the K target positions are sampled directly from
            #   target_logits at per-row T_AR. Bonus token unchanged.
            #   Same context-mismatch issue as v2 mix-logit at k>0.
            _mix_dt = os.environ.get("TIDAR_MIX_DRAFT_TARGET")
            _v1_w_runtime = None
            try:
                with open("/tmp/tidar_mix_w") as _fh:
                    _v1_w_runtime = float(_fh.read().strip())
            except Exception:
                pass
            if _v1_w_runtime is not None:
                _mix_dt_v1 = "1" if _v1_w_runtime > 0 else None
            else:
                _mix_dt_v1 = os.environ.get("TIDAR_MIX_DRAFT_TARGET_V1")
            _ar_only = os.environ.get("TIDAR_AR_ONLY")
            if (_ar_only is not None
                    and _ar_only not in ("", "0", "false", "False")
                    and spec_decode_metadata.target_logits_indices is not None
                    and sampling_metadata.temperature is not None
                    and spec_decode_metadata.bonus_logits_indices.numel() > 0):
                _b_idx_a = spec_decode_metadata.bonus_logits_indices
                _t_idx_a = spec_decode_metadata.target_logits_indices
                _num_reqs_a = _b_idx_a.numel()
                _K_a = _t_idx_a.numel() // max(_num_reqs_a, 1)
                _temp_a = (sampling_metadata.temperature
                           .to(target_logits.dtype)
                           .repeat_interleave(_K_a))
                _ar_meta = dataclasses.replace(
                    sampling_metadata,
                    temperature=_temp_a,
                    all_greedy=False,
                    all_random=True,
                )
                _ar_out = self.sampler(
                    logits=target_logits,
                    sampling_metadata=_ar_meta,
                )
                _ar_per_req = _ar_out.sampled_token_ids.view(_num_reqs_a, _K_a)
                output_token_ids = torch.cat(
                    [_ar_per_req, bonus_token_ids.view(_num_reqs_a, 1)],
                    dim=1)
                # === AR-ONLY DEBUG ===
                if not hasattr(self, '_ar_debug_steps'):
                    self._ar_debug_steps = 0
                if self._ar_debug_steps < 3:
                    self._ar_debug_steps += 1
                    print(f"[AR_DEBUG step={self._ar_debug_steps}] shape={output_token_ids.shape}", flush=True)
                # === END AR-ONLY DEBUG ===
            elif (_drafts_only is not None
                    and _drafts_only not in ("", "0", "false", "False")
                    and spec_decode_metadata.draft_token_ids is not None):
                _b_idx_d = spec_decode_metadata.bonus_logits_indices
                _t_idx_d = spec_decode_metadata.target_logits_indices
                _num_reqs_d = _b_idx_d.numel()
                _K_d = _t_idx_d.numel() // max(_num_reqs_d, 1)
                _drafts_per_req = (spec_decode_metadata.draft_token_ids
                                   .view(_num_reqs_d, _K_d))
                output_token_ids = torch.cat(
                    [_drafts_per_req, bonus_token_ids.view(_num_reqs_d, 1)],
                    dim=1)
            elif (_mix_dt_v1 is not None
                    and _mix_dt_v1 not in ("", "0", "false", "False")
                    and spec_decode_metadata.draft_logits is not None
                    and sampling_metadata.temperature is not None
                    and spec_decode_metadata.bonus_logits_indices is not None
                    and spec_decode_metadata.bonus_logits_indices.numel() > 0):
                _draft_logits_v1 = spec_decode_metadata.draft_logits.to(target_logits.dtype)
                # draft_probs faithfully reflects what the drafter sampled from:
                #   T_diff>0 → softmax(draft_logits/T_diff) (set upstream in tidar.py)
                #   T_diff=0 → None (Dirac drafter; rejection sampler must take
                #              its NO_DRAFT_PROBS branch). DO NOT synthesize a
                #              soft q here — that would lie about the drafter’s
                #              distribution and over-accept the draft argmax via
                #              the min(1, p/q) short-circuit. See 583479480.
                _v1_dp = spec_decode_metadata.draft_probs
                # Configurable target/draft weight (default 0.5 ≡ even mix).
                # TIDAR_MIX_LOGIT_TARGET_WEIGHT=w in [0,1] → w*target + (1-w)*draft.
                _v1_w = (_v1_w_runtime if _v1_w_runtime is not None else
                         float(os.environ.get("TIDAR_MIX_LOGIT_TARGET_WEIGHT", "0.5")))
                target_logits = _v1_w * target_logits + (1.0 - _v1_w) * _draft_logits_v1
                output_token_ids = self.rejection_sampler(
                    spec_decode_metadata,
                    _v1_dp,
                    target_logits,
                    bonus_token_ids,
                    sampling_metadata,
                )
            elif (_mix_dt is not None
                    and _mix_dt not in ("", "0", "false", "False")
                    and spec_decode_metadata.draft_logits is not None
                    and sampling_metadata.temperature is not None
                    and spec_decode_metadata.bonus_logits_indices is not None
                    and spec_decode_metadata.bonus_logits_indices.numel() > 0):
                # mix-logit v2: use raw drafter logits.
                # mixed = 0.5 * (target_logits + drafter_logits), then sample
                # at T_AR. Mathematically identical to softmax→log→mix but
                # skips two large ops per spec step.
                _draft_logits = spec_decode_metadata.draft_logits
                _mixed_logits = 0.5 * (target_logits + _draft_logits.to(target_logits.dtype))
                # Per-row T_AR: temperature is per-request; target has K rows
                # per request via target_logits_indices.
                _b_idx_m = spec_decode_metadata.bonus_logits_indices
                _t_idx_m = spec_decode_metadata.target_logits_indices
                _K_m = _t_idx_m.numel() // max(_b_idx_m.numel(), 1)
                _temp_per_target = (sampling_metadata.temperature
                                    .to(target_logits.dtype)
                                    .repeat_interleave(_K_m))
                _mix_meta = dataclasses.replace(
                    sampling_metadata,
                    temperature=_temp_per_target,
                    all_greedy=False,
                    all_random=True,
                )
                _mix_out = self.sampler(
                    logits=_mixed_logits,
                    sampling_metadata=_mix_meta,
                )
                _mix_target_ids = _mix_out.sampled_token_ids
                _num_reqs = _b_idx_m.numel()
                _mix_per_req = _mix_target_ids.view(_num_reqs, _K_m)
                output_token_ids = torch.cat(
                    [_mix_per_req, bonus_token_ids.view(_num_reqs, 1)], dim=1)
                # === V2 DEBUG INSTRUMENTATION ===
                if not hasattr(self, '_v2_debug_steps'):
                    self._v2_debug_steps = 0
                if self._v2_debug_steps < 3:
                    self._v2_debug_steps += 1
                    try:
                        _dft = spec_decode_metadata.draft_token_ids.view(_num_reqs, _K_m)
                        print(f"[V2_DEBUG step={self._v2_debug_steps}] shape={output_token_ids.shape}", flush=True)
                        for _i in range(min(2, _num_reqs)):
                            print(f"  req {_i} draft tokens: {_dft[_i].tolist()}", flush=True)
                            print(f"  req {_i} v2 output:    {output_token_ids[_i].tolist()}", flush=True)
                            _matches = (_dft[_i] == output_token_ids[_i, :_K_m]).sum().item()
                            print(f"  req {_i} matches: {_matches}/{_K_m}", flush=True)
                    except Exception as _e:
                        print(f"[V2_DEBUG] err: {_e}", flush=True)
                # === END V2 DEBUG ===
            else:
                output_token_ids = self.rejection_sampler(
                    spec_decode_metadata,
                    spec_decode_metadata.draft_probs,
                    target_logits,
                    bonus_token_ids,
                    sampling_metadata,
                )

            # SF K-mask + no-bonus path (from jinzhao/tidar_SF): the triton
            # rejection kernels always write bonus_token_id at column
            # num_draft_tokens on all-accept. Under VLLM_TIDAR_NO_BONUS=1 the
            # SF K-mask layout has no bonus position, so zap that slot to
            # PLACEHOLDER_TOKEN_ID — parse_output() filters it. Applies
            # uniformly across all branches above (rejection-sampler and
            # direct-sample alike) because each branch produces output of
            # shape [B, K+1] with the bonus in the last column.
            import os as _os
            if (self.speculative_config is not None
                    and self.speculative_config.use_tidar()
                    and getattr(self, "drafter", None) is not None
                    and _os.environ.get("VLLM_TIDAR_NO_BONUS", "0") == "1"):
                from vllm.v1.sample.rejection_sampler import (
                    PLACEHOLDER_TOKEN_ID,
                )
                output_token_ids[:, -1] = PLACEHOLDER_TOKEN_ID
            sampler_output.sampled_token_ids = output_token_ids
            self._update_states_after_model_execute(output_token_ids)

        return sampler_output

    def _bookkeeping_sync(
        self, scheduler_output: "SchedulerOutput",
        sampler_output: SamplerOutput, logits: Optional[torch.Tensor],
        hidden_states: torch.Tensor, num_scheduled_tokens: int
    ) -> tuple[
            dict[str, int],
            Optional[LogprobsLists],
            list[list[int]],
            dict[str, Optional[LogprobsTensors]],
            list[str],
            dict[str, int],
            list[int],
    ]:
        num_nans_in_logits = {}
        if envs.VLLM_COMPUTE_NANS_IN_LOGITS:
            num_nans_in_logits = self._get_nans_in_logits(logits)

        discard_sampled_tokens_req_indices = \
            self.discard_request_indices.np[:self.num_discarded_requests]
        for i in discard_sampled_tokens_req_indices:
            gen = self.input_batch.generators.get(int(i))
            if gen is not None:
                gen.set_offset(gen.get_offset() - 4)

        # Copy some objects so they don't get modified after returning.
        # This is important when using async scheduling.
        req_ids_output_copy = self.input_batch.req_ids.copy()
        req_id_to_index_output_copy = \
            self.input_batch.req_id_to_index.copy()

        # NOTE: GPU -> CPU Sync happens here.
        # Move as many CPU operations as possible before this sync point.
        logprobs_tensors = sampler_output.logprobs_tensors
        logprobs_lists = logprobs_tensors.tolists() \
            if logprobs_tensors is not None else None

        # Compute prompt logprobs if needed.
        prompt_logprobs_dict = self._get_prompt_logprobs_dict(
            hidden_states[:num_scheduled_tokens],
            scheduler_output.num_scheduled_tokens,
        )

        num_sampled_tokens = sampler_output.sampled_token_ids.shape[0]
        sampled_token_ids = sampler_output.sampled_token_ids
        invalid_req_indices = []
        if not self.use_async_scheduling:
            # Get the valid generated tokens.
            max_gen_len = sampled_token_ids.shape[-1]
            if max_gen_len == 1:
                # No spec decode tokens.
                valid_sampled_token_ids = self._to_list(sampled_token_ids)
            else:
                # Includes spec decode tokens.
                valid_sampled_token_ids = self.rejection_sampler.parse_output(
                    sampled_token_ids,
                    self.input_batch.vocab_size,
                )
                # TiDAR-only: now that rejection sampling has decided
                # num_accepted per req, write the post-acceptance candidate
                # state into each CCA layer's conv_states / prev_hs slot.
                # Without this, CCA state would be over-advanced by
                # (K - num_accepted) positions and the next forward's logits
                # would be garbage. See docs/diffusion_vllm_handoff.md.
                if (self.speculative_config
                        and self.speculative_config.use_tidar()):
                    self._commit_tidar_cca_state(valid_sampled_token_ids)
            # Mask out the sampled tokens that should not be sampled.
            for i in discard_sampled_tokens_req_indices:
                valid_sampled_token_ids[int(i)].clear()
            # DEBUG (TiDAR FULL cudagraph drift): dump per-step accepted
            # tokens to surface where the model collapses to mask_token.
            # Gated on VLLM_TIDAR_DEBUG_TOKENS to keep prod logs clean.
            import os as _os
            if _os.environ.get("VLLM_TIDAR_DEBUG_TOKENS", "0") == "1":
                for _i, _toks in enumerate(valid_sampled_token_ids):
                    if not _toks:
                        continue
                    logger.info(
                        "TIDAR_STEP req=%d n_accepted=%d tokens=%s",
                        _i, len(_toks) - 1, _toks[:20])
        else:
            valid_sampled_token_ids = []
            invalid_req_indices = discard_sampled_tokens_req_indices.tolist()
            invalid_req_indices_set = set(invalid_req_indices)
            assert sampled_token_ids.shape[-1] == 1

            # Cache the sampled tokens on the GPU and avoid CPU sync.
            # These will be copied into input_ids in the next step
            # when preparing inputs.
            self.input_batch.prev_sampled_token_ids = \
                sampled_token_ids
            self.input_batch.prev_sampled_token_ids_invalid_indices = \
                invalid_req_indices_set
            self.input_batch.prev_req_id_to_index = {
                req_id: i
                for i, req_id in enumerate(self.input_batch.req_ids)
                if i not in invalid_req_indices_set
            }

        # Cache the sampled tokens in the model runner, so that the scheduler
        # doesn't need to send them back.
        # NOTE(woosuk): As an exception, when using PP, the scheduler sends
        # the sampled tokens back, because there's no direct communication
        # between the first-stage worker and the last-stage worker.
        req_ids = self.input_batch.req_ids
        for req_idx in range(num_sampled_tokens):
            if self.use_async_scheduling:
                sampled_ids = [-1] if \
                    req_idx not in invalid_req_indices_set else None
            else:
                sampled_ids = valid_sampled_token_ids[req_idx]
            if not sampled_ids:
                continue

            start_idx = self.input_batch.num_tokens_no_spec[req_idx]
            end_idx = start_idx + len(sampled_ids)
            assert end_idx <= self.max_model_len, (
                "Sampled token IDs exceed the max model length. "
                f"Total number of tokens: {end_idx} > max_model_len: "
                f"{self.max_model_len}")

            self.input_batch.token_ids_cpu[req_idx,
                                           start_idx:end_idx] = sampled_ids
            self.input_batch.is_token_ids[req_idx, start_idx:end_idx] = True
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx
            self.input_batch.num_tokens[req_idx] = end_idx

            req_id = req_ids[req_idx]
            req_state = self.requests[req_id]
            req_state.output_token_ids.extend(sampled_ids)

        return (
            num_nans_in_logits,
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        )

    @contextmanager
    def synchronize_input_prep(self):
        if self.prepare_inputs_event is None:
            yield
            return

        # Ensure prior step has finished with reused CPU tensors.
        # This is required in the async scheduling case because
        # the CPU->GPU transfer happens async.
        self.prepare_inputs_event.synchronize()
        try:
            yield
        finally:
            self.prepare_inputs_event.record()

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, AsyncModelRunnerOutput, IntermediateTensors]:
        with record_function_or_nullcontext("Preprocess"):
            with self.synchronize_input_prep():
                # Update persistent batch states.
                self._update_states(scheduler_output)

                if not scheduler_output.total_num_scheduled_tokens:
                    if not has_kv_transfer_group():
                        # Return empty ModelRunnerOutput if no work to do.
                        return EMPTY_MODEL_RUNNER_OUTPUT
                    return self.kv_connector_no_forward(
                        scheduler_output, self.vllm_config)
                if self.cache_config.kv_sharing_fast_prefill:
                    assert not self.input_batch.num_prompt_logprobs, (
                        "--kv-sharing-fast-prefill produces incorrect "
                        "logprobs for prompt tokens, tokens, please disable "
                        "it when the requests need prompt logprobs")

                # Prepare the decoder inputs.
                (attn_metadata, logits_indices, spec_decode_metadata,
                 num_scheduled_tokens_np, spec_decode_common_attn_metadata,
                 max_query_len, ubatch_slices, num_tokens_after_padding
                 ) = self._prepare_inputs(scheduler_output)

            (
                num_scheduled_tokens,
                num_input_tokens,
                num_tokens_across_dp,
                input_ids,
                inputs_embeds,
                positions,
                intermediate_tensors,
                model_kwargs,
            ) = self._preprocess(scheduler_output, intermediate_tensors,
                                 ubatch_slices, num_tokens_after_padding)

            uniform_decode = (max_query_len
                              == self.uniform_decode_query_len) and (
                                  num_scheduled_tokens
                                  == self.input_batch.num_reqs * max_query_len)
            batch_descriptor = BatchDescriptor(num_tokens=num_input_tokens,
                                               uniform_decode=uniform_decode)
            cudagraph_runtime_mode, batch_descriptor = \
                self.cudagraph_dispatcher.dispatch(batch_descriptor)

        # This is currently to get around the assert in the DPMetadata
        # where it wants `num_tokens_across_dp` to align with `num_tokens`
        if ubatch_slices is not None:
            num_input_tokens = ubatch_slices[0].num_tokens

        # Run the model.
        # Use persistent buffers for CUDA graphs.
        with (set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                batch_descriptor=batch_descriptor,
                ubatch_slices=ubatch_slices,
        ), record_function_or_nullcontext("Forward"),
              self.maybe_get_kv_connector_output(scheduler_output) as
              kv_connector_output):
            model_output = self.model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
            )

        with record_function_or_nullcontext("Postprocess"):
            if self.use_aux_hidden_state_outputs:
                # True when EAGLE 3 is used.
                hidden_states, aux_hidden_states = model_output
            else:
                # Common case.
                hidden_states = model_output
                aux_hidden_states = None

            if not self.broadcast_pp_output:
                # Common case.
                if not get_pp_group().is_last_rank:
                    # Return the intermediate tensors.
                    assert isinstance(hidden_states, IntermediateTensors)
                    hidden_states.kv_connector_output = kv_connector_output
                    return hidden_states

                if self.is_pooling_model:
                    # Return the pooling output.
                    output = self._pool(hidden_states, num_scheduled_tokens,
                                        num_scheduled_tokens_np)
                    output.kv_connector_output = kv_connector_output
                    return output

                sample_hidden_states = hidden_states[logits_indices]
                logits = self.model.compute_logits(sample_hidden_states)
            else:
                # Rare case.
                assert not self.is_pooling_model

                if not get_pp_group().is_last_rank:
                    all_gather_tensors = {
                        "residual":
                        not is_residual_scattered_for_sp(
                            self.vllm_config, num_input_tokens)
                    }
                    get_pp_group().send_tensor_dict(
                        hidden_states.tensors,
                        all_gather_group=get_tp_group(),
                        all_gather_tensors=all_gather_tensors)
                    logits = None
                else:
                    sample_hidden_states = hidden_states[logits_indices]
                    logits = self.model.compute_logits(sample_hidden_states)

                model_output_broadcast_data = {}
                if logits is not None:
                    model_output_broadcast_data["logits"] = logits.contiguous()

                model_output_broadcast_data = get_pp_group(
                ).broadcast_tensor_dict(model_output_broadcast_data,
                                        src=len(get_pp_group().ranks) - 1)
                assert model_output_broadcast_data is not None
                logits = model_output_broadcast_data["logits"]

            # Apply structured output bitmasks if present
            if scheduler_output.grammar_bitmask is not None:
                apply_grammar_bitmask(scheduler_output, self.input_batch,
                                      logits, self.device)

        with record_function_or_nullcontext("Sample"):
            sampler_output = self._sample(logits, spec_decode_metadata)

        def propose_draft_token_ids(sampled_token_ids):
            assert spec_decode_common_attn_metadata is not None
            with record_function_or_nullcontext("Draft"):
                self._draft_token_ids = self.propose_draft_token_ids(
                    scheduler_output,
                    sampled_token_ids,
                    self.input_batch.sampling_metadata,
                    hidden_states,
                    sample_hidden_states,
                    aux_hidden_states,
                    spec_decode_metadata,
                    spec_decode_common_attn_metadata,
                )

        use_padded_batch_for_eagle = self._use_padded_drafter_batch()
        effective_drafter_max_model_len = self.max_model_len
        if effective_drafter_max_model_len is None:
            effective_drafter_max_model_len = self.model_config.max_model_len
        if (self.speculative_config
                and self.speculative_config.draft_model_config is not None
                and self.speculative_config.draft_model_config.max_model_len
                is not None):
            effective_drafter_max_model_len = (
                self.speculative_config.draft_model_config.max_model_len)
        input_fits_in_drafter = spec_decode_common_attn_metadata and (
            spec_decode_common_attn_metadata.seq_lens.max() +
            self.speculative_config.num_speculative_tokens
            <= effective_drafter_max_model_len)
        if use_padded_batch_for_eagle and input_fits_in_drafter:
            # EAGLE speculative decoding can use the GPU sampled tokens
            # as inputs, and does not need to wait for bookkeeping to finish.
            propose_draft_token_ids(sampler_output.sampled_token_ids)

        with record_function_or_nullcontext("Bookkeep"):
            (
                num_nans_in_logits,
                logprobs_lists,
                valid_sampled_token_ids,
                prompt_logprobs_dict,
                req_ids_output_copy,
                req_id_to_index_output_copy,
                invalid_req_indices,
            ) = self._bookkeeping_sync(scheduler_output, sampler_output,
                                       logits, hidden_states,
                                       num_scheduled_tokens)

        if (self.speculative_config and not use_padded_batch_for_eagle
                and input_fits_in_drafter):
            # ngram and other speculative decoding methods use the sampled
            # tokens on the CPU, so they are run after bookkeeping.
            propose_draft_token_ids(valid_sampled_token_ids)

        with record_function_or_nullcontext("EPLB"):
            self.eplb_step()

        output = ModelRunnerOutput(
            req_ids=req_ids_output_copy,
            req_id_to_index=req_id_to_index_output_copy,
            sampled_token_ids=valid_sampled_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=[],
            kv_connector_output=kv_connector_output,
            num_nans_in_logits=num_nans_in_logits,
        )

        if not self.use_async_scheduling:
            return output

        return AsyncGPUModelRunnerOutput(
            model_runner_output=output,
            sampled_token_ids=sampler_output.sampled_token_ids,
            invalid_req_indices=invalid_req_indices,
            async_output_copy_stream=self.async_output_copy_stream,
        )

    def take_draft_token_ids(self) -> Optional[DraftTokenIds]:
        if self._draft_token_ids is None:
            return None
        req_ids = self.input_batch.req_ids
        if isinstance(self._draft_token_ids, torch.Tensor):
            draft_token_ids = self._draft_token_ids.tolist()
        else:
            draft_token_ids = self._draft_token_ids
        self._draft_token_ids = None
        return DraftTokenIds(req_ids, draft_token_ids)

    def _use_padded_drafter_batch(self) -> bool:
        return bool(
            self.speculative_config
            and self.speculative_config.use_eagle()
            and not self.speculative_config.disable_padded_drafter_batch
            # TiDAR reuses the target model state, so it must draft only after
            # bookkeeping has compacted away rejected tokens.
            and not self.speculative_config.use_tidar())

    def propose_draft_token_ids(
        self,
        scheduler_output: "SchedulerOutput",
        sampled_token_ids: Union[torch.Tensor, list[list[int]]],
        sampling_metadata: SamplingMetadata,
        hidden_states: torch.Tensor,
        sample_hidden_states: torch.Tensor,
        aux_hidden_states: Optional[list[torch.Tensor]],
        spec_decode_metadata: Optional[SpecDecodeMetadata],
        common_attn_metadata: CommonAttentionMetadata,
    ) -> Union[list[list[int]], torch.Tensor]:
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        if self.speculative_config.method == "ngram":
            assert isinstance(sampled_token_ids, list)
            assert isinstance(self.drafter, NgramProposer)
            draft_token_ids = self.drafter.propose(
                sampled_token_ids, self.input_batch.req_ids,
                self.input_batch.num_tokens_no_spec,
                self.input_batch.token_ids_cpu,
                self.input_batch.spec_decode_unsupported_reqs)
        elif self.speculative_config.method == "medusa":
            assert isinstance(sampled_token_ids, list)
            assert isinstance(self.drafter, MedusaProposer)

            if sample_hidden_states.shape[0] == len(sampled_token_ids):
                # The input to the target model does not include draft tokens.
                hidden_states = sample_hidden_states
            else:
                indices = []
                offset = 0
                assert spec_decode_metadata is not None
                for num_draft, tokens in zip(
                        spec_decode_metadata.num_draft_tokens,
                        sampled_token_ids):
                    indices.append(offset + len(tokens) - 1)
                    offset += num_draft + 1
                indices = torch.tensor(indices, device=self.device)
                hidden_states = sample_hidden_states[indices]

            draft_token_ids = self.drafter.propose(
                target_hidden_states=hidden_states,
                sampling_metadata=sampling_metadata,
            )
        elif self.speculative_config.use_eagle():
            assert isinstance(self.drafter, EagleProposer)
            use_padded_drafter_batch = self._use_padded_drafter_batch()

            if not use_padded_drafter_batch:
                # When padded-batch is disabled, the sampled_token_ids should be
                # the cpu-side list[list[int]] of valid sampled tokens for each
                # request, with invalid requests having empty lists.
                assert isinstance(sampled_token_ids, list), \
                    "sampled_token_ids should be a python list when" \
                    "padded-batch is disabled."
                next_token_ids = self.drafter.prepare_next_token_ids_cpu(
                    sampled_token_ids, self.requests, self.input_batch,
                    scheduler_output.num_scheduled_tokens)
            else:
                # When using padded-batch, the sampled_token_ids should be
                # the gpu tensor of sampled tokens for each request, of shape
                # (num_reqs, num_spec_tokens + 1) with rejected tokens having
                # value -1.
                assert isinstance(sampled_token_ids, torch.Tensor), \
                    "sampled_token_ids should be a torch.Tensor when" \
                    "padded-batch is enabled."
                next_token_ids, valid_sampled_tokens_count = \
                    self.drafter.prepare_next_token_ids_padded(
                        common_attn_metadata,
                        sampled_token_ids,
                        self.requests,
                        self.input_batch,
                        self.discard_request_indices.gpu,
                        self.num_discarded_requests
                    )

            if spec_decode_metadata is None:
                token_indices_to_sample = None
                # input_ids can be None for multimodal models.
                target_token_ids = self.input_ids.gpu[:num_scheduled_tokens]
                # TODO(woosuk): Support M-RoPE.
                target_positions = self.positions.gpu[:num_scheduled_tokens]
                if self.use_aux_hidden_state_outputs:
                    assert aux_hidden_states is not None
                    target_hidden_states = torch.cat(
                        [h[:num_scheduled_tokens] for h in aux_hidden_states],
                        dim=-1)
                else:
                    target_hidden_states = hidden_states[:num_scheduled_tokens]
            else:
                if not use_padded_drafter_batch:
                    token_indices_to_sample = None
                    common_attn_metadata, token_indices =\
                        self.drafter.prepare_inputs(
                            common_attn_metadata,
                            sampled_token_ids,
                            spec_decode_metadata.num_draft_tokens)
                else:
                    common_attn_metadata, token_indices, \
                        token_indices_to_sample =\
                        self.drafter.prepare_inputs_padded(
                            common_attn_metadata,
                            spec_decode_metadata,
                            valid_sampled_tokens_count)

                target_token_ids = self.input_ids.gpu[token_indices]
                # TODO(woosuk): Support M-RoPE.
                target_positions = self.positions.gpu[token_indices]
                if self.use_aux_hidden_state_outputs:
                    assert aux_hidden_states is not None
                    target_hidden_states = torch.cat(
                        [h[token_indices] for h in aux_hidden_states], dim=-1)
                else:
                    target_hidden_states = hidden_states[token_indices]
            mm_embeds = None
            if self.supports_mm_inputs:
                mm_embeds = self._gather_mm_embeddings(scheduler_output,
                                                       shift_computed_tokens=1)

            # Skip drafter when there are no reqs to draft for (e.g., all
            # reqs finished in the just-completed step, or the first chunk
            # of a chunked prefill that doesn't include the prompt's last
            # position). target_positions / next_token_ids being empty
            # would IndexError in the drafter's _build_draft_inputs at
            # `target_positions[last_token_indices] + 1`.
            if (target_positions.numel() == 0
                    or next_token_ids.numel() == 0):
                if self.speculative_config.use_tidar():
                    tidar_drafter = self.drafter
                    tidar_drafter.last_draft_probs = None
                return torch.empty(
                    (0, self.speculative_config.num_speculative_tokens),
                    dtype=torch.int32,
                    device=self.device)
            if (self.speculative_config.use_tidar()
                    and getattr(self.drafter, "single_forward_mode", False)
                    and spec_decode_metadata is not None
                    and getattr(self.drafter, "last_inflated_total", None)
                    is not None):
                # Single-forward TiDAR (only when there were spec_decode
                # tokens this step -- i.e., not the bootstrap first
                # decode step after cold prefill). Drafts are sourced
                # from the verify-pass hidden states' proposal segment
                # (no second model.forward()).
                #
                # ``hidden_states`` (NOT ``target_hidden_states``) is the
                # full [num_scheduled_tokens, hidden_size] tensor from
                # the verify forward, covering verify + P proposal
                # segments per req. ``target_hidden_states`` above was
                # sliced to ``token_indices`` via drafter.prepare_inputs
                # (eagle convention) which only kept the verify-segment
                # rows the drafter would have re-anchored on -- that's
                # not what we want for single-forward draft extraction.
                # ``num_accepted_per_req`` reconstructed from
                # ``sampled_token_ids`` (list[list[int]] per req, length
                # = num_accepted + 1 because each req's list =
                # [accepted_drafts..., bonus]).
                assert isinstance(sampled_token_ids, list), (
                    "TiDAR single-forward expects list[list[int]] "
                    "sampled_token_ids (non-padded path).")
                num_accepted_per_req = torch.tensor(
                    [max(0, len(tokens) - 1)
                     for tokens in sampled_token_ids],
                    dtype=torch.int32, device=self.device)
                draft_token_ids = self.drafter.extract_drafts_from_hidden(
                    hidden_states=hidden_states,
                    num_accepted_per_req=num_accepted_per_req,
                )
            else:
                draft_token_ids = self.drafter.propose(
                    target_token_ids=target_token_ids,
                    target_positions=target_positions,
                    target_hidden_states=target_hidden_states,
                    next_token_ids=next_token_ids,
                    last_token_indices=token_indices_to_sample,
                    sampling_metadata=sampling_metadata,
                    common_attn_metadata=common_attn_metadata,
                    mm_embeds=mm_embeds,
                )
            # Stash draft_probs from TiDAR (T_Diff > 0) keyed by req_id so the
            # next step's input prep can reassemble them in batch order.
            # Cleared each step because the dict is overwritten — no stale
            # entries can leak into subsequent steps for the same req_id.
            if self.speculative_config.use_tidar():
                self._draft_probs_by_req_id.clear()
                tidar_drafter = self.drafter
                last_probs = getattr(tidar_drafter, "last_draft_probs", None)
                if last_probs is not None:
                    num_spec = self.speculative_config.num_speculative_tokens
                    # last_probs shape: [batch_size * num_spec, vocab_size]
                    per_req = last_probs.view(-1, num_spec,
                                              last_probs.shape[-1])
                    for i, req_id in enumerate(self.input_batch.req_ids):
                        if i < per_req.shape[0]:
                            self._draft_probs_by_req_id[req_id] = per_req[i]
                # Mirror the stash for raw draft_logits (used by mix-logit
                # v1). Populated unconditionally by the drafter — even at
                # T_Diff == 0 — so this branch runs whenever last_draft_logits
                # is non-None, independent of the draft_probs path above.
                self._draft_logits_by_req_id.clear()
                last_logits = getattr(tidar_drafter, "last_draft_logits", None)
                if not hasattr(self, '_stash_dbg_steps'):
                    self._stash_dbg_steps = 0
                if self._stash_dbg_steps < 3:
                    self._stash_dbg_steps += 1
                    print(f"[STASH_DBG step={self._stash_dbg_steps}] last_logits is None: {last_logits is None}", flush=True)
                if last_logits is not None:
                    num_spec = self.speculative_config.num_speculative_tokens
                    per_req_l = last_logits.view(-1, num_spec,
                                                 last_logits.shape[-1])
                    for i, req_id in enumerate(self.input_batch.req_ids):
                        if i < per_req_l.shape[0]:
                            self._draft_logits_by_req_id[req_id] = per_req_l[i]
        return draft_token_ids

    def update_config(self, overrides: dict[str, Any]) -> None:
        allowed_config_names = {"load_config", "model_config"}
        for config_name, config_overrides in overrides.items():
            assert config_name in allowed_config_names, \
                f"Config `{config_name}` not supported. " \
                f"Allowed configs: {allowed_config_names}"
            config = getattr(self, config_name)
            new_config = update_config(config, config_overrides)
            setattr(self, config_name, new_config)

    def load_model(self, eep_scale_up: bool = False) -> None:
        """
        Args:
            eep_scale_up: the model loading is for elastic EP scale up.
        """
        logger.info("Starting to load model %s...", self.model_config.model)
        if eep_scale_up:
            from vllm.distributed.parallel_state import get_ep_group
            num_local_physical_experts = torch.empty(1,
                                                     dtype=torch.int32,
                                                     device="cpu")
            torch.distributed.broadcast(num_local_physical_experts,
                                        group=get_ep_group().cpu_group,
                                        group_src=0)
            num_local_physical_experts = int(num_local_physical_experts.item())
            new_ep_size = get_ep_group().world_size
            global_expert_load, old_global_expert_indices = (
                EplbState.recv_state())
            num_logical_experts = global_expert_load.shape[1]
            self.parallel_config.eplb_config.num_redundant_experts = (
                num_local_physical_experts * new_ep_size - num_logical_experts)
            assert old_global_expert_indices.shape[
                1] % num_local_physical_experts == 0
            old_ep_size = old_global_expert_indices.shape[
                1] // num_local_physical_experts
            rank_mapping = {
                old_ep_rank: old_ep_rank
                for old_ep_rank in range(old_ep_size)
            }
        else:
            global_expert_load = None
            old_global_expert_indices = None
            rank_mapping = None

        with DeviceMemoryProfiler() as m:
            time_before_load = time.perf_counter()
            model_loader = get_model_loader(self.load_config)
            logger.info("Loading model from scratch...")
            self.model = model_loader.load_model(
                vllm_config=self.vllm_config, model_config=self.model_config)
            if self.lora_config:
                self.model = self.load_lora_model(self.model, self.vllm_config,
                                                  self.device)
            if hasattr(self, "drafter"):
                logger.info("Loading drafter model...")
                self.drafter.load_model(self.model)
            if self.use_aux_hidden_state_outputs:
                if supports_eagle3(self.model):
                    self.model.set_aux_hidden_state_layers(
                        self.model.get_eagle3_aux_hidden_state_layers())
                else:
                    raise RuntimeError(
                        "Model does not support the auxiliary hidden-state "
                        f"interface required for "
                        f"{self.speculative_config.method}.")
            time_after_load = time.perf_counter()
        self.model_memory_usage = m.consumed_memory
        logger.info("Model loading took %.4f GiB and %.6f seconds",
                    self.model_memory_usage / GiB_bytes,
                    time_after_load - time_before_load)
        prepare_communication_buffer_for_model(self.model)

        self.is_multimodal_pruning_enabled = (supports_multimodal_pruning(
            self.model) and self.model_config.multimodal_config.
                                              is_multimodal_pruning_enabled())

        if is_mixture_of_experts(
                self.model) and self.parallel_config.enable_eplb:
            logger.info("EPLB is enabled for model %s.",
                        self.model_config.model)
            self.eplb_state = EplbState.build(
                self.model,
                self.device,
                self.parallel_config,
                global_expert_load,
                old_global_expert_indices,
                rank_mapping,
            )

        if (
            self.vllm_config.compilation_config.level == \
                CompilationLevel.DYNAMO_AS_IS and supports_dynamo()
        ):
            backend = self.vllm_config.compilation_config.init_backend(
                self.vllm_config)
            compilation_counter.dynamo_as_is_count += 1
            self.model.compile(fullgraph=True, backend=backend)
            return
        # for other compilation levels, cudagraph behavior is controlled by
        # CudagraphWraper and CudagraphDispatcher of vllm.

        # wrap the model with full cudagraph wrapper if needed.
        if self.compilation_config.cudagraph_mode.has_full_cudagraphs() \
            and not self.parallel_config.enable_dbo:
            self.model = CUDAGraphWrapper(self.model,
                                          self.vllm_config,
                                          runtime_mode=CUDAGraphMode.FULL)
            # Tier 3: re-bind drafter to the FULL-wrapped model. The
            # drafter dispatches with BatchDescriptor(is_drafter_pass=
            # True), which routes to a *separate* CUDAGraphEntry inside
            # the wrapper — captured lazily on the first runtime call
            # (see propose() in tidar.py for the capture-enable
            # bracket). Drafter then runs ~5x faster than eager at the
            # K+1 shape (60ms → ~11ms on Minerva b=1 K=16) and produces
            # correct output because path 2's read=AR / write=draft
            # routing is baked from drafter-runtime metadata at the
            # drafter graph's own capture time.
            if hasattr(self, "drafter") and hasattr(self.drafter, "model"):
                self.drafter.model = self.model
        elif self.parallel_config.enable_dbo:
            if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
                self.model = UBatchWrapper(self.model, self.vllm_config,
                                           CUDAGraphMode.FULL, self.device)
            else:
                self.model = UBatchWrapper(self.model, self.vllm_config,
                                           CUDAGraphMode.NONE, self.device)

    def reload_weights(self) -> None:
        assert getattr(self, "model", None) is not None, \
            "Cannot reload weights before model is loaded."
        model_loader = get_model_loader(self.load_config)
        logger.info("Reloading weights inplace...")
        model = self.get_model()
        model_loader.load_weights(model, model_config=self.model_config)

    def save_tensorized_model(
        self,
        tensorizer_config: "TensorizerConfig",
    ) -> None:
        model = self.get_model()
        TensorizerLoader.save_model(
            model,
            tensorizer_config=tensorizer_config,
            model_config=self.model_config,
        )

    def _get_prompt_logprobs_dict(
        self,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: dict[str, int],
    ) -> dict[str, Optional[LogprobsTensors]]:
        num_prompt_logprobs_dict = self.input_batch.num_prompt_logprobs
        if not num_prompt_logprobs_dict:
            return {}

        in_progress_dict = self.input_batch.in_progress_prompt_logprobs_cpu
        prompt_logprobs_dict: dict[str, Optional[LogprobsTensors]] = {}

        # Since prompt logprobs are a rare feature, prioritize simple,
        # maintainable loop over optimal performance.
        completed_prefill_reqs = []
        for req_id, num_prompt_logprobs in num_prompt_logprobs_dict.items():
            num_tokens = num_scheduled_tokens[req_id]

            # Get metadata for this request.
            request = self.requests[req_id]
            if request.prompt_token_ids is None:
                # Prompt logprobs is incompatible with prompt embeddings
                continue

            num_prompt_tokens = len(request.prompt_token_ids)
            prompt_token_ids = torch.tensor(request.prompt_token_ids).to(
                self.device, non_blocking=True)

            # Set up target LogprobsTensors object.
            logprobs_tensors = in_progress_dict.get(req_id)
            if not logprobs_tensors:
                # Create empty logprobs CPU tensors for the entire prompt.
                # If chunked, we'll copy in slice by slice.
                logprobs_tensors = LogprobsTensors.empty_cpu(
                    num_prompt_tokens - 1, num_prompt_logprobs + 1)
                in_progress_dict[req_id] = logprobs_tensors

            # Determine number of logits to retrieve.
            start_idx = request.num_computed_tokens
            start_tok = start_idx + 1
            num_remaining_tokens = num_prompt_tokens - start_tok
            if num_tokens <= num_remaining_tokens:
                # This is a chunk, more tokens remain.
                # In the == case, there are no more prompt logprobs to produce
                # but we want to defer returning them to the next step where we
                # have new generated tokens to return.
                num_logits = num_tokens
            else:
                # This is the last chunk of prompt tokens to return.
                num_logits = num_remaining_tokens
                completed_prefill_reqs.append(req_id)
                prompt_logprobs_dict[req_id] = logprobs_tensors

            if num_logits <= 0:
                # This can happen for the final chunk if we prefilled exactly
                # (num_prompt_tokens - 1) tokens for this request in the prior
                # step. There are no more prompt logprobs to produce.
                continue

            # Get the logits corresponding to this req's prompt tokens.
            # If this is a partial request (i.e. chunked prefill),
            # then there is prompt logprob generated for each index.
            req_idx = self.input_batch.req_id_to_index[req_id]
            offset = self.query_start_loc.np[req_idx].item()
            prompt_hidden_states = hidden_states[offset:offset + num_logits]
            logits = self.model.compute_logits(prompt_hidden_states)

            # Get the "target" tokens for each index. For prompt at index i,
            # the token at prompt index i+1 is the "sampled" token we want
            # to gather the logprob for.
            tgt_token_ids = prompt_token_ids[start_tok:start_tok + num_logits]

            # Compute prompt logprobs.
            logprobs = self.sampler.compute_logprobs(logits)
            token_ids, logprobs, ranks = self.sampler.gather_logprobs(
                logprobs, num_prompt_logprobs, tgt_token_ids)

            # Transfer GPU->CPU async.
            chunk_slice = slice(start_idx, start_idx + num_logits)
            logprobs_tensors.logprob_token_ids[chunk_slice].copy_(
                token_ids, non_blocking=True)
            logprobs_tensors.logprobs[chunk_slice].copy_(logprobs,
                                                         non_blocking=True)
            logprobs_tensors.selected_token_ranks[chunk_slice].copy_(
                ranks, non_blocking=True)

        # Remove requests that have completed prefill from the batch
        # num_prompt_logprobs_dict.
        for req_id in completed_prefill_reqs:
            del num_prompt_logprobs_dict[req_id]
            del in_progress_dict[req_id]

        # Must synchronize the non-blocking GPU->CPU transfers.
        if prompt_logprobs_dict:
            self._sync_device()

        return prompt_logprobs_dict

    def _get_nans_in_logits(
        self,
        logits: Optional[torch.Tensor],
    ) -> dict[str, int]:
        try:
            if logits is None:
                return {req_id: 0 for req_id in self.input_batch.req_ids}

            num_nans_in_logits = {}
            num_nans_for_index = logits.isnan().sum(dim=-1).cpu().numpy()
            for req_id in self.input_batch.req_ids:
                req_index = self.input_batch.req_id_to_index[req_id]
                num_nans_in_logits[req_id] = (
                    int(num_nans_for_index[req_index])
                    if num_nans_for_index is not None
                    and req_index < logits.shape[0] else 0)
            return num_nans_in_logits
        except IndexError:
            return {}

    @contextmanager
    def maybe_randomize_inputs(self, input_ids: torch.Tensor):
        """
        Randomize input_ids if VLLM_RANDOMIZE_DP_DUMMY_INPUTS is set.
        This is to help balance expert-selection
         - during profile_run
         - during DP rank dummy run
        """
        dp_size = self.vllm_config.parallel_config.data_parallel_size
        randomize_inputs = envs.VLLM_RANDOMIZE_DP_DUMMY_INPUTS and dp_size > 1
        if not randomize_inputs:
            yield
        else:
            import functools

            @functools.cache
            def rand_input_ids() -> torch.Tensor:
                return torch.randint_like(
                    self.input_ids.gpu,
                    low=0,
                    high=self.model_config.get_vocab_size(),
                    dtype=input_ids.dtype)

            logger.debug_once("Randomizing dummy data for DP Rank")
            input_ids.copy_(rand_input_ids()[:input_ids.size(0)],
                            non_blocking=True)
            yield
            input_ids.fill_(0)

    def _get_mm_dummy_batch(
        self,
        modality: str,
        max_items_per_batch: int,
    ) -> BatchedTensorInputs:
        """Dummy data for profiling and precompiling multimodal models."""
        assert self.mm_budget is not None

        dummy_decoder_data = self.mm_registry.get_decoder_dummy_data(
            model_config=self.model_config,
            seq_len=self.max_model_len,
            mm_counts={modality: 1},
            cache=self.mm_budget.cache,
        )
        dummy_mm_data = dummy_decoder_data.multi_modal_data

        # Result in the maximum GPU consumption of the model
        dummy_mm_item = dummy_mm_data[modality][0]
        dummy_mm_items = [dummy_mm_item] * max_items_per_batch

        model = cast(SupportsMultiModal, self.model)
        return next(mm_kwargs_group
                    for _, _, mm_kwargs_group in group_mm_kwargs_by_modality(
                        dummy_mm_items,
                        device=self.device,
                        pin_memory=self.pin_memory,
                        merge_by_field_config=model.merge_by_field_config,
                    ))

    @torch.inference_mode()
    def _dummy_run(
        self,
        num_tokens: int,
        cudagraph_runtime_mode: Optional[CUDAGraphMode] = None,
        force_attention: bool = False,
        uniform_decode: bool = False,
        allow_microbatching: bool = True,
        skip_eplb: bool = False,
        is_profile: bool = False,
        create_mixed_batch: bool = False,
        remove_lora: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run a dummy forward pass to warm up/profile run or capture the
        CUDA graph for the model.

        Args:
            num_tokens: Number of tokens to run the dummy forward pass.
            cudagraph_runtime_mode: used to control the behavior.
                - if not set will determine the cudagraph mode based on using
                    the self.cudagraph_dispatcher.
                - CUDAGraphMode.NONE: No cudagraph, for warm up and profile run
                - CUDAGraphMode.PIECEWISE: Piecewise cudagraph.
                - CUDAGraphMode.FULL: Full cudagraph, attention metadata is
                    needed.
            force_attention: If True, always create attention metadata. Used to
                warm up attention backend when mode is NONE.
            uniform_decode: If True, the batch is a uniform decode batch.
            skip_eplb: If True, skip EPLB state update.
            is_profile: If True, this is a profile run.
            create_mixed_batch: If True, create a mixed batch with both decode
                (1 token) and prefill (multiple tokens) requests.
            remove_lora: If False, dummy LoRAs are not destroyed after the run
        """
        assert cudagraph_runtime_mode is None or cudagraph_runtime_mode in {
            CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL
        }

        # If cudagraph_mode.decode_mode() == FULL and
        # cudagraph_mode.separate_routine(). This means that we are using
        # different graphs and/or modes for mixed prefill-decode batches vs.
        # uniform decode batches. A uniform decode batch means that all
        # requests have identical query length, except a potential virtual
        # request (shorter) in the batch account for padding.
        # Uniform decode batch could either be common pure decode, where
        # max_query_len == 1, or speculative decode, where
        # max_query_len == 1 + num_spec_decode_tokens.

        # When setting max_query_len = 1, we switch to and capture the optimized
        # routine of FA2 for pure decode, i.e., Flashdecode + an optimization
        # for GQA/MQA.
        max_query_len = self.uniform_decode_query_len if uniform_decode else \
                                                                num_tokens

        # Set num_scheduled_tokens based on num_tokens and max_num_seqs
        # for dummy run with LoRA so that the num_reqs collectively
        # has num_tokens in total.
        assert num_tokens <= self.scheduler_config.max_num_batched_tokens
        max_num_reqs = self.scheduler_config.max_num_seqs
        if create_mixed_batch:
            assert not uniform_decode
            # Create mixed batch:
            # first half decode tokens, second half one prefill
            num_decode_tokens = num_tokens // 2
            num_prefill_tokens = num_tokens - num_decode_tokens
            num_reqs = num_decode_tokens + 1

            # Create decode requests (1 token each) followed by prefill request
            num_scheduled_tokens_list = [1] * num_decode_tokens + [
                num_prefill_tokens
            ]
            # Note: Overriding max_query_len to be the prefill tokens
            max_query_len = num_prefill_tokens
        elif uniform_decode:
            assert not create_mixed_batch
            num_reqs = cdiv(num_tokens, max_query_len)
            num_scheduled_tokens_list = [max_query_len] * num_reqs
            if num_tokens % max_query_len != 0:
                num_scheduled_tokens_list[-1] = num_tokens % max_query_len
        else:
            num_reqs = min(num_tokens, max_num_reqs)
            min_tokens_per_req = num_tokens // num_reqs
            num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
            num_scheduled_tokens_list[-1] += num_tokens % num_reqs

        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs
        num_scheduled_tokens = np.array(num_scheduled_tokens_list,
                                        dtype=np.int32)
        total_num_scheduled_tokens = int(num_scheduled_tokens.sum())

        ubatch_slices = None
        num_tokens_after_padding = None

        # We currently only microbatch if the number of tokens is
        # over a certain threshold.
        if self.parallel_config.enable_dbo and allow_microbatching:
            ubatch_slices, ubatch_num_tokens_after_padding = ubatch_split(
                num_scheduled_tokens,
                total_num_scheduled_tokens,
                total_num_scheduled_tokens,
                uniform_decode=uniform_decode,
                vllm_config=self.vllm_config,
            )
            # Currently when DBO is enabled `ubatch_split` returns
            # the num_tokens_after_padding for a single ubatch, but we have 2
            # TODO(sage,lucas): this is cruft that should be addressed in the
            # padding refactor.
            if ubatch_num_tokens_after_padding is not None:
                num_tokens_after_padding = ubatch_num_tokens_after_padding * 2

        # If we failed to microbatch, currently need to resynchronize
        # TODO(lucas,sage): we should be able to avoid this second sync by
        #  refactoring `get_dp_padding_ubatch` and `get_dp_padding` into
        #  a single `coordinate_batch_across_dp` function.
        if num_tokens_after_padding is None:
            num_pad, num_tokens_across_dp = self.get_dp_padding(num_tokens)
            num_tokens_after_padding = num_tokens + num_pad
        else:
            num_tokens_across_dp = num_tokens_after_padding
            num_tokens_after_padding = int(num_tokens_after_padding[0].item())

        attn_metadata: Optional[PerLayerAttnMetadata] = None

        # If force_attention is True, we always capture attention. Otherwise,
        # it only happens for cudagraph_runtime_mode=FULL.
        if force_attention or cudagraph_runtime_mode == CUDAGraphMode.FULL:
            attn_metadata = {}
            if ubatch_slices is not None:
                attn_metadata = [dict() for _ in range(len(ubatch_slices))]

            if create_mixed_batch:
                # In the mixed batch mode (used for FI warmup), we use
                # shorter sequence lengths to run faster.
                # TODO(luka) better system for describing dummy batches
                seq_lens = [1] * num_decode_tokens + [num_prefill_tokens + 1]
            else:
                seq_lens = max_query_len
            self.seq_lens.np[:num_reqs] = seq_lens
            self.seq_lens.np[num_reqs:] = 0
            self.seq_lens.copy_to_gpu()

            cum_num_tokens, _ = self._get_cumsum_and_arange(
                num_scheduled_tokens)
            self.query_start_loc.np[1:num_reqs + 1] = cum_num_tokens
            self.query_start_loc.copy_to_gpu()

            for kv_cache_group_id, kv_cache_group_spec in enumerate(
                    self.kv_cache_config.kv_cache_groups):
                common_attn_metadata = CommonAttentionMetadata(
                    query_start_loc=self.query_start_loc.gpu[:num_reqs + 1],
                    query_start_loc_cpu=self.query_start_loc.cpu[:num_reqs +
                                                                 1],
                    seq_lens=self.seq_lens.gpu[:num_reqs],
                    seq_lens_cpu=self.seq_lens.cpu[:num_reqs],
                    num_computed_tokens_cpu=self.input_batch.
                    num_computed_tokens_cpu_tensor[:num_reqs],
                    num_reqs=num_reqs,
                    num_actual_tokens=num_tokens,
                    max_query_len=max_query_len,
                    max_seq_len=self.max_model_len,
                    block_table_tensor=self.input_batch.
                    block_table[kv_cache_group_id].get_device_tensor(num_reqs),
                    slot_mapping=self.input_batch.block_table[
                        kv_cache_group_id].slot_mapping.gpu[:num_tokens],
                    causal=True)
                # TiDAR FULL cudagraph capture: mirror the runtime
                # _prepare_inputs metadata setup so CCA's captured forward
                # takes the SAME branch as runtime. Without this, the
                # captured graph at SF size (K+1+P*K) falls through to a
                # non-spec-decode CCA branch that does NOT write
                # _spec_stash_*; the captured graph then has zero stash
                # writes baked in, and commit_spec_decode_state gathers
                # zeros at every replay step -- producing garbage tokens
                # from step 1 onward. See
                # docs/tidar_full_cudagraph_handoff_2026-05-14.md.
                if (uniform_decode and self.speculative_config
                        and self.speculative_config.use_tidar()
                        and num_tokens > 0
                        and num_tokens % self.uniform_decode_query_len == 0):
                    setattr(common_attn_metadata,
                            "cca_spec_decode_mode", True)
                    if (getattr(self.drafter,
                                "single_forward_mode", False)
                            and hasattr(
                                self.drafter,
                                "set_tidar_single_forward_metadata")):
                        self.drafter.set_tidar_single_forward_metadata(
                            common_attn_metadata)
                for attn_group in self.attn_groups[kv_cache_group_id]:
                    if ubatch_slices is not None:
                        common_attn_metadata_list = split_attn_metadata(
                            ubatch_slices, common_attn_metadata)
                        for ubid, common_attn_metadata in enumerate(
                                common_attn_metadata_list):
                            assert common_attn_metadata.max_query_len == 1
                            attn_metadata_i = (attn_group\
                                               .get_metadata_builder(ubatch_id=ubid)\
                                               .build_for_cudagraph_capture(common_attn_metadata))
                            for layer_name in attn_group.layer_names:
                                assert type(attn_metadata) is list
                                attn_metadata[ubid][
                                    layer_name] = attn_metadata_i
                    else:
                        assert type(attn_metadata) is dict
                        attn_metadata_i = attn_group.get_metadata_builder()\
                            .build_for_cudagraph_capture(common_attn_metadata)
                        for layer_name in attn_group.layer_names:
                            attn_metadata[layer_name] = attn_metadata_i

        with self.maybe_dummy_run_with_lora(self.lora_config,
                                            num_scheduled_tokens, remove_lora):
            model_kwargs = self._init_model_kwargs(num_tokens)
            if (self.supports_mm_inputs
                    and not self.model_config.is_encoder_decoder):
                input_ids = None
                inputs_embeds = self.inputs_embeds.gpu[:num_tokens]
                model_kwargs = {
                    **model_kwargs,
                    **self._dummy_mm_kwargs(num_reqs),
                }
            elif self.enable_prompt_embeds:
                input_ids = None
                inputs_embeds = self.inputs_embeds.gpu[:num_tokens]
                model_kwargs = self._init_model_kwargs(num_tokens)
            else:
                input_ids = self.input_ids.gpu[:num_tokens]
                inputs_embeds = None

            if self.uses_mrope:
                positions = self.mrope_positions.gpu[:, :num_tokens]
            else:
                positions = self.positions.gpu[:num_tokens]

            if get_pp_group().is_first_rank:
                intermediate_tensors = None
            else:
                if self.intermediate_tensors is None:
                    self.intermediate_tensors = (
                        self.model.make_empty_intermediate_tensors(
                            batch_size=self.max_num_tokens,
                            dtype=self.model_config.dtype,
                            device=self.device))

                intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                    num_tokens, None, False)

            # filter out the valid batch descriptor
            _cg_mode, batch_descriptor = self.cudagraph_dispatcher.dispatch(
                BatchDescriptor(num_tokens=num_tokens_after_padding,
                                uniform_decode=uniform_decode)) \
                if not is_profile else (CUDAGraphMode.NONE, None)
            if cudagraph_runtime_mode is not None:
                # we allow forcing NONE when the dispatcher disagrees to support
                # warm ups for cudagraph capture
                assert cudagraph_runtime_mode == CUDAGraphMode.NONE or \
                    cudagraph_runtime_mode == _cg_mode, (
                    f"Cudagraph runtime mode mismatch at dummy_run. "
                    f"Expected {_cg_mode}, but got {cudagraph_runtime_mode}.")
            else:
                cudagraph_runtime_mode = _cg_mode

            if ubatch_slices is not None:
                # Adjust values to reflect a single ubatch.
                # TODO(sage,lucas): this is cruft that should be addressed in
                #  the padding refactor.
                num_tokens_after_padding = ubatch_slices[0].num_tokens
                if num_tokens_across_dp is not None:
                    num_tokens_across_dp[:] = num_tokens_after_padding

            with self.maybe_randomize_inputs(input_ids), set_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens_after_padding,
                    num_tokens_across_dp=num_tokens_across_dp,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    batch_descriptor=batch_descriptor,
                    ubatch_slices=ubatch_slices):
                outputs = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                    **model_kwargs,
                )

            if self.use_aux_hidden_state_outputs:
                hidden_states, _ = outputs
            else:
                hidden_states = outputs

            if self.speculative_config and self.speculative_config.use_eagle():
                assert isinstance(self.drafter, EagleProposer)
                self.drafter.dummy_run(num_tokens)

        # This is necessary to avoid blocking DP.
        # For dummy runs, we typically skip EPLB since we don't have any real
        # requests to process.
        # However, in DP settings, there may be cases when some DP ranks do
        # not have any requests to process, so they're executing dummy batches.
        # In such cases, we still have to trigger EPLB to make sure
        # ranks execute the rearrangement in synchronization.
        if not skip_eplb:
            self.eplb_step(is_dummy=True, is_profile=is_profile)

        logit_indices = np.cumsum(num_scheduled_tokens) - 1
        return hidden_states, hidden_states[logit_indices]

    @torch.inference_mode()
    def _dummy_sampler_run(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # The dummy hidden states may contain special values,
        # like `inf` or `nan`.
        # To avoid breaking the sampler, we use a random tensor here instead.
        hidden_states = torch.rand_like(hidden_states)

        logits = self.model.compute_logits(hidden_states)
        num_reqs = logits.size(0)

        dummy_tensors = lambda v: torch.full(
            (num_reqs, ), v, device=self.device)

        dummy_metadata = SamplingMetadata(
            temperature=dummy_tensors(0.5),
            all_greedy=False,
            all_random=False,
            top_p=dummy_tensors(0.9),
            top_k=dummy_tensors(logits.size(1) - 1),
            generators={},
            max_num_logprobs=None,
            no_penalties=True,
            prompt_token_ids=None,
            frequency_penalties=dummy_tensors(0.1),
            presence_penalties=dummy_tensors(0.1),
            repetition_penalties=dummy_tensors(0.1),
            output_token_ids=[[] for _ in range(num_reqs)],
            allowed_token_ids_mask=None,
            bad_words_token_ids={},
            logitsprocs=LogitsProcessors(),
        )
        try:
            sampler_output = self.sampler(logits=logits,
                                          sampling_metadata=dummy_metadata)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                raise RuntimeError(
                    "CUDA out of memory occurred when warming up sampler with "
                    f"{num_reqs} dummy requests. Please try lowering "
                    "`max_num_seqs` or `gpu_memory_utilization` when "
                    "initializing the engine.") from e
            else:
                raise e
        if self.speculative_config:
            draft_token_ids = [[0] for _ in range(num_reqs)]
            dummy_spec_decode_metadata = SpecDecodeMetadata.make_dummy(
                draft_token_ids, self.device)

            num_tokens = sum(len(ids) for ids in draft_token_ids)
            # draft_probs = torch.randn(
            #     num_tokens, logits.shape[-1], device=self.device,
            #     dtype=logits.dtype)
            draft_probs = None
            target_logits = torch.randn(num_tokens,
                                        logits.shape[-1],
                                        device=self.device,
                                        dtype=logits.dtype)
            # NOTE(woosuk): Here, we should use int32 because the sampler uses
            # int32 for bonus_token_ids. If the dtype mismatches, re-compilation
            # will occur at runtime.
            bonus_token_ids = torch.zeros(num_reqs,
                                          device=self.device,
                                          dtype=torch.int32)
            self.rejection_sampler(
                dummy_spec_decode_metadata,
                draft_probs,
                target_logits,
                bonus_token_ids,
                dummy_metadata,
            )
        return sampler_output

    def _dummy_pooler_run_task(
        self,
        hidden_states: torch.Tensor,
        task: PoolingTask,
    ) -> PoolerOutput:
        num_tokens = hidden_states.shape[0]
        max_num_reqs = self.scheduler_config.max_num_seqs
        num_reqs = min(num_tokens, max_num_reqs)
        min_tokens_per_req = num_tokens // num_reqs
        num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
        num_scheduled_tokens_list[-1] += num_tokens % num_reqs
        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs

        req_num_tokens = num_tokens // num_reqs

        dummy_prompt_lens = torch.tensor(
            num_scheduled_tokens_list,
            device="cpu",
        )
        dummy_token_ids = torch.zeros((num_reqs, req_num_tokens),
                                      dtype=torch.int32,
                                      device=self.device)

        model = cast(VllmModelForPooling, self.get_model())
        dummy_pooling_params = PoolingParams(task=task)
        dummy_pooling_params.verify(task=task, model_config=self.model_config)
        to_update = model.pooler.get_pooling_updates(task)
        to_update.apply(dummy_pooling_params)

        dummy_metadata = PoolingMetadata(
            prompt_lens=dummy_prompt_lens,
            prompt_token_ids=dummy_token_ids,
            pooling_params=[dummy_pooling_params] * num_reqs,
        )

        dummy_metadata.build_pooling_cursor(num_scheduled_tokens_list,
                                            device=hidden_states.device)

        try:
            return model.pooler(hidden_states=hidden_states,
                                pooling_metadata=dummy_metadata)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                raise RuntimeError(
                    "CUDA out of memory occurred when warming up pooler "
                    f"({task=}) with {num_reqs} dummy requests. Please try "
                    "lowering `max_num_seqs` or `gpu_memory_utilization` when "
                    "initializing the engine.") from e
            else:
                raise e

    @torch.inference_mode()
    def _dummy_pooler_run(
        self,
        hidden_states: torch.Tensor,
    ) -> PoolerOutput:
        # Find the task that has the largest output for subsequent steps
        output_size = dict[PoolingTask, float]()
        for task in self.get_supported_pooling_tasks():
            # Run a full batch with each task to ensure none of them OOMs
            output = self._dummy_pooler_run_task(hidden_states, task)
            output_size[task] = sum(o.nbytes for o in output)
            del output  # Allow GC

        max_task = max(output_size.items(), key=lambda x: x[1])[0]
        return self._dummy_pooler_run_task(hidden_states, max_task)

    def profile_run(self) -> None:
        # Profile with multimodal encoder & encoder cache.
        if self.supports_mm_inputs:
            if self.model_config.multimodal_config.skip_mm_profiling:
                logger.info(
                    "Skipping memory profiling for multimodal encoder and "
                    "encoder cache.")
            else:
                mm_budget = self.mm_budget
                assert mm_budget is not None

                if (encoder_budget := mm_budget.get_encoder_budget()) > 0:
                    # NOTE: Currently model is profiled with a single non-text
                    # modality with the max possible input tokens even when
                    # it supports multiple.
                    dummy_modality = mm_budget.get_modality_with_max_tokens()
                    max_mm_items_per_batch = mm_budget \
                        .max_items_per_batch_by_modality[dummy_modality]

                    logger.info(
                        "Encoder cache will be initialized with a budget of "
                        "%s tokens, and profiled with %s %s items of the "
                        "maximum feature size.",
                        encoder_budget,
                        max_mm_items_per_batch,
                        dummy_modality,
                    )

                    # Create dummy batch of multimodal inputs.
                    batched_dummy_mm_inputs = self._get_mm_dummy_batch(
                        dummy_modality,
                        max_mm_items_per_batch,
                    )

                    # Run multimodal encoder.
                    dummy_encoder_outputs = \
                        self.model.get_multimodal_embeddings(
                        **batched_dummy_mm_inputs)

                    sanity_check_mm_encoder_outputs(
                        dummy_encoder_outputs,
                        expected_num_items=max_mm_items_per_batch,
                    )

                    # NOTE: This happens when encoder cache needs to store
                    # the embeddings that encoder outputs are scattered onto.
                    # In this case we create dummy embeddings of size
                    # (encode_budget, hidden_size) and scatter encoder
                    # output into it.
                    encoder_output_shape = dummy_encoder_outputs[0].shape
                    if encoder_output_shape[0] < encoder_budget:
                        expanded_outputs = []
                        for output in dummy_encoder_outputs:
                            expanded = output.new_zeros(
                                (encoder_budget, encoder_output_shape[-1]))
                            num_tokens = output.shape[0]
                            expanded[:num_tokens].copy_(output)
                            expanded_outputs.append(expanded)

                        dummy_encoder_outputs = expanded_outputs

                    # Cache the dummy encoder outputs.
                    self.encoder_cache["tmp"] = dict(
                        enumerate(dummy_encoder_outputs))

        # Add `is_profile` here to pre-allocate communication buffers
        hidden_states, last_hidden_states \
            = self._dummy_run(self.max_num_tokens, is_profile=True)
        if get_pp_group().is_last_rank:
            if self.is_pooling_model:
                output = self._dummy_pooler_run(hidden_states)
            else:
                output = self._dummy_sampler_run(last_hidden_states)
        else:
            output = None
        self._sync_device()
        del hidden_states, output
        self.encoder_cache.clear()
        gc.collect()

    def capture_model(self) -> int:
        if self.compilation_config.cudagraph_mode == CUDAGraphMode.NONE:
            logger.warning(
                "Skipping CUDA graph capture. To turn on CUDA graph capture, "
                "ensure `cudagraph_mode` was not manually set to `NONE`")
            return 0
        else:
            self.initialize_cudagraph_capture()

        compilation_counter.num_gpu_runner_capture_triggers += 1

        start_time = time.perf_counter()
        start_free_gpu_memory = torch.cuda.mem_get_info()[0]

        @contextmanager
        def freeze_gc():
            # Optimize garbage collection during CUDA graph capture.
            # Clean up, then freeze all remaining objects from being included
            # in future collections.
            gc.collect()
            should_freeze = not envs.VLLM_ENABLE_CUDAGRAPH_GC
            if should_freeze:
                gc.freeze()
            try:
                yield
            finally:
                if should_freeze:
                    gc.unfreeze()
                    gc.collect()

        # Trigger CUDA graph capture for specific shapes.
        # Capture the large shapes first so that the smaller shapes
        # can reuse the memory pool allocated for the large shapes.
        set_cudagraph_capturing_enabled(True)
        with freeze_gc(), graph_capture(device=self.device):
            cudagraph_mode = self.compilation_config.cudagraph_mode
            assert cudagraph_mode is not None
            if cudagraph_mode.mixed_mode() != CUDAGraphMode.NONE:
                cudagraph_runtime_mode = cudagraph_mode.mixed_mode()

                compilation_cases = list(reversed(self.cudagraph_batch_sizes))
                self._capture_cudagraphs(
                    compilation_cases,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    uniform_decode=False)

            # Capture full cudagraph for uniform decode batches if we
            # don't already have full mixed prefill-decode cudagraphs.
            if cudagraph_mode.decode_mode() == CUDAGraphMode.FULL and \
                cudagraph_mode.separate_routine():
                max_num_tokens = self.scheduler_config.max_num_seqs * \
                        self.uniform_decode_query_len
                decode_cudagraph_batch_sizes = [
                    x for x in self.cudagraph_batch_sizes if
                    x <= max_num_tokens and x >= self.uniform_decode_query_len
                ]
                compilation_cases_decode = list(
                    reversed(decode_cudagraph_batch_sizes))
                self._capture_cudagraphs(
                    compilation_cases=compilation_cases_decode,
                    cudagraph_runtime_mode=CUDAGraphMode.FULL,
                    uniform_decode=True)

        # Disable cudagraph capturing globally, so any unexpected cudagraph
        # capturing will be detected and raise an error after here.
        # Note: We don't put it into graph_capture context manager because
        # we may do lazy capturing in future that still allows capturing
        # after here.
        set_cudagraph_capturing_enabled(False)

        end_time = time.perf_counter()
        end_free_gpu_memory = torch.cuda.mem_get_info()[0]
        elapsed_time = end_time - start_time
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
        # This usually takes 5~20 seconds.
        logger.info("Graph capturing finished in %.0f secs, took %.2f GiB",
                    elapsed_time, cuda_graph_size / (1 << 30))
        return cuda_graph_size

    def _capture_cudagraphs(self, compilation_cases: list[int],
                            cudagraph_runtime_mode: CUDAGraphMode,
                            uniform_decode: bool):
        assert cudagraph_runtime_mode != CUDAGraphMode.NONE and \
            cudagraph_runtime_mode in [CUDAGraphMode.FULL,
                                        CUDAGraphMode.PIECEWISE]

        # Only rank 0 should print progress bar during capture
        if is_global_first_rank():
            compilation_cases = tqdm(
                compilation_cases,
                disable=not self.load_config.use_tqdm_on_load,
                desc="Capturing CUDA graphs ({}, {})".format(
                    "decode" if uniform_decode else "mixed prefill-decode",
                    cudagraph_runtime_mode.name))

        # We skip EPLB here since we don't want to record dummy metrics
        for num_tokens in compilation_cases:
            # We currently only capture ubatched graphs when its a FULL
            # cudagraph, a uniform decode batch, and the number of tokens
            # is above the threshold. Otherwise we just capture a non-ubatched
            # version of the graph
            allow_microbatching = self.parallel_config.enable_dbo \
                and cudagraph_runtime_mode == CUDAGraphMode.FULL \
                and uniform_decode \
                and check_ubatch_thresholds(
                    config=self.vllm_config.parallel_config,
                    num_tokens=num_tokens,
                    uniform_decode=uniform_decode,
                )

            for _ in range(self.compilation_config.cudagraph_num_of_warmups):
                # Use CUDAGraphRuntimeStyle.NONE (default) for warmup.
                # But be careful, warm up with `NONE`is orthogonal to
                # if we want to warm up attention or not. This is
                # different from the case where `FULL` implies capture
                # attention while `PIECEWISE` implies no attention.
                force_attention = (
                    cudagraph_runtime_mode == CUDAGraphMode.FULL)
                self._dummy_run(num_tokens,
                                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                                force_attention=force_attention,
                                uniform_decode=uniform_decode,
                                allow_microbatching=allow_microbatching,
                                skip_eplb=True,
                                remove_lora=False)
            self._dummy_run(num_tokens,
                            cudagraph_runtime_mode=cudagraph_runtime_mode,
                            uniform_decode=uniform_decode,
                            allow_microbatching=allow_microbatching,
                            skip_eplb=True,
                            remove_lora=False)
        self.maybe_remove_all_loras(self.lora_config)

    def initialize_attn_backend(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize the attention backends and attention metadata builders.
        """
        assert len(self.attn_groups) == 0, \
            "Attention backends are already initialized"

        class AttentionGroupKey(NamedTuple):
            attn_backend: type[AttentionBackend]
            kv_cache_spec: KVCacheSpec

        def get_attn_backends_for_group(
            kv_cache_group_spec: KVCacheGroupSpec,
        ) -> dict[AttentionGroupKey, list[str]]:
            layers = get_layers_from_vllm_config(
                self.vllm_config, AttentionLayerBase,
                kv_cache_group_spec.layer_names)
            attn_backends = {}
            attn_backend_layers = defaultdict(list)
            # Dedupe based on full class name; this is a bit safer than
            # using the class itself as the key because when we create dynamic
            # attention backend subclasses (e.g. ChunkedLocalAttention) unless
            # they are cached correctly, there will be different objects per
            # layer.
            for layer_name in kv_cache_group_spec.layer_names:
                attn_backend = layers[layer_name].get_attn_backend()

                if layer_name in self.kv_sharing_fast_prefill_eligible_layers:
                    attn_backend = create_fast_prefill_custom_backend(
                        "FastPrefill",
                        attn_backend,
                    )

                full_cls_name = attn_backend.full_cls_name()
                layer_kv_cache_spec = kv_cache_group_spec.kv_cache_spec
                if isinstance(layer_kv_cache_spec, UniformTypeKVCacheSpecs):
                    layer_kv_cache_spec = layer_kv_cache_spec.kv_cache_specs[
                        layer_name]
                key = (full_cls_name, layer_kv_cache_spec)
                attn_backends[key] = AttentionGroupKey(attn_backend,
                                                       layer_kv_cache_spec)
                attn_backend_layers[key].append(layer_name)
            return {
                attn_backends[k]: v
                for k, v in attn_backend_layers.items()
            }

        def create_attn_groups(
            attn_backends_map: dict[AttentionGroupKey, list[str]],
        ) -> list[AttentionGroup]:
            attn_groups: list[AttentionGroup] = []
            for (attn_backend,
                 kv_cache_spec), layer_names in attn_backends_map.items():
                attn_group = AttentionGroup.create_with_metadata_builders(
                    attn_backend,
                    layer_names,
                    kv_cache_spec,
                    self.vllm_config,
                    self.device,
                    num_metadata_builders=1
                    if not self.parallel_config.enable_dbo else 2,
                )

                attn_groups.append(attn_group)
            return attn_groups

        for kv_cache_group_spec in kv_cache_config.kv_cache_groups:
            attn_backends = get_attn_backends_for_group(kv_cache_group_spec)
            self.attn_groups.append(create_attn_groups(attn_backends))

        # Calculate reorder batch threshold (if needed)
        self.calculate_reorder_batch_threshold()

    def initialize_cudagraph_capture(self) -> None:
        min_cg_support = AttentionCGSupport.ALWAYS
        min_cg_builder_name = None

        for attn_group in self._attn_group_iterator():
            builder = attn_group.get_metadata_builder()
            if builder.cudagraph_support.value < min_cg_support.value:
                min_cg_support = builder.cudagraph_support
                min_cg_builder_name = builder.__class__.__name__
        # Flexible resolve the cudagraph mode
        cudagraph_mode = self.compilation_config.cudagraph_mode
        # check cudagraph for mixed batch is supported
        if cudagraph_mode.mixed_mode() == CUDAGraphMode.FULL \
            and min_cg_support != AttentionCGSupport.ALWAYS:
            msg = (f"CUDAGraphMode.{cudagraph_mode.name} is not supported "
                   f"with {min_cg_builder_name} backend (support: "
                   f"{min_cg_support})")
            if min_cg_support == AttentionCGSupport.NEVER:
                # if not supported any full cudagraphs, just raise it.
                msg += "; please try cudagraph_mode=PIECEWISE, and "\
                    "make sure compilation level is piecewise"
                raise ValueError(msg)

            # attempt to resolve the full cudagraph related mode
            if self.compilation_config.splitting_ops_contain_attention():
                msg += "; setting cudagraph_mode=FULL_AND_PIECEWISE"
                cudagraph_mode = self.compilation_config.cudagraph_mode = \
                    CUDAGraphMode.FULL_AND_PIECEWISE
            else:
                msg += "; setting cudagraph_mode=FULL_DECODE_ONLY"
                cudagraph_mode = self.compilation_config.cudagraph_mode = \
                    CUDAGraphMode.FULL_DECODE_ONLY
            logger.warning(msg)

        # check that if we are doing decode full-cudagraphs it is supported
        if (cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
                and min_cg_support == AttentionCGSupport.NEVER):
            msg = (f"CUDAGraphMode.{cudagraph_mode.name} is not supported "
                   f"with {min_cg_builder_name} backend (support: "
                   f"{min_cg_support})")
            if (self.compilation_config.level == CompilationLevel.PIECEWISE and
                (self.compilation_config.splitting_ops_contain_attention()
                 or self.compilation_config.use_inductor_graph_partition)):
                msg += "; setting cudagraph_mode=PIECEWISE because "\
                    "attention is compiled piecewise"
                cudagraph_mode = self.compilation_config.cudagraph_mode = \
                    CUDAGraphMode.PIECEWISE
            else:
                msg += "; setting cudagraph_mode=NONE because "\
                    "attention is not compiled piecewise"
                cudagraph_mode = self.compilation_config.cudagraph_mode = \
                    CUDAGraphMode.NONE
            logger.warning(msg)

        # check that if we are doing spec-decode + decode full-cudagraphs it is
        # supported
        if (cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
                and self.uniform_decode_query_len > 1 and min_cg_support.value
                < AttentionCGSupport.UNIFORM_BATCH.value):
            msg = (f"CUDAGraphMode.{cudagraph_mode.name} is not supported"
                   f" with spec-decode for attention backend "
                   f"{min_cg_builder_name} (support: {min_cg_support})")
            if self.compilation_config.splitting_ops_contain_attention():
                msg += "; setting cudagraph_mode=PIECEWISE"
                cudagraph_mode = self.compilation_config.cudagraph_mode = \
                    CUDAGraphMode.PIECEWISE
            else:
                msg += "; setting cudagraph_mode=NONE"
                cudagraph_mode = self.compilation_config.cudagraph_mode = \
                    CUDAGraphMode.NONE
            logger.warning(msg)

        # double check that we can support full cudagraph if they are requested
        # even after automatic downgrades
        if cudagraph_mode.has_full_cudagraphs() \
            and min_cg_support == AttentionCGSupport.NEVER:
            raise ValueError(f"CUDAGraphMode.{cudagraph_mode.name} is not "
                             f"supported with {min_cg_builder_name} backend ("
                             f"support:{min_cg_support}) "
                             "; please try cudagraph_mode=PIECEWISE, "
                             "and make sure compilation level is piecewise")

        # Trigger cudagraph dispatching keys initialization here (after
        # initializing attn backends).
        self.cudagraph_dispatcher.initialize_cudagraph_keys(
            self.compilation_config.cudagraph_mode,
            self.uniform_decode_query_len)

    def calculate_reorder_batch_threshold(self) -> None:
        """
        Check that if any backends reorder batches; that the reordering
        is compatible (e.g., decode threshold is the same)
        """
        for group in self._attn_group_iterator():
            attn_metadata_builder_i = group.get_metadata_builder()

            # check that if any backends reorder batches; that the reordering
            # is compatible (e.g., decode threshold is the same)
            reorder_batch_threshold_i = (
                attn_metadata_builder_i.reorder_batch_threshold)
            if reorder_batch_threshold_i is not None:
                if self.reorder_batch_threshold is not None:
                    if reorder_batch_threshold_i != \
                        self.reorder_batch_threshold:
                        raise ValueError(
                            f"Attention backend reorders decodes with "
                            f"threshold {reorder_batch_threshold_i} but other "
                            f"backend uses threshold "
                            f"{self.reorder_batch_threshold}")
                else:
                    self.reorder_batch_threshold = reorder_batch_threshold_i

    def may_reinitialize_input_batch(self,
                                     kv_cache_config: KVCacheConfig) -> None:
        """
        Re-initialize the input batch if the block sizes are different from
        `[self.cache_config.block_size]`. This usually happens when there
        are multiple KV cache groups.

        Args:
            kv_cache_config: The KV cache configuration.
        """
        block_sizes = [
            kv_cache_group.kv_cache_spec.block_size
            for kv_cache_group in kv_cache_config.kv_cache_groups
        ]
        if block_sizes != [self.cache_config.block_size]:
            assert self.cache_config.cpu_offload_gb == 0, (
                "Cannot re-initialize the input batch when CPU weight "
                "offloading is enabled. See https://github.com/vllm-project/vllm/pull/18298 "  # noqa: E501
                "for more details.")
            self.input_batch = InputBatch(
                max_num_reqs=self.max_num_reqs,
                max_model_len=max(self.max_model_len, self.max_encoder_len),
                max_num_batched_tokens=self.max_num_tokens,
                device=self.device,
                pin_memory=self.pin_memory,
                vocab_size=self.model_config.get_vocab_size(),
                block_sizes=block_sizes,
                is_spec_decode=bool(self.vllm_config.speculative_config),
                logitsprocs=self.input_batch.logitsprocs,
                is_pooling_model=self.is_pooling_model,
                num_speculative_tokens=(
                    self.vllm_config.speculative_config.num_speculative_tokens
                    if self.vllm_config.speculative_config else 0),
            )

    def _allocate_kv_cache_tensors(
            self, kv_cache_config: KVCacheConfig) -> dict[str, torch.Tensor]:
        """
        Initializes the KV cache buffer with the correct size. The buffer needs
        to be reshaped to the desired shape before being used by the models.

        Args:
            kv_cache_config: The KV cache config
        Returns:
            dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
         """
        kv_cache_raw_tensors: dict[str, torch.Tensor] = {}
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            tensor = torch.zeros(kv_cache_tensor.size,
                                 dtype=torch.int8,
                                 device=self.device)
            for layer_name in kv_cache_tensor.shared_by:
                kv_cache_raw_tensors[layer_name] = tensor

        layer_names = set()
        for group in kv_cache_config.kv_cache_groups:
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                layer_names.add(layer_name)
        assert layer_names == set(kv_cache_raw_tensors.keys(
        )), "Some layers are not correctly initialized"
        return kv_cache_raw_tensors

    def _attn_group_iterator(self) -> Iterator[AttentionGroup]:
        return itertools.chain.from_iterable(self.attn_groups)

    def _kv_cache_spec_attn_group_iterator(self) -> Iterator[AttentionGroup]:
        if not self.kv_cache_config.kv_cache_groups:
            return
        for attn_groups in self.attn_groups:
            yield from attn_groups

    def _reshape_kv_cache_tensors(
        self,
        kv_cache_config: KVCacheConfig,
        kv_cache_raw_tensors: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Reshape the KV cache tensors to the desired shape and dtype.

        Args:
            kv_cache_config: The KV cache config
            kv_cache_raw_tensors: The KV cache buffer of each layer, with
                correct size but uninitialized shape.
        Returns:
            Dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        kv_caches: dict[str, torch.Tensor] = {}
        has_attn, has_mamba = False, False
        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            attn_backend = group.backend
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                raw_tensor = kv_cache_raw_tensors[layer_name]
                assert raw_tensor.numel() % kv_cache_spec.page_size_bytes == 0
                num_blocks = (raw_tensor.numel() //
                              kv_cache_spec.page_size_bytes)
                if isinstance(kv_cache_spec, AttentionSpec):
                    has_attn = True
                    kv_cache_shape = attn_backend.get_kv_cache_shape(
                        num_blocks,
                        kv_cache_spec.block_size,
                        kv_cache_spec.num_kv_heads,
                        kv_cache_spec.head_size,
                        cache_dtype_str=self.cache_config.cache_dtype)
                    dtype = kv_cache_spec.dtype
                    try:
                        kv_cache_stride_order = \
                            attn_backend.get_kv_cache_stride_order()
                        assert len(kv_cache_stride_order) == len(
                            kv_cache_shape)
                    except (AttributeError, NotImplementedError):
                        kv_cache_stride_order = tuple(
                            range(len(kv_cache_shape)))
                    # The allocation respects the backend-defined stride order
                    # to ensure the semantic remains consistent for each
                    # backend. We first obtain the generic kv cache shape and
                    # then permute it according to the stride order which could
                    # result in a non-contiguous tensor.
                    kv_cache_shape = tuple(kv_cache_shape[i]
                                           for i in kv_cache_stride_order)
                    # Maintain original KV shape view.
                    inv_order = [
                        kv_cache_stride_order.index(i)
                        for i in range(len(kv_cache_stride_order))
                    ]
                    kv_caches[layer_name] = kv_cache_raw_tensors[
                        layer_name].view(dtype).view(kv_cache_shape).permute(
                            *inv_order)
                elif isinstance(kv_cache_spec, MambaSpec):
                    has_mamba = True
                    raw_tensor = kv_cache_raw_tensors[layer_name]
                    state_tensors = []
                    storage_offset_bytes = 0
                    for (shape, dtype) in zip(kv_cache_spec.shapes,
                                              kv_cache_spec.dtypes):
                        dtype_size = get_dtype_size(dtype)
                        num_element_per_page = (
                            kv_cache_spec.page_size_bytes // dtype_size)
                        target_shape = (num_blocks, *shape)
                        stride = torch.empty(target_shape).stride()
                        target_stride = (num_element_per_page, *stride[1:])
                        assert storage_offset_bytes % dtype_size == 0
                        tensor = torch.as_strided(
                            raw_tensor.view(dtype),
                            size=target_shape,
                            stride=target_stride,
                            storage_offset=storage_offset_bytes // dtype_size,
                        )
                        state_tensors.append(tensor)
                        storage_offset_bytes += stride[0] * dtype_size

                    kv_caches[layer_name] = state_tensors
                else:
                    raise NotImplementedError

        if has_attn and has_mamba:
            self._update_hybrid_attention_mamba_layout(kv_caches)

        return kv_caches

    def _update_hybrid_attention_mamba_layout(
            self, kv_caches: dict[str, torch.Tensor]) -> None:
        """
        Update the layout of attention layers from (2, num_blocks, ...) to
        (num_blocks, 2, ...).

        Args:
            kv_caches: The KV cache buffer of each layer.
        """

        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            for layer_name in group.layer_names:
                kv_cache = kv_caches[layer_name]
                if (isinstance(kv_cache_spec, AttentionSpec)
                        and kv_cache.shape[0] == 2):
                    assert kv_cache.shape[1] != 2, \
                        "Fail to determine whether the layout is " \
                        "(2, num_blocks, ...) or (num_blocks, 2, ...) for " \
                        f"a tensor of shape {kv_cache.shape}"
                    hidden_size = kv_cache.shape[2:].numel()
                    kv_cache.as_strided_(size=kv_cache.shape,
                                         stride=(hidden_size, 2 * hidden_size,
                                                 *kv_cache.stride()[2:]))

    def initialize_kv_cache_tensors(
            self, kv_cache_config: KVCacheConfig) -> dict[str, torch.Tensor]:
        """
        Initialize the memory buffer for KV cache.

        Args:
            kv_cache_config: The KV cache config
        Returns:
            Dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        # Initialize the memory buffer for KV cache
        kv_cache_raw_tensors = self._allocate_kv_cache_tensors(kv_cache_config)
        # Change the memory buffer to the desired shape
        kv_caches = self._reshape_kv_cache_tensors(kv_cache_config,
                                                   kv_cache_raw_tensors)

        # Set up cross-layer KV cache sharing
        for layer_name, target_layer_name in self.shared_kv_cache_layers.items(
        ):
            logger.debug("%s reuses KV cache of %s", layer_name,
                         target_layer_name)
            kv_caches[layer_name] = kv_caches[target_layer_name]

        num_attn_module = 2 \
            if self.model_config.hf_config.model_type == "longcat_flash" else 1
        bind_kv_cache(kv_caches,
                      self.compilation_config.static_forward_context,
                      self.kv_caches, num_attn_module)
        return kv_caches

    def maybe_add_kv_sharing_layers_to_kv_cache_groups(
            self, kv_cache_config: KVCacheConfig) -> None:
        """
        Add layers that re-use KV cache to KV cache group of its target layer.
        Mapping of KV cache tensors happens in `initialize_kv_cache_tensors()`
        """
        if not self.shared_kv_cache_layers:
            # No cross-layer KV sharing, return
            return

        add_kv_sharing_layers_to_kv_cache_groups(
            self.shared_kv_cache_layers,
            kv_cache_config.kv_cache_groups,
            self.runner_only_attn_layers,
        )

        if self.cache_config.kv_sharing_fast_prefill:
            # In You Only Cache Once (https://arxiv.org/abs/2405.05254) or other
            # similar KV sharing setups, only the layers that generate KV caches
            # are involved in the prefill phase, enabling prefill to early exit.
            attn_layers = get_layers_from_vllm_config(self.vllm_config,
                                                      Attention)
            for layer_name in reversed(attn_layers):
                if layer_name in self.shared_kv_cache_layers:
                    self.kv_sharing_fast_prefill_eligible_layers.add(
                        layer_name)
                else:
                    break

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        kv_cache_config = deepcopy(kv_cache_config)
        self.kv_cache_config = kv_cache_config
        # Set num_gpu_blocks on cache_config now (the engine sets it via
        # a separate "initialize_cache" RPC later, AFTER capture_model has
        # already run). FlexAttention's metadata builder needs it at
        # capture time (total_cache_tokens drives the static block_mask
        # bound); without this early assignment the assertion in build()
        # fires during the FULL_DECODE_ONLY dummy_run.
        self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks
        self.may_reinitialize_input_batch(kv_cache_config)
        self.may_add_encoder_only_layers_to_kv_cache_config()
        self.maybe_add_kv_sharing_layers_to_kv_cache_groups(kv_cache_config)
        self.initialize_attn_backend(kv_cache_config)
        kv_caches = self.initialize_kv_cache_tensors(kv_cache_config)

        if self.speculative_config and self.speculative_config.use_eagle():
            assert isinstance(self.drafter, EagleProposer)
            # validate all draft model layers belong to the same kv cache
            # group
            self.drafter.validate_same_kv_cache_group(kv_cache_config)

        if has_kv_transfer_group():
            get_kv_transfer_group().register_kv_caches(kv_caches)
            if self.device.type == 'xpu':
                get_kv_transfer_group().set_host_xfer_buffer_ops(
                    copy_kv_blocks)

        if self.dcp_world_size > 1:
            layer_names = self.attn_groups[0][0].layer_names
            layers = get_layers_from_vllm_config(self.vllm_config,
                                                 AttentionLayerBase,
                                                 layer_names)
            for layer in layers.values():
                assert layer.impl.need_to_return_lse_for_decode, (
                    "DCP requires attention impls to return"
                    " the softmax lse for decode, but the impl "
                    f"{layer.impl.__class__.__name__} "
                    "does not return the softmax lse for decode.")

    def may_add_encoder_only_layers_to_kv_cache_config(self) -> None:
        """
        Add encoder-only layers to the KV cache config.
        """
        block_size = self.vllm_config.cache_config.block_size
        encoder_only_attn_specs: dict[AttentionSpec,
                                      list[str]] = defaultdict(list)
        attn_layers = get_layers_from_vllm_config(self.vllm_config, Attention)
        for layer_name, attn_module in attn_layers.items():
            if attn_module.attn_type == AttentionType.ENCODER_ONLY:
                attn_spec: AttentionSpec = EncoderOnlyAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=attn_module.num_kv_heads,
                    head_size=attn_module.head_size,
                    dtype=self.kv_cache_dtype)
                encoder_only_attn_specs[attn_spec].append(layer_name)
                self.runner_only_attn_layers.add(layer_name)
        if len(encoder_only_attn_specs) > 0:
            assert len(
                encoder_only_attn_specs
            ) == 1, "Only support one encoder-only attention spec now"
            spec, layer_names = encoder_only_attn_specs.popitem()
            self.kv_cache_config.kv_cache_groups.append(
                KVCacheGroupSpec(layer_names=layer_names, kv_cache_spec=spec))

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """

        block_size = self.vllm_config.cache_config.block_size
        use_mla = self.vllm_config.model_config.use_mla
        cache_dtype_str = self.vllm_config.cache_config.cache_dtype
        kv_cache_spec: dict[str, KVCacheSpec] = {}
        attn_layers = get_layers_from_vllm_config(self.vllm_config, Attention)
        for layer_name, attn_module in attn_layers.items():
            if (kv_tgt_layer :=
                    attn_module.kv_sharing_target_layer_name) is not None:
                # The layer doesn't need its own KV cache and will use that of
                # the target layer. We skip creating a KVCacheSpec for it, so
                # that KV cache management logic will act as this layer does
                # not exist, and doesn't allocate KV cache for the layer. This
                # enables the memory saving of cross-layer kv sharing, allowing
                # a given amount of memory to accommodate longer context lengths
                # or enable more requests to be processed simultaneously.
                self.shared_kv_cache_layers[layer_name] = kv_tgt_layer
                continue

            # TODO(lucas): move the attention specs into the model layers like
            # the attention backends
            if attn_module.attn_type == AttentionType.DECODER:
                if attn_module.sliding_window is not None:
                    assert not use_mla, "MLA is not supported for sliding" \
                        "window"
                    kv_cache_spec[layer_name] = SlidingWindowSpec(
                        block_size=block_size,
                        num_kv_heads=attn_module.num_kv_heads,
                        head_size=attn_module.head_size,
                        dtype=self.kv_cache_dtype,
                        sliding_window=attn_module.sliding_window)
                elif use_mla:
                    kv_cache_spec[layer_name] = MLAAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=attn_module.num_kv_heads,
                        head_size=attn_module.head_size,
                        dtype=self.kv_cache_dtype,
                        cache_dtype_str=cache_dtype_str)
                elif self.attention_chunk_size is not None \
                        and isinstance(attn_module, ChunkedLocalAttention):
                    kv_cache_spec[layer_name] = ChunkedLocalAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=attn_module.num_kv_heads,
                        head_size=attn_module.head_size,
                        dtype=self.kv_cache_dtype,
                        attention_chunk_size=self.attention_chunk_size)
                else:
                    kv_cache_spec[layer_name] = FullAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=attn_module.num_kv_heads,
                        head_size=attn_module.head_size,
                        dtype=self.kv_cache_dtype)
            elif attn_module.attn_type == AttentionType.ENCODER_DECODER:
                kv_cache_spec[layer_name] = CrossAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=attn_module.num_kv_heads,
                    head_size=attn_module.head_size,
                    dtype=self.kv_cache_dtype)
            elif attn_module.attn_type in (AttentionType.ENCODER,
                                           AttentionType.ENCODER_ONLY):
                # encoder-only attention does not need KV cache.
                continue
            else:
                raise ValueError(
                    f"Unknown attention type: {attn_module.attn_type}")

        mamba_layers = get_layers_from_vllm_config(self.vllm_config, MambaBase)
        if len(mamba_layers) > 0:
            model_type = self.vllm_config.model_config.hf_config.model_type
            allow_mamba_spec_decode = (
                model_type == "qwen3_next"
                or (self.speculative_config is not None
                    and self.speculative_config.use_tidar()
                    and model_type == "smoe"))
            if (self.vllm_config.speculative_config is not None
                    and not allow_mamba_spec_decode):
                raise NotImplementedError(
                    "Mamba with speculative decoding is not supported yet.")
            if self.vllm_config.cache_config.enable_prefix_caching:
                raise NotImplementedError(
                    "Prefix caching is not supported for Mamba yet.")
            max_model_len = self.vllm_config.model_config.max_model_len

            page_size_padded = (
                self.vllm_config.cache_config.mamba_page_size_padded)

            # Set block_size to max_model_len, so that mamba model will always
            # have only one block in the KV cache.
            for layer_name, mamba_module in mamba_layers.items():
                num_speculative_blocks = (
                    self.speculative_config.num_speculative_tokens
                    if self.speculative_config else 0)
                if (self.speculative_config is not None
                        and self.speculative_config.use_tidar()
                        and mamba_module.mamba_type == "cca"):
                    # TiDAR only needs one scratch CCA state per request.
                    num_speculative_blocks = 1
                kv_cache_spec[layer_name] = MambaSpec(
                    shapes=mamba_module.get_state_shape(),
                    dtypes=mamba_module.get_state_dtype(),
                    block_size=max_model_len,
                    page_size_padded=page_size_padded,
                    mamba_type=mamba_module.mamba_type,
                    num_speculative_blocks=num_speculative_blocks,
                )
        ds_indexer_layers = get_layers_from_vllm_config(
            self.vllm_config, DeepseekV32IndexerCache)
        for layer_name, ds_indexer_module in ds_indexer_layers.items():
            kv_cache_spec[layer_name] = ds_indexer_module.get_kv_cache_spec()

        return kv_cache_spec

    def _to_list(self, sampled_token_ids: torch.Tensor) -> list[list[int]]:
        # This is a short term mitigation for issue mentioned in
        # https://github.com/vllm-project/vllm/issues/22754.
        # `tolist` would trigger a cuda wise stream sync, which
        # would block other copy ops from other cuda streams.
        # A cuda event sync would avoid such a situation. Since
        # this is in the critical path of every single model
        # forward loop, this has caused perf issue for a disagg
        # setup.
        pinned = self.sampled_token_ids_pinned_cpu[:sampled_token_ids.shape[0]]
        pinned.copy_(sampled_token_ids, non_blocking=True)
        self.transfer_event.record()
        self.transfer_event.synchronize()
        return pinned.tolist()
