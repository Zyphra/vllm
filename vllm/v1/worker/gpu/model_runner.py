# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
import os
import time
from contextlib import nullcontext
from copy import deepcopy
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.eplb.eplb_state import EplbState
from vllm.distributed.parallel_state import (
    get_dp_group,
    prepare_communication_buffer_for_model,
)
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
    select_routed_experts_kv_group,
)
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.models.interfaces import is_mixture_of_experts
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.utils.mem_utils import DeviceMemoryProfiler, format_gib
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.worker.gpu.async_utils import AsyncOutput
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    build_slot_mappings_by_layer,
    get_kv_cache_spec,
    init_attn_backend,
    init_kv_cache,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
from vllm.v1.worker.gpu.dp_utils import (
    get_cudagraph_and_dp_padding,
    make_num_tokens_across_dp,
)
from vllm.v1.worker.gpu.input_batch import (
    InputBatch,
    InputBuffers,
    combine_sampled_and_draft_tokens,
    expand_idx_mapping,
    get_num_sampled_and_rejected,
    post_update,
    prepare_pos_seq_lens,
    prepare_prefill_inputs,
)
from vllm.v1.worker.gpu.kv_connector import (
    NO_OP_KV_CONNECTOR,
    KVConnector,
    get_kv_connector,
)
from vllm.v1.worker.gpu.lora_utils import LoraState
from vllm.v1.worker.gpu.mm.encoder_runner import EncoderRunner
from vllm.v1.worker.gpu.mm.mrope_utils import MRopeState
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.prompt_logprob import PromptLogprobsWorker
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.spec_decode import init_speculator
from vllm.v1.worker.gpu.spec_decode.rejection_sample import rejection_sample
from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.gpu.structured_outputs import StructuredOutputsWorker
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin

logger = init_logger(__name__)


class GPUModelRunner(LoRAModelRunnerMixin):
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

        self.device = device
        self.dtype = self.model_config.dtype
        self.kv_cache_dtype = self.dtype
        if self.cache_config.cache_dtype != "auto":
            # Quantized KV cache.
            self.kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
                self.cache_config.cache_dtype
            ]
        self.is_pooling_model = False

        self.vocab_size = self.model_config.get_vocab_size()
        self.max_model_len = self.model_config.max_model_len
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.inputs_embeds_size = self.model_config.get_inputs_embeds_size()

        # Multimodal
        self.mm_registry = MULTIMODAL_REGISTRY
        self.supports_mm_inputs = self.mm_registry.supports_multimodal_inputs(
            self.model_config
        )
        if self.supports_mm_inputs:
            self.encoder_runner = EncoderRunner(
                max_num_tokens=self.max_num_tokens,
                hidden_size=self.inputs_embeds_size,
                dtype=self.dtype,
                device=self.device,
            )
        self.uses_mrope = self.model_config.uses_mrope
        if self.uses_mrope:
            self.mrope_states = MRopeState(
                max_num_reqs=self.max_num_reqs,
                max_num_tokens=self.max_num_tokens,
                max_model_len=self.max_model_len,
                device=self.device,
            )

        self.use_async_scheduling = self.scheduler_config.async_scheduling
        self.output_copy_stream = torch.cuda.Stream(self.device)
        self.output_copy_event = torch.cuda.Event()

        if self.speculative_config is not None:
            self.do_spec_decode = True
            self.num_speculative_steps = self.speculative_config.num_speculative_tokens
            self.speculator = init_speculator(self.vllm_config, self.device)
        else:
            self.do_spec_decode = False
            self.num_speculative_steps = 0
            self.speculator = None

        self.req_states = RequestState(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            num_speculative_steps=self.num_speculative_steps,
            vocab_size=self.vocab_size,
            device=self.device,
        )
        self.input_buffers = InputBuffers(
            max_num_reqs=self.max_num_reqs,
            max_num_tokens=self.max_num_tokens,
            device=self.device,
        )
        self.sampler = Sampler(
            max_num_reqs=self.max_num_reqs,
            vocab_size=self.vocab_size,
            device=self.device,
            logprobs_mode=self.model_config.logprobs_mode,
            num_speculative_tokens=self.num_speculative_steps + 1,
        )
        self.prompt_logprobs_worker = PromptLogprobsWorker(self.max_num_reqs)

        # CUDA graphs.
        self.cudagraph_manager = CudaGraphManager(
            self.vllm_config, self.uses_mrope, self.device
        )
        # Structured outputs worker.
        self.structured_outputs_worker = StructuredOutputsWorker(
            max_num_logits=self.max_num_reqs * (self.num_speculative_steps + 1),
            vocab_size=self.vocab_size,
            device=self.device,
        )
        # LoRA-related workers.
        self.lora_state = LoraState(max_num_reqs=self.max_num_reqs)

        # Draft tokens propagation - for spec-dec + struct outputs.
        self.draft_tokens_handler = DraftTokensHandler(self.device)
        self._tidar_cca_commit_ctx: dict[str, Any] | None = None
        self._tidar_cca_layers: dict[str, Any] | None = None
        self._tidar_cca_state_slots: torch.Tensor | None = None
        self._tidar_cca_group_ids: set[int] | None = None
        self._tidar_draft_real_across_dp: torch.Tensor | None = None
        self._tidar_draft_eff_across_dp: torch.Tensor | None = None
        self._tidar_draft_should_run = False
        self._tidar_draft_coordinate_ran = False
        self.eplb_state: EplbState | None = None

        # KV Connector if configured.
        self.kv_connector: KVConnector = NO_OP_KV_CONNECTOR

    def update_max_model_len(self, max_model_len: int) -> None:
        self.max_model_len = max_model_len
        self.req_states.max_model_len = max_model_len

    @staticmethod
    def get_supported_tasks() -> tuple[str]:
        return ("generate",)

    def load_model(self, *args, **kwargs) -> None:
        if self.parallel_config.enable_eplb:
            self.eplb_state = EplbState(self.parallel_config, self.device)

        time_before_load = time.perf_counter()
        with DeviceMemoryProfiler() as m:
            model_loader = get_model_loader(self.vllm_config.load_config)
            logger.info("Loading model from scratch...")

            self.model = model_loader.load_model(
                vllm_config=self.vllm_config,
                model_config=self.vllm_config.model_config,
            )
            if self.lora_config:
                self.model = self.load_lora_model(
                    self.model, self.vllm_config, self.device
                )
            if self.do_spec_decode:
                self.speculator.load_model(self.model)
        time_after_load = time.perf_counter()

        self.model_memory_usage = m.consumed_memory
        logger.info(
            "Model loading took %s GiB and %.6f seconds",
            format_gib(m.consumed_memory),
            time_after_load - time_before_load,
        )

        prepare_communication_buffer_for_model(self.model)
        if self.do_spec_decode:
            speculator_model = getattr(self.speculator, "model", None)
            if speculator_model is not None:
                prepare_communication_buffer_for_model(speculator_model)
                if (speculator_model is not self.model
                        and is_mixture_of_experts(speculator_model)
                        and self.parallel_config.enable_eplb):
                    spec_config = self.vllm_config.speculative_config
                    assert spec_config is not None
                    assert spec_config.draft_model_config is not None
                    logger.info("EPLB is enabled for drafter model %s.",
                                spec_config.draft_model_config.model)
                    assert self.eplb_state is not None
                    self.eplb_state.add_model(
                        speculator_model,
                        spec_config.draft_model_config,
                    )

        if is_mixture_of_experts(self.model) and self.parallel_config.enable_eplb:
            logger.info("EPLB is enabled for model %s.", self.model_config.model)
            assert self.eplb_state is not None
            self.eplb_state.add_model(self.model, self.model_config)
            if self.eplb_state.is_async:
                self.eplb_state.start_async_loop(rank_mapping=None)

    def get_model(self) -> nn.Module:
        return self.model

    def _dp_debug(self, message: str, *args: object) -> None:
        if os.environ.get("VLLM_TIDAR_V2_DP_DEBUG") != "1":
            return
        logger.info("[V2DPDBG rank=%d] " + message,
                    self.parallel_config.data_parallel_rank, *args)

    def eplb_step(self,
                  is_dummy: bool = False,
                  is_profile: bool = False,
                  skip_if_all_dummy: bool = False) -> None:
        if not self.parallel_config.enable_eplb:
            return

        assert self.eplb_state is not None
        model = self.get_model()
        assert is_mixture_of_experts(model)
        if skip_if_all_dummy and self.parallel_config.data_parallel_size > 1:
            has_real_work = torch.tensor(
                [0 if is_dummy else 1], dtype=torch.int32, device="cpu"
            )
            torch.distributed.all_reduce(has_real_work, group=get_dp_group().cpu_group)
            if int(has_real_work.item()) == 0:
                self._dp_debug("eplb_skip_all_dummy")
                return

        self.eplb_state.step(
            is_dummy,
            is_profile,
            log_stats=self.parallel_config.eplb_config.log_balancedness,
        )

    def get_kv_cache_spec(self):
        return get_kv_cache_spec(self.vllm_config)

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        kv_cache_config = deepcopy(kv_cache_config)
        self.kv_cache_config = kv_cache_config
        block_sizes = [
            kv_cache_group.kv_cache_spec.block_size
            for kv_cache_group in kv_cache_config.kv_cache_groups
        ]

        self.block_tables = BlockTables(
            block_sizes=block_sizes,
            max_num_reqs=self.max_num_reqs,
            max_num_batched_tokens=self.max_num_tokens,
            max_model_len=self.max_model_len,
            device=self.device,
        )

        self.attn_backends, self.attn_metadata_builders = init_attn_backend(
            self.kv_cache_config, self.vllm_config, self.device
        )
        if self.do_spec_decode:
            # HACK(woosuk)
            self.speculator.set_attn(
                self.kv_cache_config,
                self.attn_metadata_builders,
                self.block_tables,
            )

        self.kv_caches: list[torch.Tensor] = []
        kv_caches_dict = init_kv_cache(
            self.kv_caches,
            self.compilation_config.static_forward_context,
            self.kv_cache_config,
            self.attn_backends,
            self.device,
        )
        self.kv_connector = get_kv_connector(self.vllm_config, kv_caches_dict)

        # Attention groups are not supported.
        self.attn_groups = []  # type: ignore

        if self.model_config.enable_return_routed_experts:
            self.init_routed_experts_capturer()

    def init_routed_experts_capturer(self) -> None:
        logger.info("Initializing routed experts capturer for the V2 runner.")
        capturer = RoutedExpertsCapturer.create()
        (
            self._routed_experts_kv_group_idx,
            route_block_size,
        ) = select_routed_experts_kv_group(self.kv_cache_config.kv_cache_groups)
        max_num_kv_tokens = (
            self.kv_cache_config.num_blocks
            * self.parallel_config.data_parallel_size
            * len(self.kv_cache_config.kv_cache_groups)
            + 1
        ) * route_block_size
        capturer.init_buffer(
            max_num_batched_tokens=self.scheduler_config.max_num_batched_tokens,
            max_num_kv_tokens=max_num_kv_tokens,
            vllm_config=self.vllm_config,
        )
        self._bind_routed_experts_capturer(capturer)

    def _bind_routed_experts_capturer(
        self, capturer: RoutedExpertsCapturer
    ) -> None:
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE
        from vllm.model_executor.layers.fused_moe.router.base_router import (
            BaseRouter,
        )

        for module in self.compilation_config.static_forward_context.values():
            if isinstance(module, FusedMoE) and isinstance(module.router, BaseRouter):
                layer_id = module.layer_id

                def _capture_fn(topk_ids, _layer_id=layer_id, _capturer=capturer):
                    _capturer.capture(_layer_id, topk_ids)

                module.router.set_capture_fn(_capture_fn)

    def prepare_dummy_attn_metadata(self, input_batch: InputBatch) -> None:
        block_tables = self.block_tables.get_dummy_block_tables(input_batch.num_reqs)
        slot_mappings = self.block_tables.get_dummy_slot_mappings(
            input_batch.num_tokens
        )
        slot_mappings_by_layer = build_slot_mappings_by_layer(
            slot_mappings, self.kv_cache_config
        )
        attn_metadata = build_attn_metadata(
            attn_metadata_builders=self.attn_metadata_builders,
            num_reqs=input_batch.num_reqs,
            num_tokens=input_batch.num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=torch.from_numpy(input_batch.query_start_loc_np),
            seq_lens=input_batch.seq_lens,
            max_seq_len=self.max_model_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=self.kv_cache_config,
        )
        input_batch.attn_metadata = attn_metadata
        input_batch.slot_mappings = slot_mappings_by_layer

    @torch.inference_mode()
    def _dummy_run(
        self,
        num_tokens: int,
        *args: Any,
        skip_attn: bool = True,
        skip_eplb: bool = False,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Create a dummy scheduler output.
        num_reqs = min(num_tokens, self.max_num_reqs)
        num_tokens_per_request = [num_tokens // num_reqs] * num_reqs
        num_tokens_per_request[-1] += num_tokens % num_reqs
        assert sum(num_tokens_per_request) == num_tokens
        num_scheduled_tokens = {
            f"_dummy_req_{i}": n for i, n in enumerate(num_tokens_per_request)
        }
        dummy_scheduler_output = SchedulerOutput.make_empty()
        dummy_scheduler_output.total_num_scheduled_tokens = num_tokens
        dummy_scheduler_output.num_scheduled_tokens = num_scheduled_tokens

        # Disable any use of KVConnector for dummy runs.
        self.kv_connector.set_disabled(True)

        # Execute the model.
        self.execute_model(
            dummy_scheduler_output, dummy_run=True, skip_attn_for_dummy_run=skip_attn
        )
        self._maybe_run_tidar_dummy_draft()
        if not skip_eplb:
            self._dp_debug("dummy_eplb_begin tokens=%d", num_tokens)
            self.eplb_step(is_dummy=True, skip_if_all_dummy=True)
            self._dp_debug("dummy_eplb_end tokens=%d", num_tokens)
        self.kv_connector.set_disabled(False)
        assert self.execute_model_state is not None
        hidden_states, input_batch, _ = self.execute_model_state
        self.execute_model_state = None  # type: ignore
        sample_hidden_states = hidden_states[input_batch.logits_indices]
        return hidden_states, sample_hidden_states

    @torch.inference_mode()
    def _dummy_sampler_run(self, hidden_states: torch.Tensor) -> None:
        num_reqs = hidden_states.shape[0]
        logits = self.model.compute_logits(hidden_states)
        idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=self.device)
        idx_mapping_np = np.arange(num_reqs, dtype=np.int32)
        pos = torch.zeros(num_reqs, dtype=torch.int64, device=self.device)
        dummy_input_ids = torch.zeros(num_reqs, dtype=torch.int32, device=self.device)
        expanded_local_pos = torch.zeros(
            num_reqs, dtype=torch.int32, device=self.device
        )
        # NOTE(woosuk): During the initial memory profiling, the sampler may skip
        # top_k, top_p, and logprobs, using less GPU memory than what is possible
        # during actual execution.
        self.sampler(
            logits,
            idx_mapping,
            idx_mapping_np,
            idx_mapping_np,
            pos,
            dummy_input_ids,
            expanded_local_pos,
        )

    @torch.inference_mode()
    def profile_run(self) -> None:
        hidden_states, sample_hidden_states = self._dummy_run(
            self.max_num_tokens, skip_attn=True, skip_eplb=True
        )
        self._dummy_sampler_run(sample_hidden_states)
        if self.do_spec_decode:
            num_tokens_across_dp = make_num_tokens_across_dp(
                self.parallel_config.data_parallel_size, self.max_num_tokens
            )
            self.speculator.run_model(
                self.max_num_tokens,
                attn_metadata=None,
                slot_mappings=None,
                num_tokens_across_dp=num_tokens_across_dp,
            )
        self.eplb_step(is_dummy=True, is_profile=True)
        torch.cuda.synchronize()
        del hidden_states, sample_hidden_states
        gc.collect()

    def reset_mm_cache(self) -> None:
        if self.supports_mm_inputs:
            self.encoder_runner.reset_mm_cache()

    def reset_encoder_cache(self) -> None:
        if self.supports_mm_inputs:
            self.encoder_runner.reset_encoder_cache()

    def _get_num_input_tokens(self, num_scheduled_tokens: int) -> int:
        # SP is not supported yet.
        return num_scheduled_tokens

    @torch.inference_mode()
    def capture_model(self) -> int:
        if not self.cudagraph_manager.needs_capture():
            logger.warning(
                "Skipping CUDA graph capture. To turn on CUDA graph capture, "
                "ensure `cudagraph_mode` was not manually set to `NONE`"
            )
            return 0

        start_time = time.perf_counter()
        gc.collect()
        torch.cuda.empty_cache()
        start_free_gpu_memory = torch.cuda.mem_get_info()[0]

        with self.maybe_setup_dummy_loras(self.lora_config):
            mrope_positions = None
            if self.uses_mrope:
                mrope_positions = self.mrope_states.mrope_positions
            inputs_embeds = None
            if self.supports_mm_inputs:
                inputs_embeds = self.encoder_runner.inputs_embeds
            self.cudagraph_manager.capture(
                model=self.model,
                input_buffers=self.input_buffers,
                mrope_positions=mrope_positions,
                inputs_embeds=inputs_embeds,
                block_tables=self.block_tables,
                attn_metadata_builders=self.attn_metadata_builders,
                kv_cache_config=self.kv_cache_config,
            )
            if self.do_spec_decode:
                self.speculator.capture_model()
            # Capturing TiDAR graphs runs real forwards against live CCA
            # buffers; reset them before serving actual requests.
            self._reset_tidar_cca_runtime_state()
            self.eplb_step(is_dummy=True)

        end_time = time.perf_counter()
        end_free_gpu_memory = torch.cuda.mem_get_info()[0]
        elapsed_time = end_time - start_time
        cuda_graph_size = start_free_gpu_memory - end_free_gpu_memory
        # This usually takes 5~20 seconds.
        logger.info(
            "Graph capturing finished in %.0f secs, took %.2f GiB",
            elapsed_time,
            cuda_graph_size / (1 << 30),
        )
        return cuda_graph_size

    def warmup_for_prefill(self) -> None:
        # For FlashInfer, we would like to execute a dummy prefill run
        # to trigger JIT compilation.
        if all("FLASHINFER" in b.get_name() for b in self.attn_backends.values()):
            self._dummy_run(
                self.max_num_tokens, skip_attn=False, skip_eplb=True
            )
            self.eplb_step(is_dummy=True)
            torch.cuda.synchronize()

    def finish_requests(self, scheduler_output: SchedulerOutput) -> None:
        finished_req_ids = scheduler_output.finished_req_ids
        preempted_req_ids = scheduler_output.preempted_req_ids
        if preempted_req_ids:
            finished_req_ids = finished_req_ids.union(preempted_req_ids)
        for req_id in finished_req_ids:
            req_idx = self.req_states.req_id_to_index.get(req_id)
            if req_idx is not None and self._tidar_cca_state_slots is not None:
                self._tidar_cca_state_slots[req_idx] = PAD_SLOT_ID
            self.req_states.remove_request(req_id)
            if self.supports_mm_inputs:
                self.encoder_runner.remove_request(req_id)
            self.prompt_logprobs_worker.remove_request(req_id)
            self.lora_state.remove_request(req_id)

    def free_states(self, scheduler_output: SchedulerOutput) -> None:
        if self.supports_mm_inputs:
            for mm_hash in scheduler_output.free_encoder_mm_hashes:
                self.encoder_runner.free_encoder_cache(mm_hash)

    def add_requests(self, scheduler_output: SchedulerOutput) -> None:
        for new_req_data in scheduler_output.scheduled_new_reqs:
            assert new_req_data.prompt_token_ids is not None
            assert new_req_data.prefill_token_ids is not None
            assert new_req_data.sampling_params is not None
            req_id = new_req_data.req_id
            prompt_len = len(new_req_data.prompt_token_ids)
            self.req_states.add_request(
                req_id=req_id,
                prompt_len=prompt_len,
                prefill_token_ids=new_req_data.prefill_token_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
            )
            req_index = self.req_states.req_id_to_index[req_id]

            if self.supports_mm_inputs:
                self.encoder_runner.add_request(req_id, new_req_data.mm_features)

            # Pre-compute M-RoPE positions for prefill.
            if self.uses_mrope:
                self.mrope_states.init_prefill_mrope_positions(
                    req_index,
                    self.model,  # type: ignore
                    new_req_data.prefill_token_ids,
                    mm_features=new_req_data.mm_features,
                )

            self.block_tables.append_block_ids(
                req_index, new_req_data.block_ids, overwrite=True
            )
            self.sampler.add_request(
                req_index, prompt_len, new_req_data.sampling_params
            )
            self.prompt_logprobs_worker.add_request(
                req_id, req_index, new_req_data.sampling_params
            )
            self.lora_state.add_request(req_id, req_index, new_req_data.lora_request)

        if scheduler_output.scheduled_new_reqs:
            self.req_states.apply_staged_writes()
            self.sampler.apply_staged_writes(
                self.req_states.prefill_token_ids.gpu,
                self.req_states.prefill_len.np,
                self.req_states.prompt_len,
            )
            if self.uses_mrope:
                self.mrope_states.apply_staged_writes()

    def update_requests(self, scheduler_output: SchedulerOutput) -> None:
        # Add new blocks for the existing requests.
        reqs = scheduler_output.scheduled_cached_reqs
        for req_new_block_ids, req_id in zip(reqs.new_block_ids, reqs.req_ids):
            if req_new_block_ids is not None:
                req_index = self.req_states.req_id_to_index[req_id]
                self.block_tables.append_block_ids(
                    req_index, req_new_block_ids, overwrite=False
                )

    def prepare_inputs(
        self, scheduler_output: SchedulerOutput, num_tokens_after_padding: int
    ) -> InputBatch:
        num_tokens = scheduler_output.total_num_scheduled_tokens
        assert num_tokens > 0
        num_tokens_per_req = scheduler_output.num_scheduled_tokens
        num_reqs = len(num_tokens_per_req)

        # Decode first, then prefill.
        # batch_idx -> req_id
        req_ids = sorted(num_tokens_per_req, key=num_tokens_per_req.get)  # type: ignore[arg-type]
        numtoks_iter = map(num_tokens_per_req.get, req_ids)
        num_scheduled_tokens = np.fromiter(numtoks_iter, dtype=np.int32, count=num_reqs)

        idx_mapping_iter = map(self.req_states.req_id_to_index.get, req_ids)
        idx_mapping_np = np.fromiter(idx_mapping_iter, dtype=np.int32, count=num_reqs)
        idx_mapping = async_copy_to_gpu(idx_mapping_np, device=self.device)

        # Get the number of draft tokens for each request.
        draft_tokens = scheduler_output.scheduled_spec_decode_tokens
        if not draft_tokens:
            # No draft token scheduled (common case).
            total_num_draft_tokens = 0
            total_num_logits = num_reqs
            cu_num_logits_np = np.arange(num_reqs + 1, dtype=np.int32)
            cu_num_logits = torch.arange(
                num_reqs + 1, device=self.device, dtype=torch.int32
            )
            expanded_idx_mapping = idx_mapping
            expanded_local_pos = torch.zeros(
                num_reqs, dtype=torch.int32, device=self.device
            )
        else:
            num_draft_tokens = np.array(
                [len(draft_tokens.get(req_id, ())) for req_id in req_ids],
                dtype=np.int32,
            )
            total_num_draft_tokens = int(num_draft_tokens.sum())
            total_num_logits = num_reqs + total_num_draft_tokens

            num_logits = num_draft_tokens + 1
            cu_num_logits_np = np.empty(num_reqs + 1, dtype=np.int32)
            cu_num_logits_np[0] = 0
            np.cumsum(num_logits, out=cu_num_logits_np[1:])
            cu_num_logits = async_copy_to_gpu(cu_num_logits_np, device=self.device)

            max_expand_len = self.num_speculative_steps + 1
            expanded_idx_mapping, expanded_local_pos = expand_idx_mapping(
                idx_mapping, total_num_logits, cu_num_logits, max_expand_len
            )

        # Block tables: num_kv_cache_groups x [num_reqs, max_num_blocks]
        block_tables = self.block_tables.gather_block_tables(idx_mapping)
        self._apply_tidar_cca_state_slots(idx_mapping, block_tables, num_reqs)

        # Get query_start_loc.
        query_start_loc_np = np.empty(self.max_num_reqs + 1, dtype=np.int32)
        query_start_loc_np[0] = 0
        np.cumsum(num_scheduled_tokens, out=query_start_loc_np[1 : num_reqs + 1])
        # Pad for full CUDA graph mode.
        # Some attention backends like FA3 require query_start_loc to be non-decreasing.
        query_start_loc_np[num_reqs + 1 :] = num_tokens
        async_copy_to_gpu(query_start_loc_np, out=self.input_buffers.query_start_loc)

        query_start_loc_np = query_start_loc_np[: num_reqs + 1]
        query_start_loc_cpu = torch.from_numpy(query_start_loc_np)
        query_start_loc = self.input_buffers.query_start_loc[: num_reqs + 1]

        # Get prefill tokens.
        prepare_prefill_inputs(
            self.input_buffers.input_ids,
            self.req_states.next_prefill_tokens,
            idx_mapping,
            query_start_loc,
            self.req_states.prefill_token_ids.gpu,
            self.req_states.prefill_len.gpu,
            self.req_states.num_computed_tokens.gpu,
        )

        # Prepare positions and seq_lens.
        prepare_pos_seq_lens(
            idx_mapping,
            query_start_loc,
            self.req_states.num_computed_tokens.gpu,
            self.input_buffers.positions,
            self.input_buffers.seq_lens,
        )
        seq_lens = self.input_buffers.seq_lens[:num_reqs]

        # Prepare M-RoPE positions.
        if self.uses_mrope:
            self.mrope_states.prepare_mrope_positions(
                idx_mapping,
                query_start_loc,
                self.req_states.prefill_len.gpu,
                self.req_states.num_computed_tokens.gpu,
            )

        # Some input token ids are directly read from the last sampled tokens
        # and draft tokens. Also, get the logits indices to sample tokens from.
        logits_indices = combine_sampled_and_draft_tokens(
            self.input_buffers.input_ids,
            idx_mapping,
            self.req_states.last_sampled_tokens,
            query_start_loc,
            seq_lens,
            self.req_states.prefill_len.gpu,
            self.req_states.draft_tokens,
            cu_num_logits,
            total_num_logits,
        )

        # Compute slot mappings: [num_kv_cache_groups, num_tokens]
        slot_mappings = self.block_tables.compute_slot_mappings(
            idx_mapping,
            query_start_loc,
            self.input_buffers.positions[:num_tokens],
        )
        # Layer name -> slot mapping.
        slot_mappings_by_layer = build_slot_mappings_by_layer(
            slot_mappings, self.kv_cache_config
        )

        # Layer name -> attention metadata.
        attn_metadata = build_attn_metadata(
            attn_metadata_builders=self.attn_metadata_builders,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=self.input_buffers.seq_lens,
            max_seq_len=self.max_model_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=self.kv_cache_config,
        )

        input_ids = self.input_buffers.input_ids[:num_tokens_after_padding]
        positions = self.input_buffers.positions[:num_tokens_after_padding]
        mrope_positions = None
        if self.uses_mrope:
            mrope_positions = self.mrope_states.mrope_positions
            mrope_positions = mrope_positions[:, :num_tokens_after_padding]
        routed_expert_indices = None
        if self.model_config.enable_return_routed_experts:
            routed_expert_indices = slot_mappings[
                self._routed_experts_kv_group_idx, :num_tokens
            ]

        return InputBatch(
            req_ids=req_ids,
            num_reqs=num_reqs,
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping_np,
            expanded_idx_mapping=expanded_idx_mapping,
            expanded_local_pos=expanded_local_pos,
            num_scheduled_tokens=num_scheduled_tokens,
            num_tokens=num_tokens,
            num_tokens_after_padding=num_tokens_after_padding,
            num_draft_tokens=total_num_draft_tokens,
            query_start_loc=query_start_loc,
            query_start_loc_np=query_start_loc_np,
            seq_lens=seq_lens,
            input_ids=input_ids,
            positions=positions,
            mrope_positions=mrope_positions,
            inputs_embeds=None,
            attn_metadata=attn_metadata,
            slot_mappings=slot_mappings_by_layer,
            logits_indices=logits_indices,
            cu_num_logits=cu_num_logits,
            cu_num_logits_np=cu_num_logits_np,
            has_structured_output_reqs=scheduler_output.has_structured_output_requests,
            routed_expert_indices=routed_expert_indices,
        )

    @torch.inference_mode()
    def get_mm_embeddings(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
        input_batch: InputBatch,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        mm_hashes, mm_kwargs = self.encoder_runner.prepare_mm_inputs(
            scheduled_encoder_inputs
        )
        self.encoder_runner.execute_mm_encoder(self.model, mm_hashes, mm_kwargs)
        mm_embeds, is_mm_embed = self.encoder_runner.gather_mm_embeddings(
            input_batch.req_ids,
            input_batch.num_tokens,
            input_batch.num_scheduled_tokens,
            input_batch.query_start_loc_np,
            self.req_states.prefill_len.np[input_batch.idx_mapping_np],
            self.req_states.num_computed_prefill_tokens[input_batch.idx_mapping_np],
        )
        return mm_embeds, is_mm_embed

    def sample(
        self,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        grammar_output: GrammarOutput | None,
    ) -> tuple[SamplerOutput, torch.Tensor, torch.Tensor]:
        sample_hidden_states = hidden_states[input_batch.logits_indices]
        sample_pos = input_batch.positions[input_batch.logits_indices]
        input_ids = input_batch.input_ids[input_batch.logits_indices]
        logits = self.model.compute_logits(sample_hidden_states)
        if grammar_output is not None:
            # Apply grammar bitmask to the logits in-place.
            self.structured_outputs_worker.apply_grammar_bitmask(
                logits,
                input_batch,
                grammar_output.structured_output_request_ids,
                grammar_output.grammar_bitmask,
            )

        # Sample tokens and compute logprobs (if needed).
        sampler_output = self.sampler(
            logits,
            input_batch.expanded_idx_mapping,
            input_batch.idx_mapping_np,
            input_batch.cu_num_logits_np,
            sample_pos,
            input_ids,
            input_batch.expanded_local_pos,
        )

        if input_batch.num_draft_tokens == 0:
            # No draft tokens (common case).
            num_sampled = torch.ones(
                input_batch.num_reqs, dtype=torch.int32, device=self.device
            )
        else:
            # Rejection sampling for spec decoding.
            sampled_tokens, num_sampled = rejection_sample(
                sampler_output.sampled_token_ids,
                input_ids,
                input_batch.cu_num_logits,
                self.num_speculative_steps,
            )
            sampler_output.sampled_token_ids = sampled_tokens

        # Get the number of sampled and rejected tokens.
        # For chunked prefills, num_sampled and num_rejected are both 0.
        num_sampled, num_rejected = get_num_sampled_and_rejected(
            num_sampled,
            input_batch.seq_lens,
            input_batch.cu_num_logits,
            input_batch.idx_mapping,
            self.req_states.prefill_len.gpu,
        )
        return sampler_output, num_sampled, num_rejected

    def postprocess(
        self,
        input_batch: InputBatch,
        sampled_tokens: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
    ) -> None:
        # Update the number of computed tokens.
        post_update(
            input_batch.idx_mapping,
            self.req_states.num_computed_tokens.gpu,
            self.req_states.last_sampled_tokens,
            self.sampler.penalties_state.output_bin_counts,
            sampled_tokens,
            num_sampled,
            num_rejected,
            input_batch.query_start_loc,
        )

        # Update the number of computed prefill tokens.
        idx_mapping_np = input_batch.idx_mapping_np
        computed_prefill = self.req_states.num_computed_prefill_tokens
        computed_prefill[idx_mapping_np] += input_batch.num_scheduled_tokens
        np.minimum(
            computed_prefill, self.req_states.prefill_len.np, out=computed_prefill
        )

    def _use_tidar(self) -> bool:
        return (self.speculative_config is not None
                and self.speculative_config.use_tidar())

    def _tidar_step_draft_tokens(self, num_reqs: int, dummy_run: bool) -> int:
        if (dummy_run
                or not self._use_tidar()
                or self.parallel_config.data_parallel_size <= 1
                or self.parallel_config.is_moe_model is False
                or num_reqs <= 0):
            return 0
        return num_reqs * (self.num_speculative_steps + 1)

    def _store_tidar_draft_fold(
        self,
        real_across_dp: torch.Tensor | None,
    ) -> None:
        self._tidar_draft_real_across_dp = real_across_dp
        self._tidar_draft_coordinate_ran = False
        if real_across_dp is None or not self._use_tidar():
            self._tidar_draft_should_run = False
            self._tidar_draft_eff_across_dp = None
            return

        should_run = int(real_across_dp.max().item()) > 0
        self._tidar_draft_should_run = should_run
        if not should_run:
            self._tidar_draft_eff_across_dp = real_across_dp
            return

        query_len = self.num_speculative_steps + 1
        eff_across_dp = real_across_dp.clone()
        for rank in range(eff_across_dp.numel()):
            if int(eff_across_dp[rank].item()) == 0:
                eff_across_dp[rank] = query_len
        self._tidar_draft_eff_across_dp = eff_across_dp

    def _get_tidar_draft_num_tokens_across_dp(self) -> torch.Tensor | None:
        if (not self._use_tidar()
                or self.parallel_config.data_parallel_size <= 1
                or self.parallel_config.is_moe_model is False
                or not self._tidar_draft_should_run):
            return None
        return self._tidar_draft_eff_across_dp

    def _maybe_run_tidar_dummy_draft(self) -> None:
        if (not self._use_tidar()
                or self.parallel_config.data_parallel_size <= 1
                or self.parallel_config.is_moe_model is False
                or not self._tidar_draft_should_run
                or self._tidar_draft_coordinate_ran):
            return
        num_tokens_across_dp = self._tidar_draft_eff_across_dp
        if num_tokens_across_dp is None:
            return
        dp_rank = self.parallel_config.data_parallel_rank
        num_tokens = int(num_tokens_across_dp[dp_rank].item())
        if num_tokens <= 0:
            return
        assert self.speculator is not None
        dummy_run = getattr(self.speculator, "dummy_run", None)
        if dummy_run is None:
            return
        dummy_run(num_tokens, num_tokens_across_dp)
        self._tidar_draft_coordinate_ran = True

    def _get_tidar_cca_layers(self) -> dict[str, Any]:
        if self._tidar_cca_layers is None:
            from vllm.config import get_layers_from_vllm_config
            from vllm.model_executor.layers.mamba.cca import CCA

            self._tidar_cca_layers = get_layers_from_vllm_config(
                self.vllm_config, CCA)
        return self._tidar_cca_layers

    def _get_tidar_cca_group_ids(self) -> set[int]:
        if self._tidar_cca_group_ids is not None:
            return self._tidar_cca_group_ids

        cca_layer_names = set(self._get_tidar_cca_layers().keys())
        self._tidar_cca_group_ids = {
            i for i, group in enumerate(self.kv_cache_config.kv_cache_groups)
            if any(name in cca_layer_names for name in group.layer_names)
        }
        return self._tidar_cca_group_ids

    def _apply_tidar_cca_state_slots(
        self,
        idx_mapping: torch.Tensor,
        block_tables: tuple[torch.Tensor, ...],
        num_reqs: int,
    ) -> None:
        if not self._use_tidar():
            return
        cca_group_ids = self._get_tidar_cca_group_ids()
        if not cca_group_ids:
            return
        if self._tidar_cca_state_slots is None:
            self._tidar_cca_state_slots = torch.full(
                (self.max_num_reqs,),
                PAD_SLOT_ID,
                dtype=torch.int32,
                device=self.device,
            )

        req_indices = idx_mapping[:num_reqs].to(torch.long)
        for group_idx in cca_group_ids:
            block_table = block_tables[group_idx]
            current_slots = block_table[:num_reqs, 0]
            stable_slots = self._tidar_cca_state_slots[req_indices]
            unset = stable_slots == PAD_SLOT_ID
            # Async scheduling can compact request rows between steps. Keep
            # each request's recurrent CCA state on its first allocated slot.
            stable_slots = torch.where(unset, current_slots, stable_slots)
            self._tidar_cca_state_slots[req_indices] = stable_slots
            block_table[:num_reqs, 0].copy_(stable_slots)

    def _reset_tidar_cca_runtime_state(self) -> None:
        if not self._use_tidar():
            return

        for cca_layer in self._get_tidar_cca_layers().values():
            for kv_cache in getattr(cca_layer, "kv_cache", ()):
                if not isinstance(kv_cache, (list, tuple)):
                    continue
                for state_tensor in kv_cache:
                    if torch.is_tensor(state_tensor) and state_tensor.numel() > 0:
                        state_tensor.zero_()

            for attr in (
                    "_spec_stash_conv",
                    "_spec_stash_hs",
                    "_spec_stash_slots",
            ):
                buf = getattr(cca_layer, attr, None)
                if torch.is_tensor(buf) and buf.numel() > 0:
                    buf.zero_()
            cca_layer._spec_stash_eager_seq = -1
            cca_layer._spec_stash_eager_rows = []

        if self._tidar_cca_state_slots is not None:
            self._tidar_cca_state_slots.fill_(PAD_SLOT_ID)

    def _build_tidar_cca_commit_ctx(
        self,
        input_batch: InputBatch,
        scheduler_output: SchedulerOutput,
    ) -> None:
        self._tidar_cca_commit_ctx = None
        if not self._use_tidar():
            return

        cca_layers = self._get_tidar_cca_layers()
        if not cca_layers:
            return

        first_layer = next(iter(cca_layers.values()))
        spec_max_P = getattr(first_layer, "_spec_max_P", 0) or 0
        spec_max_S = getattr(first_layer, "_spec_max_S", 0) or 0
        stash_conv = getattr(first_layer, "_spec_stash_conv", None)
        if spec_max_P <= 0 or spec_max_S <= 0 or stash_conv is None:
            return

        spec_toks = scheduler_output.scheduled_spec_decode_tokens or {}
        if not spec_toks:
            return

        q_lens = [int(x) for x in input_batch.num_scheduled_tokens]
        num_reqs = len(q_lens)

        num_decodes = 0
        while num_decodes < num_reqs and q_lens[num_decodes] <= 1:
            num_decodes += 1
        if any(q_lens[i] <= 1 for i in range(num_decodes, num_reqs)):
            logger.warning_once(
                "TiDAR V2 CCA commit: batch is not decode-first ordered; "
                "skipping spec-state commit for this step.")
            return

        prefill_lens = q_lens[num_decodes:]
        if not prefill_lens:
            return

        uniform = (
            all(length == spec_max_S for length in prefill_lens)
            and len(prefill_lens) <= spec_max_P
        )
        if uniform:
            seg_rows = list(range(len(prefill_lens)))
        else:
            from vllm.model_executor.layers.mamba.cca import CCA

            if (getattr(first_layer, "_spec_stash_eager_seq", -1)
                    != CCA._tidar_step_seq):
                return
            seg_rows = list(
                getattr(first_layer, "_spec_stash_eager_rows", []) or [])
            if not seg_rows:
                return

        stash_rows: list[int] = []
        batch_rows: list[int] = []
        for j, seg_row in enumerate(seg_rows):
            batch_row = num_decodes + seg_row
            if (batch_row < num_reqs
                    and len(spec_toks.get(input_batch.req_ids[batch_row], ())) > 0):
                stash_rows.append(j)
                batch_rows.append(batch_row)
        if not stash_rows:
            return

        device = stash_conv.device
        self._tidar_cca_commit_ctx = {
            "stash_rows_gpu": torch.as_tensor(
                stash_rows, dtype=torch.long, device=device),
            "batch_rows_gpu": torch.as_tensor(
                batch_rows, dtype=torch.long, device=device),
            "batch_rows": batch_rows,
            "k_max": spec_max_S - 1,
            "device": device,
        }

    def _commit_tidar_cca_layers(self, idx_gpu: torch.Tensor) -> None:
        ctx = self._tidar_cca_commit_ctx
        if ctx is None:
            return
        n_rows = int(ctx["stash_rows_gpu"].shape[0])
        dummy_counts = [0] * n_rows
        for cca_layer in self._get_tidar_cca_layers().values():
            cca_layer.commit_spec_decode_state(
                dummy_counts,
                idx_gpu=idx_gpu,
                arange_gpu=ctx["stash_rows_gpu"],
            )

    def _commit_tidar_cca_state_from_num_sampled(
        self,
        num_sampled: torch.Tensor,
    ) -> None:
        ctx = getattr(self, "_tidar_cca_commit_ctx", None)
        if ctx is None:
            return
        batch_rows_gpu = ctx["batch_rows_gpu"]
        if int(num_sampled.shape[0]) <= int(ctx["batch_rows"][-1]):
            return
        counts = num_sampled.to(
            device=ctx["device"], dtype=torch.long,
            non_blocking=True)[batch_rows_gpu]
        idx_gpu = (counts - 1).clamp_(min=0, max=ctx["k_max"])
        self._commit_tidar_cca_layers(idx_gpu)

    @torch.inference_mode()
    def propose_draft(
        self,
        input_batch: InputBatch,
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
    ) -> torch.Tensor:
        assert self.speculator is not None
        draft_num_tokens_across_dp = self._get_tidar_draft_num_tokens_across_dp()
        draft_tokens = self.speculator.propose(
            input_batch,
            last_hidden_states,
            aux_hidden_states,
            num_sampled,
            num_rejected,
            self.req_states.last_sampled_tokens,
            self.req_states.next_prefill_tokens,
            self.sampler.sampling_states.temperature.gpu,
            self.sampler.sampling_states.seeds.gpu,
            draft_num_tokens_across_dp,
        )
        if draft_num_tokens_across_dp is not None:
            self._tidar_draft_coordinate_ran = True
        return draft_tokens

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        intermediate_tensors: Any | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
    ) -> ModelRunnerOutput | None:
        assert intermediate_tensors is None
        if self.model_config.enable_return_routed_experts:
            capturer = RoutedExpertsCapturer.get_instance()
            if capturer is not None:
                capturer.clear_buffer()
            elif not dummy_run:
                raise RuntimeError("RoutedExpertsCapturer is not initialized.")

        if not dummy_run:
            # Update the request states.
            self.finish_requests(scheduler_output)
            self.free_states(scheduler_output)
            self.add_requests(scheduler_output)
            self.update_requests(scheduler_output)
            self.block_tables.apply_staged_writes()

        # Get the CUDA graph size. None means no CUDA graph is used.
        cudagraph_size = self.cudagraph_manager.get_cudagraph_size(
            scheduler_output.total_num_scheduled_tokens,
            scheduler_output.num_scheduled_tokens.values(),
        )
        tidar_draft_tokens = self._tidar_step_draft_tokens(
            len(scheduler_output.num_scheduled_tokens),
            dummy_run,
        )
        (use_cudagraph, num_tokens_after_padding, num_tokens_across_dp,
         tidar_draft_across_dp) = (
            get_cudagraph_and_dp_padding(
                scheduler_output.total_num_scheduled_tokens,
                cudagraph_size,
                self.parallel_config.data_parallel_size,
                self.parallel_config.data_parallel_rank,
                tidar_draft_tokens=tidar_draft_tokens,
            )
        )
        self._store_tidar_draft_fold(tidar_draft_across_dp)
        self._dp_debug(
            "execute metadata total=%d padded=%d dummy_arg=%s use_cg=%s "
            "tokens_across=%s draft_across=%s",
            scheduler_output.total_num_scheduled_tokens,
            num_tokens_after_padding,
            dummy_run,
            use_cudagraph,
            None if num_tokens_across_dp is None else num_tokens_across_dp.tolist(),
            None if tidar_draft_across_dp is None
            else tidar_draft_across_dp.tolist(),
        )
        if num_tokens_after_padding == 0:
            # All DP ranks have zero tokens to run.
            empty_output = self.kv_connector.no_forward(scheduler_output)
            return empty_output

        runtime_dummy_run = (
            not dummy_run and scheduler_output.total_num_scheduled_tokens == 0
        )
        model_dummy_run = dummy_run or runtime_dummy_run

        if not model_dummy_run:
            # Common case.
            # Prepare all the inputs and copy to the input buffers.
            input_batch = self.prepare_inputs(
                scheduler_output, num_tokens_after_padding
            )
            if self.lora_config:
                # Activate LoRA adapters.
                lora_inputs = self.lora_state.make_lora_inputs(
                    input_batch.req_ids,
                    input_batch.idx_mapping_np,
                    input_batch.num_scheduled_tokens,
                )
                self._set_active_loras(*lora_inputs)

            if self.supports_mm_inputs:
                # Execute the multimodal encoder.
                mm_embeds, is_mm_embed = self.get_mm_embeddings(
                    scheduler_output.scheduled_encoder_inputs, input_batch
                )
                inputs_embeds = self.encoder_runner.get_inputs_embeds(
                    self.model, input_batch.input_ids, mm_embeds, is_mm_embed
                )
                input_batch.inputs_embeds = inputs_embeds[
                    : input_batch.num_tokens_after_padding
                ]
        else:
            # No actual tokens to run. A dummy run for DP or memory profiling.
            num_reqs = min(num_tokens_after_padding, self.max_num_reqs)
            input_batch = InputBatch.make_dummy(
                num_reqs=num_reqs,
                num_tokens=num_tokens_after_padding,
                input_buffers=self.input_buffers,
                device=self.device,
            )
            if self.uses_mrope:
                input_batch.mrope_positions = self.mrope_states.mrope_positions[
                    :, :num_tokens_after_padding
                ]
            if not skip_attn_for_dummy_run and not runtime_dummy_run:
                self.prepare_dummy_attn_metadata(input_batch)
            # FIXME(woosuk): Fix warmup for LoRA.

        if not model_dummy_run:
            if self._use_tidar():
                from vllm.model_executor.layers.mamba.cca import CCA
                CCA._tidar_step_seq += 1

        # Run model.
        if use_cudagraph:
            # Run CUDA graph.
            # NOTE(woosuk): Here, we don't need to pass the input tensors,
            # because they are already copied to the CUDA graph input buffers.
            self._dp_debug("main_forward_begin runtime_dummy=%s tokens=%d cg=%s",
                           runtime_dummy_run,
                           input_batch.num_tokens_after_padding,
                           use_cudagraph)
            self.kv_connector.pre_forward(scheduler_output)
            hidden_states = self.cudagraph_manager.run(
                input_batch.num_tokens_after_padding
            )
            self._dp_debug("main_forward_end runtime_dummy=%s",
                           runtime_dummy_run)
        else:
            # Run PyTorch model in eager mode.
            positions = input_batch.positions
            if self.uses_mrope:
                assert input_batch.mrope_positions is not None
                positions = input_batch.mrope_positions
            with set_forward_context(
                input_batch.attn_metadata,
                self.vllm_config,
                num_tokens=input_batch.num_tokens_after_padding,
                # TODO(woosuk): Support piecewise CUDA graph.
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                num_tokens_across_dp=num_tokens_across_dp,
                slot_mapping=input_batch.slot_mappings,
            ):
                self._dp_debug(
                    "main_forward_begin runtime_dummy=%s tokens=%d cg=%s",
                    runtime_dummy_run,
                    input_batch.num_tokens_after_padding,
                    use_cudagraph,
                )
                self.kv_connector.pre_forward(scheduler_output)
                hidden_states = self.model(
                    input_ids=input_batch.input_ids,
                    positions=positions,
                    inputs_embeds=input_batch.inputs_embeds,
                )
                self._dp_debug("main_forward_end runtime_dummy=%s",
                               runtime_dummy_run)

        if not model_dummy_run:
            self._build_tidar_cca_commit_ctx(input_batch, scheduler_output)

        if self.model_config.enable_return_routed_experts and not model_dummy_run:
            capturer = RoutedExpertsCapturer.get_instance()
            assert capturer is not None
            assert input_batch.routed_expert_indices is not None
            capturer.save_captured_experts(
                input_batch.routed_expert_indices.cpu().numpy()
            )

        kv_connector_output = self.kv_connector.post_forward(scheduler_output)
        if runtime_dummy_run:
            self._dp_debug("runtime_dummy_draft_begin")
            self._maybe_run_tidar_dummy_draft()
            self._dp_debug("runtime_dummy_draft_end")
            self._dp_debug("runtime_dummy_eplb_begin")
            self.eplb_step(is_dummy=True, skip_if_all_dummy=True)
            self._dp_debug("runtime_dummy_eplb_end")
            return ModelRunnerOutput(
                req_ids=[],
                req_id_to_index={},
                kv_connector_output=kv_connector_output,
            )

        self.execute_model_state = hidden_states, input_batch, kv_connector_output
        return None

    @torch.inference_mode()
    def sample_tokens(
        self, grammar_output: GrammarOutput | None
    ) -> AsyncOutput | ModelRunnerOutput:
        assert self.execute_model_state is not None
        hidden_states, input_batch, kv_connector_output = self.execute_model_state
        self.execute_model_state = None  # type: ignore
        self._dp_debug("sample_begin reqs=%d tokens=%d", input_batch.num_reqs,
                       input_batch.num_tokens)

        sampler_output, num_sampled, num_rejected = self.sample(
            hidden_states, input_batch, grammar_output
        )
        prompt_logprobs_dict = self.prompt_logprobs_worker.compute_prompt_logprobs(
            self.model.compute_logits,
            hidden_states,
            input_batch,
            self.req_states.prefill_token_ids.gpu,
            self.req_states.num_computed_tokens.gpu,
            self.req_states.prompt_len,
            self.req_states.prefill_len.np,
            self.req_states.num_computed_prefill_tokens,
        )

        # Prepare the model runner output.
        model_runner_output = ModelRunnerOutput(
            req_ids=input_batch.req_ids,
            # NOTE(woosuk): req_id_to_index is unused in this model runner.
            # Only for compatibility with the existing model runner and scheduler.
            req_id_to_index={req_id: i for i, req_id in enumerate(input_batch.req_ids)},
            sampled_token_ids=None,  # type: ignore
            prompt_logprobs_dict=prompt_logprobs_dict,  # type: ignore[arg-type]
            kv_connector_output=kv_connector_output,
        )
        async_output = AsyncOutput(
            model_runner_output=model_runner_output,
            sampler_output=sampler_output,
            num_sampled_tokens=num_sampled,
            copy_stream=self.output_copy_stream,
            copy_event=self.output_copy_event,
        )

        # Postprocess results and update request states.
        # NOTE: This is intentionally done after creating the AsyncOutput,
        # ensuring that `copy_event` is recorded before calling postprocess.
        # This sequencing may slightly reduce latency as async D2H copy does not
        # need to wait for the postprocess to finish.
        self.postprocess(
            input_batch, sampler_output.sampled_token_ids, num_sampled, num_rejected
        )
        if self.do_spec_decode:
            if self._use_tidar():
                self._commit_tidar_cca_state_from_num_sampled(num_sampled)
            self._dp_debug("draft_begin reqs=%d", input_batch.num_reqs)
            capture_guard = nullcontext()
            if (
                self.model_config.enable_return_routed_experts
                and self._use_tidar()
            ):
                capturer = RoutedExpertsCapturer.get_instance()
                if capturer is not None:
                    capture_guard = capturer.capture_disabled()
            with capture_guard:
                draft_tokens = self.propose_draft(
                    input_batch,
                    hidden_states,
                    None,  # aux_hidden_states
                    num_sampled,
                    num_rejected,
                )
            self._dp_debug("draft_end reqs=%d", input_batch.num_reqs)
            self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens
            self.draft_tokens_handler.set_draft_tokens(
                input_batch,
                draft_tokens,
                copy_all=not self.use_async_scheduling,
            )
        self._dp_debug("sample_eplb_begin")
        self.eplb_step(skip_if_all_dummy=True)
        self._dp_debug("sample_eplb_end")

        if self.use_async_scheduling:
            return async_output
        return async_output.get_output()

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        return self.draft_tokens_handler.get_draft_tokens()
