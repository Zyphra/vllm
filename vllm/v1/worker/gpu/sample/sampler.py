# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import os

import numpy as np
import torch

import vllm.envs as envs
from vllm.config.model import LogprobsMode
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.worker.gpu.metrics.logits import get_num_nans
from vllm.v1.worker.gpu.sample.gumbel import (
    apply_temperature,
    gumbel_sample,
    gumbel_sample_with_probs,
    tidar_sample_bonus_tokens,
    tidar_target_stats,
)
from vllm.v1.worker.gpu.sample.logit_bias import LogitBiasState
from vllm.v1.worker.gpu.sample.logprob import compute_topk_logprobs
from vllm.v1.worker.gpu.sample.min_p import apply_min_p
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.penalties import PenaltiesState
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS, SamplingStates


logger = init_logger(__name__)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


class Sampler:
    def __init__(
        self,
        max_num_reqs: int,
        vocab_size: int,
        device: torch.device,
        logprobs_mode: LogprobsMode = "raw_logprobs",
        num_speculative_tokens: int = 1,
    ):
        if logprobs_mode not in ("processed_logprobs", "raw_logprobs"):
            raise NotImplementedError(f"Unsupported logprobs_mode: {logprobs_mode}")
        self.logprobs_mode = logprobs_mode
        # VERL compares rollout scores against Megatron scores computed from
        # temperature-scaled AR verifier logits. The legacy model runner honors
        # this flag in RejectionSampler. The V2 runner has a separate sampler,
        # so it must opt into the same score space explicitly.
        self.return_ar_logprobs = _env_flag("VLLM_TIDAR_RETURN_AR_LOGPROBS")
        self.assert_ar_logprobs = _env_flag("VLLM_TIDAR_ASSERT_AR_LOGPROBS")
        if self.assert_ar_logprobs and not self.return_ar_logprobs:
            raise ValueError(
                "VLLM_TIDAR_ASSERT_AR_LOGPROBS=1 requires "
                "VLLM_TIDAR_RETURN_AR_LOGPROBS=1"
            )

        self.ar_logprob_temperature: float | None = None
        configured_ar_temperature = os.environ.get(
            "VLLM_TIDAR_AR_TEMPERATURE", ""
        ).strip()
        if configured_ar_temperature:
            try:
                self.ar_logprob_temperature = float(configured_ar_temperature)
            except ValueError as exc:
                if self.assert_ar_logprobs:
                    raise ValueError(
                        "VLLM_TIDAR_AR_TEMPERATURE must be a finite positive float "
                        "when VLLM_TIDAR_ASSERT_AR_LOGPROBS=1"
                    ) from exc
        if self.assert_ar_logprobs and (
            self.ar_logprob_temperature is None
            or not math.isfinite(self.ar_logprob_temperature)
            or self.ar_logprob_temperature <= 0
        ):
            raise ValueError(
                "VLLM_TIDAR_AR_TEMPERATURE must be a finite positive float "
                "when VLLM_TIDAR_ASSERT_AR_LOGPROBS=1"
            )
        if self.return_ar_logprobs:
            logger.info(
                "V2 AR logprobs enabled: scores use request-temperature-scaled "
                "verifier logits before min_p/top_k/top_p; strict=%s, "
                "configured_ar_temperature=%s",
                self.assert_ar_logprobs,
                self.ar_logprob_temperature,
            )
        self.compute_nans = envs.VLLM_COMPUTE_NANS_IN_LOGITS  # False by default.

        self.sampling_states = SamplingStates(max_num_reqs, vocab_size)
        self.penalties_state = PenaltiesState(max_num_reqs, vocab_size, device)
        self.logit_bias_state = LogitBiasState(max_num_reqs, device)
        self.num_speculative_tokens = num_speculative_tokens

    def add_request(
        self, req_idx: int, prompt_len: int, sampling_params: SamplingParams
    ) -> None:
        if self.assert_ar_logprobs:
            request_temperature = float(sampling_params.temperature)
            if self.ar_logprob_temperature is None:
                raise RuntimeError(
                    "Strict V2 AR logprobs enabled without a configured "
                    "VLLM_TIDAR_AR_TEMPERATURE"
                )
            if (
                not math.isfinite(request_temperature)
                or request_temperature <= 0
                or not math.isclose(
                    request_temperature,
                    self.ar_logprob_temperature,
                    rel_tol=1e-6,
                    abs_tol=1e-8,
                )
            ):
                raise ValueError(
                    "V2 AR logprob temperature mismatch: SamplingParams.temperature="
                    f"{request_temperature!r}, VLLM_TIDAR_AR_TEMPERATURE="
                    f"{self.ar_logprob_temperature!r}"
                )
        self.sampling_states.add_request(req_idx, sampling_params)
        self.penalties_state.add_request(req_idx, sampling_params)
        self.logit_bias_state.add_request(req_idx, prompt_len, sampling_params)

    def apply_staged_writes(
        self,
        prefill_token_ids: torch.Tensor,
        prefill_lens: np.ndarray,
        prompt_lens: np.ndarray,
    ) -> None:
        self.sampling_states.apply_staged_writes()
        self.penalties_state.apply_staged_writes(
            prefill_token_ids, prefill_lens, prompt_lens
        )
        self.logit_bias_state.apply_staged_writes()

    def __call__(
        self,
        logits: torch.Tensor,
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        cu_num_logits_np: np.ndarray,
        pos: torch.Tensor,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        prob_token_ids: torch.Tensor | None = None,
        tidar_cu_num_logits: torch.Tensor | None = None,
        defer_logprobs: bool = False,
    ) -> SamplerOutput:
        # NOTE(woosuk): We intentionally compute num_nans before sampling to make clear
        # that num_nans is computed before applying penalties and temperature.
        num_nans = get_num_nans(logits) if self.compute_nans else None
        max_num_logprobs = self.sampling_states.max_num_logprobs(idx_mapping_np)
        (
            sampled,
            processed_logits,
            ar_logprob_logits,
            prob_token_probs,
            logsumexp,
            logprob_logsumexp,
            logprob_top1_token_ids,
        ) = self.sample(
            logits,
            idx_mapping,
            idx_mapping_np,
            pos,
            input_ids,
            expanded_local_pos,
            prob_token_ids,
            tidar_cu_num_logits,
            collect_tidar_logprob_stats=(
                defer_logprobs and max_num_logprobs == 1
            ),
        )

        if max_num_logprobs != NO_LOGPROBS and not defer_logprobs:
            sampled_before_logprobs = (
                sampled.clone() if self.assert_ar_logprobs else None
            )
            if self.return_ar_logprobs:
                if ar_logprob_logits is None:
                    raise RuntimeError(
                        "V2 AR logprobs were requested but the temperature-scaled "
                        "verifier-logit snapshot was not produced"
                    )
                logprob_logits = ar_logprob_logits
            elif self.logprobs_mode == "processed_logprobs":
                logprob_logits = processed_logits
            else:
                logprob_logits = logits
            expanded_logits = logprob_logits.shape[0] != idx_mapping_np.shape[0]
            cu_num_logits = cu_num_logits_np.tolist() if expanded_logits else None
            logprobs_tensors = compute_topk_logprobs(
                logprob_logits, max_num_logprobs, sampled, cu_num_logits
            )
            if sampled_before_logprobs is not None:
                if not torch.equal(sampled, sampled_before_logprobs):
                    raise RuntimeError(
                        "AR logprob collection changed sampled token IDs"
                    )
        else:
            logprobs_tensors = None

        # These are GPU tensors.
        sampler_output = SamplerOutput(
            # The sampled tokens are expanded to 2D tensor with shape
            # [num_requests, 1], where each row represents one generated
            # token per request.
            sampled_token_ids=sampled.view(-1, 1),
            logprobs_tensors=logprobs_tensors,
            num_nans=num_nans,
            prob_token_probs=prob_token_probs,
            logsumexp=logsumexp,
            processed_logits=(
                processed_logits if prob_token_ids is not None else None
            ),
            ar_logprob_logits=(
                ar_logprob_logits if prob_token_ids is not None else None
            ),
            logprob_logsumexp=logprob_logsumexp,
            logprob_top1_token_ids=logprob_top1_token_ids,
        )
        return sampler_output

    def sample(
        self,
        logits: torch.Tensor,
        idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        prob_token_ids: torch.Tensor | None = None,
        tidar_cu_num_logits: torch.Tensor | None = None,
        collect_tidar_logprob_stats: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        raw_logits = logits
        needs_ar_logprobs = (
            self.return_ar_logprobs
            and self.sampling_states.max_num_logprobs(idx_mapping_np) != NO_LOGPROBS
        )
        simple_sampling = (
            os.environ.get("VLLM_DISABLE_SIMPLE_SAMPLER_FAST_PATH", "0") != "1"
            and not np.any(self.logit_bias_state.use_logit_bias[idx_mapping_np])
            and not np.any(self.penalties_state.use_penalty[idx_mapping_np])
            and not self.sampling_states.do_min_p(idx_mapping_np)
            and not self.sampling_states.do_top_k(idx_mapping_np)
            and not self.sampling_states.do_top_p(idx_mapping_np)
            and self.sampling_states.max_num_logprobs(idx_mapping_np)
            == NO_LOGPROBS
            and prob_token_ids is None
        )
        if simple_sampling:
            use_fp32_gumbel = os.environ.get(
                "VLLM_SIMPLE_SAMPLER_FP32_GUMBEL", "0") == "1"
            if use_fp32_gumbel:
                sampled = gumbel_sample(
                    logits,
                    idx_mapping,
                    self.sampling_states.temperature.gpu,
                    self.sampling_states.seeds.gpu,
                    pos,
                    apply_temperature=True,
                    use_fp64_noise=False,
                )
                return sampled, logits, None, None, None, None, None

            if logits.dtype != torch.float32:
                logits = torch.empty_like(logits, dtype=torch.float32).copy_(logits)
            apply_temperature(
                logits, idx_mapping, self.sampling_states.temperature.gpu
            )
            sampled = gumbel_sample(
                logits,
                idx_mapping,
                self.sampling_states.temperature.gpu,
                self.sampling_states.seeds.gpu,
                pos,
                apply_temperature=False,
            )
            return sampled, logits, None, None, None, None, None

        # Copy logits to a new FP32 tensor.
        logits = torch.empty_like(logits, dtype=torch.float32).copy_(logits)

        # Apply logit bias (e.g., allowed_token_ids, min_tokens) in place.
        self.logit_bias_state.apply_logit_bias(logits, idx_mapping, idx_mapping_np, pos)

        # Apply penalties in place.
        self.penalties_state.apply_penalties(
            logits,
            idx_mapping,
            idx_mapping_np,
            input_ids,
            expanded_local_pos,
            self.num_speculative_tokens,
        )

        # Apply temperature in place.
        apply_temperature(logits, idx_mapping, self.sampling_states.temperature.gpu)

        do_min_p = self.sampling_states.do_min_p(idx_mapping_np)
        do_top_k = self.sampling_states.do_top_k(idx_mapping_np)
        do_top_p = self.sampling_states.do_top_p(idx_mapping_np)

        # Preserve the AR verifier distribution after processors, penalties,
        # and temperature, but before all sampling-time filters. Min-p and top-k
        # may mutate their input in place, so clone only when a filter is active.
        # The common unfiltered rollout path incurs no extra allocation.
        ar_logprob_logits = None
        if needs_ar_logprobs:
            ar_logprob_logits = (
                logits.clone() if do_min_p or do_top_k or do_top_p else logits
            )

        # Apply min_p in place if any request has a non-zero min_p.
        if do_min_p:
            apply_min_p(logits, idx_mapping, self.sampling_states.min_p.gpu)

        # Apply top_k and/or top_p. This might return a new tensor.
        top_k = self.sampling_states.top_k.gpu[idx_mapping] if do_top_k else None
        top_p = self.sampling_states.top_p.gpu[idx_mapping] if do_top_p else None
        if do_top_k or do_top_p:
            logits = apply_top_k_top_p(logits, top_k, top_p)

        # Sample the next token.
        if prob_token_ids is None:
            sampled = gumbel_sample(
                logits,
                idx_mapping,
                self.sampling_states.temperature.gpu,
                self.sampling_states.seeds.gpu,
                pos,
                apply_temperature=False,
            )
            prob_token_probs = None
            logsumexp = None
            logprob_logsumexp = None
            logprob_top1_token_ids = None
        elif tidar_cu_num_logits is not None:
            logprob_logits = None
            if collect_tidar_logprob_stats:
                if self.return_ar_logprobs:
                    if ar_logprob_logits is None:
                        raise RuntimeError(
                            "V2 AR logprobs were requested but the "
                            "temperature-scaled verifier-logit snapshot was "
                            "not produced"
                        )
                    logprob_logits = ar_logprob_logits
                elif self.logprobs_mode == "raw_logprobs":
                    logprob_logits = raw_logits
            (
                prob_token_probs,
                logsumexp,
                logprob_logsumexp,
                logprob_top1_token_ids,
            ) = tidar_target_stats(
                logits,
                prob_token_ids,
                logprob_logits=logprob_logits,
            )
            sampled = tidar_sample_bonus_tokens(
                logits,
                tidar_cu_num_logits,
                idx_mapping,
                self.sampling_states.seeds.gpu,
                pos,
            )
        else:
            sampled, prob_token_probs, logsumexp = gumbel_sample_with_probs(
                logits,
                idx_mapping,
                self.sampling_states.temperature.gpu,
                self.sampling_states.seeds.gpu,
                pos,
                apply_temperature=False,
                prob_token_ids=prob_token_ids,
            )
            logprob_logsumexp = None
            logprob_top1_token_ids = None
        return (
            sampled,
            logits,
            ar_logprob_logits,
            prob_token_probs,
            logsumexp,
            logprob_logsumexp,
            logprob_top1_token_ids,
        )
