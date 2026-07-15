# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

import vllm.v1.worker.gpu.sample.sampler as sampler_module
from vllm.v1.worker.gpu.sample.sampler import Sampler


class _NoOpState:
    def __init__(self) -> None:
        self.use_logit_bias = np.array([False])
        self.use_penalty = np.array([False])

    def apply_logit_bias(self, *args) -> None:
        pass

    def apply_penalties(self, *args) -> None:
        pass


class _SamplingStates:
    def __init__(self, *, min_p: bool = False, top_k: bool = False) -> None:
        self.temperature = SimpleNamespace(gpu=torch.tensor([0.6]))
        self.seeds = SimpleNamespace(gpu=torch.tensor([123], dtype=torch.int64))
        self.min_p = SimpleNamespace(gpu=torch.tensor([0.0]))
        self.top_k = SimpleNamespace(gpu=torch.tensor([2], dtype=torch.int32))
        self.top_p = SimpleNamespace(gpu=torch.tensor([1.0]))
        self._min_p = min_p
        self._top_k = top_k

    def max_num_logprobs(self, _idx_mapping_np: np.ndarray) -> int:
        return 0

    def do_min_p(self, _idx_mapping_np: np.ndarray) -> bool:
        return self._min_p

    def do_top_k(self, _idx_mapping_np: np.ndarray) -> bool:
        return self._top_k

    def do_top_p(self, _idx_mapping_np: np.ndarray) -> bool:
        return False


def _bare_sampler(
    *, return_ar_logprobs: bool, min_p: bool = False, top_k: bool = False
) -> Sampler:
    sampler = Sampler.__new__(Sampler)
    sampler.logprobs_mode = "raw_logprobs"
    sampler.return_ar_logprobs = return_ar_logprobs
    sampler.assert_ar_logprobs = False
    sampler.ar_logprob_temperature = 0.6
    sampler.compute_nans = False
    sampler.sampling_states = _SamplingStates(min_p=min_p, top_k=top_k)
    sampler.logit_bias_state = _NoOpState()
    sampler.penalties_state = _NoOpState()
    sampler.num_speculative_tokens = 17
    return sampler


def test_ar_logprob_snapshot_is_temperature_scaled_and_pre_filters(monkeypatch) -> None:
    sampler = _bare_sampler(return_ar_logprobs=True, min_p=True, top_k=True)
    raw_logits = torch.tensor([[2.0, 1.0, -1.0]])

    def fake_apply_temperature(logits, _idx_mapping, _temperatures) -> None:
        logits.div_(0.6)

    def fake_apply_top_k_top_p(logits, _top_k, _top_p):
        # Mutate in place, matching the V2 top-k-only implementation.
        logits[:, 1] = -torch.inf
        return logits

    def fake_apply_min_p(logits, _idx_mapping, _min_p) -> None:
        # Min-p is also an in-place filter and must not affect AR scores.
        logits[:, 2] = -torch.inf

    sampled = torch.tensor([1], dtype=torch.int64)

    monkeypatch.setattr(sampler_module, "apply_temperature", fake_apply_temperature)
    monkeypatch.setattr(sampler_module, "apply_min_p", fake_apply_min_p)
    monkeypatch.setattr(sampler_module, "apply_top_k_top_p", fake_apply_top_k_top_p)
    monkeypatch.setattr(
        sampler_module,
        "gumbel_sample",
        lambda *args, **kwargs: sampled.clone(),
    )

    (
        actual_sampled,
        filtered_logits,
        ar_logprob_logits,
        *_unused,
    ) = sampler.sample(
        raw_logits,
        torch.tensor([0]),
        np.array([0]),
        torch.tensor([0]),
        torch.tensor([0]),
        torch.tensor([0]),
    )

    assert torch.equal(actual_sampled, sampled)
    assert torch.equal(raw_logits, torch.tensor([[2.0, 1.0, -1.0]]))
    assert ar_logprob_logits is not None
    torch.testing.assert_close(ar_logprob_logits, raw_logits / 0.6)
    assert torch.isneginf(filtered_logits[0, 1])
    assert torch.isneginf(filtered_logits[0, 2])
    assert torch.isfinite(ar_logprob_logits[0, 1])
    assert torch.isfinite(ar_logprob_logits[0, 2])


def test_ar_logprob_override_changes_scores_not_sampled_tokens(monkeypatch) -> None:
    raw_logits = torch.tensor([[2.0, 1.0, -1.0]])
    ar_logits = raw_logits / 0.6
    sampled = torch.tensor([1], dtype=torch.int64)
    captured_logprobs: list[torch.Tensor] = []

    def fake_compute_topk_logprobs(logits, _num_logprobs, token_ids, _cu_num_logits):
        selected = torch.log_softmax(logits, dim=-1).gather(1, token_ids.view(-1, 1))
        captured_logprobs.append(selected)
        return selected

    monkeypatch.setattr(
        sampler_module, "compute_topk_logprobs", fake_compute_topk_logprobs
    )

    outputs = []
    for return_ar_logprobs in (False, True):
        sampler = _bare_sampler(return_ar_logprobs=return_ar_logprobs)

        def fake_sample(self, *args, **kwargs):
            return (
                sampled.clone(),
                ar_logits.clone(),
                ar_logits.clone(),
                None,
                None,
                None,
                None,
            )

        sampler.sample = MethodType(fake_sample, sampler)
        outputs.append(
            sampler(
                raw_logits.clone(),
                torch.tensor([0]),
                np.array([0]),
                np.array([0, 1]),
                torch.tensor([0]),
                torch.tensor([0]),
                torch.tensor([0]),
            )
        )

    expected_raw = torch.log_softmax(raw_logits, dim=-1)[:, 1:2]
    expected_scaled = torch.log_softmax(ar_logits, dim=-1)[:, 1:2]
    torch.testing.assert_close(captured_logprobs[0], expected_raw)
    torch.testing.assert_close(captured_logprobs[1], expected_scaled)
    assert not torch.equal(captured_logprobs[0], captured_logprobs[1])
    assert torch.equal(outputs[0].sampled_token_ids, outputs[1].sampled_token_ids)


def test_deferred_ar_logprob_stats_use_pre_filter_snapshot(monkeypatch) -> None:
    sampler = _bare_sampler(return_ar_logprobs=True, top_k=True)
    sampler.sampling_states.max_num_logprobs = lambda _idx_mapping_np: 1
    raw_logits = torch.tensor([[2.0, 1.0, -1.0]])
    captured: dict[str, torch.Tensor] = {}

    def fake_apply_temperature(logits, _idx_mapping, _temperatures) -> None:
        logits.div_(0.6)

    def fake_apply_top_k_top_p(logits, _top_k, _top_p):
        logits[:, 1:] = -torch.inf
        return logits

    def fake_tidar_target_stats(target_logits, _prob_token_ids, logprob_logits=None):
        assert logprob_logits is not None
        captured["target"] = target_logits.clone()
        captured["logprob"] = logprob_logits.clone()
        return (
            torch.tensor([0.5]),
            torch.tensor([1.0]),
            torch.tensor([2.0]),
            torch.tensor([0], dtype=torch.int64),
        )

    monkeypatch.setattr(sampler_module, "apply_temperature", fake_apply_temperature)
    monkeypatch.setattr(sampler_module, "apply_top_k_top_p", fake_apply_top_k_top_p)
    monkeypatch.setattr(sampler_module, "tidar_target_stats", fake_tidar_target_stats)
    monkeypatch.setattr(
        sampler_module,
        "tidar_sample_bonus_tokens",
        lambda *args, **kwargs: torch.tensor([0], dtype=torch.int64),
    )

    result = sampler.sample(
        raw_logits,
        torch.tensor([0]),
        np.array([0]),
        torch.tensor([0]),
        torch.tensor([0]),
        torch.tensor([0]),
        prob_token_ids=torch.tensor([1]),
        tidar_cu_num_logits=torch.tensor([0, 1]),
        collect_tidar_logprob_stats=True,
    )

    torch.testing.assert_close(result[2], raw_logits / 0.6)
    assert torch.isneginf(captured["target"][0, 1])
    assert torch.isfinite(captured["logprob"][0, 1])
    torch.testing.assert_close(captured["logprob"], raw_logits / 0.6)


def test_strict_mode_requires_ar_logprobs(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_TIDAR_ASSERT_AR_LOGPROBS", "1")
    monkeypatch.delenv("VLLM_TIDAR_RETURN_AR_LOGPROBS", raising=False)

    with pytest.raises(
        ValueError,
        match="VLLM_TIDAR_ASSERT_AR_LOGPROBS=1 requires",
    ):
        Sampler(1, 3, torch.device("cpu"))


def test_strict_mode_rejects_request_temperature_mismatch() -> None:
    sampler = Sampler.__new__(Sampler)
    sampler.assert_ar_logprobs = True
    sampler.ar_logprob_temperature = 0.6

    with pytest.raises(ValueError, match="V2 AR logprob temperature mismatch"):
        sampler.add_request(
            req_idx=0,
            prompt_len=1,
            sampling_params=SimpleNamespace(temperature=0.7),
        )
