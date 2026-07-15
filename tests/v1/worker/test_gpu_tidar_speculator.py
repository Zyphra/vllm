# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import torch

import vllm.v1.worker.gpu.spec_decode.tidar as tidar_module
from vllm.v1.worker.gpu.spec_decode.tidar import TiDARSpeculator


def test_dspark_markov_global_reset_and_chain(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_DSPARK_GLOBAL_RESET", "1")
    monkeypatch.delenv("VLLM_DSPARK_NO_MARKOV", raising=False)

    speculator = object.__new__(TiDARSpeculator)
    speculator.num_speculative_steps = 3
    speculator.diff_temperature = 0.0
    speculator.model = SimpleNamespace(
        dspark_markov_enabled=True,
        dspark_block_len=2,
        diffusion_output_layer=SimpleNamespace(weight=torch.eye(4, 2)),
        diffusion_markov_head=SimpleNamespace(
            w1=torch.eye(4, 2),
            w2=torch.tensor(
                [
                    [0.0, 5.0, 0.0, 0.0],
                    [0.0, 0.0, 5.0, 0.0],
                ]
            ),
        ),
    )

    draft_tokens = speculator._dspark_sample_drafts(
        hidden_states=torch.zeros(3, 2),
        batch_size=1,
        mask_positions=torch.tensor([[1, 2, 3]]),
        prev_token=torch.tensor([0]),
        sampling_seeds=torch.tensor([1234]),
    )

    # Position 2 resets to token 0; position 3 chains from position 2's sample.
    assert draft_tokens.tolist() == [1, 1, 2]
    assert speculator.last_draft_probs is None
    assert speculator.last_draft_token_probs is None
    assert speculator.last_draft_logsumexp is None
    assert speculator.last_draft_logits.shape == (3, 4)


def test_dspark_stochastic_keeps_compact_draft_distribution(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_DSPARK_GLOBAL_RESET", "1")
    monkeypatch.delenv("VLLM_DSPARK_NO_MARKOV", raising=False)
    def fake_gumbel_sample_with_probs(logits, *_args, **_kwargs):
        sampled = logits.argmax(dim=-1)
        scaled_logits = logits.float() / 0.7
        logsumexp = torch.logsumexp(scaled_logits, dim=-1)
        selected_logits = scaled_logits.gather(
            1, sampled.unsqueeze(1)).squeeze(1)
        return sampled, torch.exp(selected_logits - logsumexp), logsumexp

    monkeypatch.setattr(
        tidar_module,
        "gumbel_sample_with_probs",
        fake_gumbel_sample_with_probs,
    )

    speculator = object.__new__(TiDARSpeculator)
    speculator.num_speculative_steps = 3
    speculator.diff_temperature = 0.7
    speculator.model = SimpleNamespace(
        dspark_markov_enabled=True,
        dspark_block_len=2,
        diffusion_output_layer=SimpleNamespace(weight=torch.eye(4, 2)),
        diffusion_markov_head=SimpleNamespace(
            w1=torch.eye(4, 2),
            w2=torch.tensor(
                [
                    [0.0, 5.0, 0.0, 0.0],
                    [0.0, 0.0, 5.0, 0.0],
                ]
            ),
        ),
    )

    draft_tokens = speculator._dspark_sample_drafts(
        hidden_states=torch.zeros(3, 2),
        batch_size=1,
        mask_positions=torch.tensor([[1, 2, 3]]),
        prev_token=torch.tensor([0]),
        sampling_seeds=torch.tensor([1234]),
    )

    assert draft_tokens.tolist() == [1, 1, 2]
    assert speculator.last_draft_probs is None
    assert speculator.last_draft_token_probs is not None
    assert speculator.last_draft_logsumexp is not None
    assert speculator.last_draft_logits is not None
    scaled_logits = speculator.last_draft_logits.float() / 0.7
    expected_logsumexp = torch.logsumexp(scaled_logits, dim=-1)
    expected_probs = torch.exp(
        scaled_logits.gather(1, draft_tokens.unsqueeze(1)).squeeze(1)
        - expected_logsumexp
    )
    assert torch.allclose(speculator.last_draft_logsumexp, expected_logsumexp)
    assert torch.allclose(speculator.last_draft_token_probs, expected_probs)


def test_compact_draft_state_stays_in_batch_order() -> None:
    speculator = object.__new__(TiDARSpeculator)
    speculator.num_speculative_steps = 2
    speculator.draft_batch_index_cache = torch.full(
        (4,), -1, dtype=torch.int64
    )
    speculator.last_draft_logits = torch.arange(24).view(4, 6)
    speculator.last_draft_token_probs = torch.arange(4, dtype=torch.float32)
    speculator.last_draft_logsumexp = torch.arange(4, dtype=torch.float32) + 10

    speculator._cache_compact_draft_state(
        torch.tensor([2, 0], dtype=torch.int64), batch_size=2
    )

    assert speculator.draft_batch_index_cache.tolist() == [1, -1, 0, -1]
    assert speculator.draft_logits_cache.shape == (2, 2, 6)
    assert torch.equal(
        speculator.draft_logits_cache,
        speculator.last_draft_logits.view(2, 2, 6),
    )
    assert torch.equal(
        speculator.draft_token_probs_cache,
        speculator.last_draft_token_probs.view(2, 2),
    )
    assert torch.equal(
        speculator.draft_logsumexp_cache,
        speculator.last_draft_logsumexp.view(2, 2),
    )


def test_partial_draft_state_preserves_other_request_rows() -> None:
    speculator = object.__new__(TiDARSpeculator)
    speculator.num_speculative_steps = 2
    speculator.max_num_reqs = 4
    speculator.draft_batch_index_cache = torch.full(
        (4,), -1, dtype=torch.int64
    )
    speculator._draft_logits_by_req = None
    speculator._draft_token_probs_by_req = None
    speculator._draft_logsumexp_by_req = None

    speculator.last_draft_logits = torch.arange(24).view(4, 6)
    speculator.last_draft_token_probs = torch.arange(4, dtype=torch.float32)
    speculator.last_draft_logsumexp = torch.arange(4, dtype=torch.float32) + 10
    speculator.draft_logits_cache = None
    speculator.draft_token_probs_cache = None
    speculator.draft_logsumexp_cache = None
    speculator._cache_compact_draft_state(
        torch.tensor([2, 0], dtype=torch.int64), batch_size=2
    )
    old_req_2_logits = speculator.draft_logits_cache[0].clone()
    old_req_0_logits = speculator.draft_logits_cache[1].clone()

    speculator.last_draft_logits = torch.arange(12).view(2, 6) + 100
    speculator.last_draft_token_probs = torch.tensor([20.0, 21.0])
    speculator.last_draft_logsumexp = torch.tensor([30.0, 31.0])
    speculator._cache_compact_draft_state(
        torch.tensor([1], dtype=torch.int64),
        batch_size=1,
        preserve_existing=True,
    )

    assert speculator.draft_batch_index_cache.tolist() == [0, 1, 2, 3]
    assert torch.equal(speculator.draft_logits_cache[2], old_req_2_logits)
    assert torch.equal(speculator.draft_logits_cache[0], old_req_0_logits)
    assert torch.equal(
        speculator.draft_logits_cache[1],
        speculator.last_draft_logits.view(1, 2, 6)[0],
    )
    assert torch.equal(
        speculator.draft_token_probs_cache[1], torch.tensor([20.0, 21.0])
    )
    assert torch.equal(
        speculator.draft_logsumexp_cache[1], torch.tensor([30.0, 31.0])
    )
