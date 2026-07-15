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
