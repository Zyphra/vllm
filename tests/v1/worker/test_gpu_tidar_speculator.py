# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import torch

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
    )

    # Position 2 resets to token 0; position 3 chains from position 2's sample.
    assert draft_tokens.tolist() == [1, 1, 2]
    assert speculator.last_draft_probs is None
    assert speculator.last_draft_logits.shape == (3, 4)
