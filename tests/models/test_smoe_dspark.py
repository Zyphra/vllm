# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.utils.dspark import (
    enforce_dspark_target_contract,
    validate_dspark_load_format,
    validate_dspark_target_contract,
    validate_tidar_temperature_contract,
)


class _ContractTarget(SimpleNamespace):
    def compute_logits(self, hidden_states):
        return self.logits_processor(self.lm_head, hidden_states)


class _BadContractTarget(_ContractTarget):
    def compute_logits(self, hidden_states):
        return self.logits_processor(
            self.diffusion_output_layer,
            hidden_states,
        )


def _contract_target(
    *,
    tied: bool = True,
    trainable_draft: bool = False,
    alias_draft: bool = False,
):
    embedding = torch.nn.Parameter(torch.empty(4, 2))
    lm_head = embedding if tied else torch.nn.Parameter(torch.empty(4, 2))
    draft = (
        lm_head
        if alias_draft
        else torch.nn.Parameter(
            torch.empty(4, 2), requires_grad=trainable_draft
        )
    )
    return _ContractTarget(
        config=SimpleNamespace(
            tie_word_embeddings=True,
            vocab_size=4,
            hidden_size=2,
        ),
        model=SimpleNamespace(
            embed_tokens=SimpleNamespace(weight=embedding),
        ),
        lm_head=SimpleNamespace(weight=lm_head),
        diffusion_output_layer=SimpleNamespace(weight=draft),
        diffusion_markov_head=SimpleNamespace(
            w1=torch.nn.Parameter(torch.empty(4, 3), requires_grad=False),
            w2=torch.nn.Parameter(torch.empty(3, 4), requires_grad=False),
        ),
    )


def test_dspark_rejects_dummy_load_format(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_TIDAR_DSPARK", "1")

    with pytest.raises(ValueError, match="checkpoint-only diffusion_output_layer"):
        validate_dspark_load_format(True, "dummy")


def test_dspark_allows_checkpoint_loader(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_TIDAR_DSPARK", "1")

    validate_dspark_load_format(True, "auto")


def test_disabled_dspark_allows_dummy_loader(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_TIDAR_DSPARK", "0")

    validate_dspark_load_format(True, "dummy")


def test_dspark_target_contract_accepts_tied_ar_and_frozen_draft_heads() -> None:
    target = _contract_target()

    validate_dspark_target_contract(target, retained_target_model=target)


def test_dspark_target_contract_rejects_untied_ar_head() -> None:
    target = _contract_target(tied=False)

    with pytest.raises(RuntimeError, match="not the same Parameter"):
        validate_dspark_target_contract(target)


def test_dspark_target_contract_rejects_trainable_draft_head() -> None:
    target = _contract_target(trainable_draft=True)

    with pytest.raises(RuntimeError, match="must be frozen"):
        validate_dspark_target_contract(target)


def test_dspark_target_contract_rejects_ar_draft_alias() -> None:
    target = _contract_target(alias_draft=True)

    with pytest.raises(RuntimeError, match="aliases regular lm_head"):
        validate_dspark_target_contract(target)


def test_dspark_target_contract_rejects_missing_markov_head() -> None:
    target = _contract_target()
    del target.diffusion_markov_head

    with pytest.raises(RuntimeError, match="missing a required AR/draft head"):
        validate_dspark_target_contract(target)


def test_dspark_target_contract_rejects_draft_verifier_head() -> None:
    target = _contract_target()
    target.__class__ = _BadContractTarget

    with pytest.raises(RuntimeError, match="target compute_logits must"):
        validate_dspark_target_contract(target)


def test_required_dspark_target_contract_validates_active_target() -> None:
    target = _contract_target()

    enforce_dspark_target_contract(
        target,
        retained_target_model=target,
        dspark_active=True,
        required=True,
    )


def test_required_dspark_target_contract_rejects_inactive_dspark() -> None:
    target = _contract_target()

    with pytest.raises(RuntimeError, match="requires the loaded target checkpoint"):
        enforce_dspark_target_contract(
            target,
            retained_target_model=target,
            dspark_active=False,
            required=True,
        )


def test_tidar_temperature_contract_accepts_one_shared_temperature() -> None:
    validate_tidar_temperature_contract(
        draft_temperature=0.6,
        ar_temperature=0.6,
        configured_ar_temperature="0.6",
    )


@pytest.mark.parametrize(
    ("draft_temperature", "ar_temperature", "configured_ar_temperature"),
    [
        (0.7, 0.6, "0.6"),
        (0.6, 1.0, "0.6"),
        (0.6, 0.6, None),
        (0.6, 0.6, "nan"),
    ],
)
def test_tidar_temperature_contract_rejects_drift(
    draft_temperature,
    ar_temperature,
    configured_ar_temperature,
) -> None:
    with pytest.raises(ValueError, match="TiDAR temperature contract failed"):
        validate_tidar_temperature_contract(
            draft_temperature=draft_temperature,
            ar_temperature=ar_temperature,
            configured_ar_temperature=configured_ar_temperature,
        )
