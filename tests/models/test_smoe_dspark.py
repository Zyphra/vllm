# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.utils.dspark import validate_dspark_load_format


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
