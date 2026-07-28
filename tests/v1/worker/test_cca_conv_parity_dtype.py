# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import subprocess
import sys

import torch
import torch.nn.functional as F
from torch import nn

from vllm.model_executor.layers.mamba import cca as cca_module
from vllm.model_executor.layers.mamba.cca import CCA


def test_megatron_parity_env_flag_is_opt_in() -> None:
    code = (
        "from vllm.model_executor.layers.mamba.cca import "
        "_CCA_MATCH_MEGATRON_CONV_DTYPE_ENABLED as enabled; "
        "print(int(enabled))"
    )
    for value, expected in ((None, "0"), ("1", "1")):
        env = os.environ.copy()
        if value is None:
            env.pop("VLLM_CCA_MATCH_MEGATRON_CONV_DTYPE", None)
        else:
            env["VLLM_CCA_MATCH_MEGATRON_CONV_DTYPE"] = value
        result = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        assert result.stdout.strip().endswith(expected)


def _make_lightweight_cca() -> CCA:
    cca = CCA.__new__(CCA)
    nn.Module.__init__(cca)
    cca.num_k_heads = 1
    cca.num_q_heads = 1
    cca.conv_qk = nn.Sequential(
        nn.Conv1d(
            4,
            4,
            kernel_size=2,
            groups=4,
            dtype=torch.bfloat16,
        ),
        nn.Conv1d(
            4,
            4,
            kernel_size=2,
            groups=2,
            dtype=torch.bfloat16,
        ),
    )
    cca._conv_qk_fp32_cache = None
    cca._conv_qk_fp32_versions = None

    with torch.no_grad():
        for index, parameter in enumerate(cca.conv_qk.parameters()):
            values = torch.linspace(
                -0.9 + 0.1 * index,
                1.1 + 0.1 * index,
                parameter.numel(),
                dtype=torch.float32,
            )
            parameter.copy_(values.reshape_as(parameter))
    return cca


def _fp32_two_stage_conv(cca: CCA, x: torch.Tensor) -> torch.Tensor:
    conv0, conv1 = cca.conv_qk
    mid = F.conv1d(
        x.float(),
        conv0.weight.float(),
        None if conv0.bias is None else conv0.bias.float(),
        stride=conv0.stride,
        padding=conv0.padding,
        dilation=conv0.dilation,
        groups=conv0.groups,
    )
    return F.conv1d(
        mid,
        conv1.weight.float(),
        None if conv1.bias is None else conv1.bias.float(),
        stride=conv1.stride,
        padding=conv1.padding,
        dilation=conv1.dilation,
        groups=conv1.groups,
    )


def test_megatron_parity_path_rounds_between_conv_stages(
    monkeypatch,
) -> None:
    cca = _make_lightweight_cca()
    x = torch.linspace(
        -1.3,
        1.7,
        2 * 4 * 7,
        dtype=torch.float32,
    ).reshape(2, 4, 7)

    monkeypatch.setattr(
        cca_module,
        "_CCA_BATCH_INVARIANT_CONV_ENABLED",
        False,
    )
    monkeypatch.setattr(
        cca_module,
        "_CCA_MATCH_MEGATRON_CONV_DTYPE_ENABLED",
        True,
    )

    actual = cca._conv_qk_apply(x)
    expected = cca.conv_qk(x.to(torch.bfloat16))

    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.equal(actual.float(), _fp32_two_stage_conv(cca, x))
    assert cca._conv_qk_fp32_cache is None


def test_default_path_remains_two_stage_fp32(monkeypatch) -> None:
    cca = _make_lightweight_cca()
    x = torch.linspace(
        -1.3,
        1.7,
        2 * 4 * 7,
        dtype=torch.bfloat16,
    ).reshape(2, 4, 7)

    monkeypatch.setattr(
        cca_module,
        "_CCA_BATCH_INVARIANT_CONV_ENABLED",
        False,
    )
    monkeypatch.setattr(
        cca_module,
        "_CCA_MATCH_MEGATRON_CONV_DTYPE_ENABLED",
        False,
    )

    actual = cca._conv_qk_apply(x)
    expected = _fp32_two_stage_conv(cca, x)

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert cca._conv_qk_fp32_cache is not None


def test_batch_invariant_path_keeps_precedence(monkeypatch) -> None:
    cca = _make_lightweight_cca()
    x = torch.zeros(2, 4, 7, dtype=torch.bfloat16)
    sentinel = torch.full((2, 4, 5), 3.0, dtype=torch.float32)
    calls = []

    def fake_batch_invariant(*args):
        calls.append(args)
        return sentinel

    monkeypatch.setattr(
        cca_module,
        "cca_conv1d_batch_invariant",
        fake_batch_invariant,
    )
    monkeypatch.setattr(
        cca_module,
        "_CCA_BATCH_INVARIANT_CONV_ENABLED",
        True,
    )
    monkeypatch.setattr(
        cca_module,
        "_CCA_MATCH_MEGATRON_CONV_DTYPE_ENABLED",
        True,
    )

    actual = cca._conv_qk_apply(x)

    assert actual is sentinel
    assert len(calls) == 1
    assert calls[0][0] is x
    assert cca._conv_qk_fp32_cache is None
