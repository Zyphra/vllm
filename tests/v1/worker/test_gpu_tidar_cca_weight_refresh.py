# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.mamba.cca import CCA
from vllm.model_executor.models import smoe
from vllm.model_executor.models.smoe import SMoEForCausalLM


def _make_lightweight_cca(device: torch.device) -> CCA:
    """Construct only the CCA state needed by runtime-weight refresh."""
    cca = CCA.__new__(CCA)
    nn.Module.__init__(cca)

    cca.num_k_heads = 1
    cca.num_q_heads = 1
    cca.sqrt_head_dim = 2.0
    cca.config = SimpleNamespace(clamp_temp=False)

    # Two heads with head_dim=2. This preserves the production depthwise then
    # grouped-convolution geometry while keeping the test very small.
    cca.conv_qk = nn.Sequential(
        nn.Conv1d(4, 4, kernel_size=2, groups=4, device=device,
                  dtype=torch.bfloat16),
        nn.Conv1d(4, 4, kernel_size=2, groups=2, device=device,
                  dtype=torch.bfloat16),
    )
    cca.temp = nn.Parameter(
        torch.tensor([0.25], device=device, dtype=torch.bfloat16),
        requires_grad=False,
    )

    cca._gw_weight_T = None
    cca._conv_qk_fp32_cache = None
    cca._conv_qk_fp32_versions = None
    cca._temp_fp32_cache = None
    cca._temp_fp32_key = None
    cca.refresh_runtime_weight_views()
    return cca


def _materialize_derived_caches(cca: CCA) -> None:
    # The FP32 convolution and temperature caches are lazy in production.
    conv_input = torch.randn(
        1, 4, 4, device=cca.temp.device, dtype=torch.bfloat16)
    cca._conv_qk_apply(conv_input)
    query = torch.randn(
        1, 1, 2, device=cca.temp.device, dtype=torch.bfloat16)
    key = torch.randn_like(query)
    cca._rms_normalize_qk(query, key, torch.bfloat16)

    # The transposed grouped-convolution weight is lazy in the fused decode
    # path. Materialize it exactly as that path does before graph capture.
    gw = cca.gw_weight_flat
    cca._gw_weight_T = gw.permute(0, 2, 3, 1).contiguous().view(
        gw.shape[0], gw.shape[2] * gw.shape[3], gw.shape[1])


def _derived_cache_tensors(cca: CCA) -> list[torch.Tensor]:
    assert cca._conv_qk_fp32_cache is not None
    assert cca._temp_fp32_cache is not None
    assert cca._gw_weight_T is not None
    conv0_weight, conv0_bias = cca._conv_qk_fp32_cache[0]
    conv1_weight, conv1_bias = cca._conv_qk_fp32_cache[1]
    assert conv0_bias is not None
    assert conv1_bias is not None
    assert cca.gw_bias_flat is not None
    return [
        conv0_weight,
        conv0_bias,
        conv1_weight,
        conv1_bias,
        cca._temp_fp32_cache,
        cca.dw_weight_flat,
        cca.gw_weight_flat,
        cca.gw_bias_flat,
        cca._gw_weight_T,
    ]


def _write_cache_signature(cca: CCA, output: torch.Tensor) -> None:
    for index, tensor in enumerate(_derived_cache_tensors(cca)):
        output[index].copy_(tensor.float().sum())


def _expected_signature_from_parameters(cca: CCA) -> torch.Tensor:
    conv0 = cca.conv_qk[0]
    conv1 = cca.conv_qk[1]
    assert conv0.bias is not None
    assert conv1.bias is not None
    groups = cca.num_k_heads + cca.num_q_heads
    dim, _, kernel_width = conv0.weight.shape
    head_dim = dim // groups
    dw = conv0.weight.reshape(dim, kernel_width)
    gw = conv1.weight.reshape(groups, head_dim, -1, kernel_width)
    gw_bias = conv1.bias.reshape(groups, -1)
    gw_t = gw.permute(0, 2, 3, 1).contiguous().view(
        gw.shape[0], gw.shape[2] * gw.shape[3], gw.shape[1])
    return torch.stack([
        conv0.weight.float().sum(),
        conv0.bias.float().sum(),
        conv1.weight.float().sum(),
        conv1.bias.float().sum(),
        cca.temp.float().sum(),
        dw.float().sum(),
        gw.float().sum(),
        gw_bias.float().sum(),
        gw_t.float().sum(),
    ])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cca_runtime_weight_refresh_preserves_cudagraph_pointers() -> None:
    torch.manual_seed(7)
    device = torch.device("cuda")
    cca = _make_lightweight_cca(device)
    _materialize_derived_caches(cca)

    cache_tensors = _derived_cache_tensors(cca)
    pointers_before = [tensor.data_ptr() for tensor in cache_tensors]

    graph_output = torch.empty(len(cache_tensors), device=device,
                               dtype=torch.float32)
    # Warm the exact operations before capture so CUDA graph setup itself does
    # not obscure the pointer/content invariant under test.
    _write_cache_signature(cca, graph_output)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _write_cache_signature(cca, graph_output)
    graph.replay()
    torch.cuda.synchronize()
    output_before_update = graph_output.clone()

    # Match live model.load_weights(): vLLM loaders mutate Parameter.data in
    # place, which preserves parameter pointers and does not bump _version.
    with torch.no_grad():
        cca.conv_qk[0].weight.data.add_(0.5)
        cca.conv_qk[0].bias.data.add_(0.75)
        cca.conv_qk[1].weight.data.sub_(0.25)
        cca.conv_qk[1].bias.data.add_(1.25)
        cca.temp.data.add_(0.5)

    expected_after_update = _expected_signature_from_parameters(cca)
    cca.refresh_runtime_weight_views()

    cache_tensors_after = _derived_cache_tensors(cca)
    assert [tensor.data_ptr() for tensor in cache_tensors_after] == pointers_before

    # Replaying the already-captured graph must consume the refreshed contents
    # at the original addresses; recapturing would hide a stale-pointer bug.
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        graph_output,
        expected_after_update,
        rtol=0,
        atol=0,
    )
    assert not torch.equal(output_before_update, graph_output)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_smoe_runtime_loader_refreshes_cca_caches(monkeypatch) -> None:
    device = torch.device("cuda")
    cca = _make_lightweight_cca(device)
    _materialize_derived_caches(cca)
    pointers_before = [
        tensor.data_ptr() for tensor in _derived_cache_tensors(cca)
    ]

    model = SMoEForCausalLM.__new__(SMoEForCausalLM)
    nn.Module.__init__(model)
    model.cca = cca
    monkeypatch.setattr(smoe, "get_tensor_model_parallel_rank", lambda: 1)

    new_values = {
        "cca.conv_qk.0.weight": torch.full_like(
            cca.conv_qk[0].weight, 0.5),
        "cca.conv_qk.0.bias": torch.full_like(
            cca.conv_qk[0].bias, 0.75),
        "cca.conv_qk.1.weight": torch.full_like(
            cca.conv_qk[1].weight, -0.25),
        "cca.conv_qk.1.bias": torch.full_like(
            cca.conv_qk[1].bias, 1.25),
        "cca.temp": torch.full_like(cca.temp, 0.5),
    }
    loaded = model.load_weights(new_values.items())

    assert loaded == set(new_values)
    assert [
        tensor.data_ptr() for tensor in _derived_cache_tensors(cca)
    ] == pointers_before
    actual = torch.empty(9, device=device, dtype=torch.float32)
    _write_cache_signature(cca, actual)
    torch.testing.assert_close(
        actual,
        _expected_signature_from_parameters(cca),
        rtol=0,
        atol=0,
    )
