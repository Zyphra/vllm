# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.mamba.cca import (
    _gather_cached_state_or_zeros,
)
from vllm.model_executor.layers.mamba.ops import (
    fused_pad_gather_scatter,
    run_causal_conv1d_update,
)


def test_cold_and_invalid_cca_rows_read_zero_state() -> None:
    cache = torch.tensor(
        [
            [[10.0, 11.0], [12.0, 13.0]],
            [[20.0, 21.0], [22.0, 23.0]],
        ]
    )
    state_indices = torch.tensor([1, 0, -1, 99])
    has_initial_states = torch.tensor([True, False, True, True])

    actual = _gather_cached_state_or_zeros(
        cache,
        state_indices,
        has_initial_states,
        torch.float32,
    )

    assert torch.equal(actual[0], cache[1])
    assert torch.count_nonzero(actual[1:]) == 0


def test_cca_cached_state_gather_converts_dtype() -> None:
    cache = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    actual = _gather_cached_state_or_zeros(
        cache,
        torch.tensor([0]),
        torch.tensor([True]),
        torch.bfloat16,
    )
    assert actual.dtype == torch.bfloat16
    assert torch.equal(actual.float(), cache)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA/ROCm GPU")
@pytest.mark.parametrize("use_cuda_graph", [False, True])
def test_masked_cache_scatter_preserves_slot_zero_and_invalid_rows(
    use_cuda_graph: bool,
) -> None:
    """An active slot 0 must not alias simultaneous PAD/OOB rows."""
    device = torch.device("cuda")
    initial = torch.arange(
        4 * 2 * 2, device=device, dtype=torch.float32).view(4, 2, 2)
    cache = initial.clone()
    indices = torch.tensor([0, -1, 2, 99, -7], device=device)
    values = torch.stack([
        torch.full((2, 2), 101.0, device=device),
        torch.full((2, 2), float("nan"), device=device),
        torch.full((2, 2), 202.0, device=device),
        torch.full((2, 2), float("nan"), device=device),
        torch.full((2, 2), float("nan"), device=device),
    ])

    if use_cuda_graph:
        # Warm Triton before capture, then capture the same raw-indexed
        # mutation primitive used by forward_triton's graph path.
        warm_cache = initial.clone()
        fused_pad_gather_scatter(indices, values, warm_cache)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            gathered, invalid = fused_pad_gather_scatter(
                indices, values, cache)
        cache.copy_(initial)
        graph.replay()
        torch.cuda.synchronize()
    else:
        gathered, invalid = fused_pad_gather_scatter(indices, values, cache)

    assert invalid.tolist() == [False, True, False, True, True]
    assert torch.equal(gathered[0], initial[0])
    assert torch.equal(gathered[2], initial[2])
    assert torch.isfinite(gathered).all()
    assert torch.count_nonzero(gathered[invalid]) == 0
    assert torch.equal(cache[0], values[0])
    assert torch.equal(cache[2], values[2])
    assert torch.equal(cache[1], initial[1])
    assert torch.equal(cache[3], initial[3])


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA/ROCm GPU")
def test_raw_index_conv_incremental_padded_matches_unpadded() -> None:
    """PAD/OOB rows produce zero and cannot perturb incremental cache state."""
    device = torch.device("cuda")
    torch.manual_seed(7)
    initial = torch.randn(4, 8, 2, device=device)
    cache_padded = initial.clone()
    cache_unpadded = initial.clone()
    weight = torch.randn(8, 2, device=device)
    bias = torch.randn(8, device=device)
    padded_indices = torch.tensor([0, -1, 2, 99], device=device)
    active_indices = torch.tensor([0, 2], device=device)

    for _ in range(3):
        active_x = torch.randn(2, 8, 1, device=device)
        padded_x = torch.full(
            (4, 8, 1), float("nan"), device=device)
        padded_x[[0, 2]] = active_x

        out_padded = run_causal_conv1d_update(
            padded_x,
            cache_padded,
            weight,
            bias,
            padded_indices,
            seqlen=1,
        )
        out_unpadded = run_causal_conv1d_update(
            active_x,
            cache_unpadded,
            weight,
            bias,
            active_indices,
            seqlen=1,
        )

        assert torch.allclose(
            out_padded[[0, 2]], out_unpadded, rtol=0, atol=0)
        assert torch.isfinite(out_padded).all()
        assert torch.count_nonzero(out_padded[[1, 3]]) == 0
        assert torch.equal(cache_padded, cache_unpadded)
