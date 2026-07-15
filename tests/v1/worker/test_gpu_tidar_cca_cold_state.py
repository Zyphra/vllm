# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.mamba.cca import (
    _gather_cached_state_or_zeros,
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
