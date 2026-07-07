# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.layers.mamba.cca import CCA


def _make_cca(
    *,
    block_size: int = 4,
    total_padding: int = 2,
    mamba_cache_mode: str = "all",
) -> CCA:
    cca = CCA.__new__(CCA)
    object.__setattr__(
        cca,
        "cache_config",
        SimpleNamespace(
            mamba_cache_mode=mamba_cache_mode,
            mamba_block_size=block_size,
        ),
    )
    object.__setattr__(cca, "total_padding", total_padding)
    return cca


def test_decode_all_mode_uses_computed_block_for_input_and_scheduled_for_output():
    cca = _make_cca()
    state_indices = torch.tensor(
        [
            [10, 11, 12, 13],
            [20, 21, 22, 23],
            [30, 31, 32, 33],
        ],
        dtype=torch.int32,
    )
    last_computed = torch.tensor([0, 2, 1], dtype=torch.int32)
    last_scheduled = torch.tensor([1, 2, 3], dtype=torch.int32)

    input_indices = cca._select_decode_input_state_indices(
        state_indices, last_computed)
    output_indices = cca._select_decode_output_state_indices(
        state_indices, last_scheduled)

    assert input_indices.tolist() == [10, 22, 31]
    assert output_indices.tolist() == [11, 22, 33]


def test_prefill_all_mode_skips_decode_rows_when_gathering_block_indices():
    cca = _make_cca()
    block_idx_last_computed = torch.tensor([1, 2, 0, 2], dtype=torch.int32)
    block_idx_last_scheduled = torch.tensor([1, 3, 1, 3], dtype=torch.int32)
    num_decodes = 2
    prefill_state_indices = torch.tensor(
        [
            [40, 41, 42, 43],
            [50, 51, 52, 53],
        ],
        dtype=torch.int32,
    )

    initial = cca._select_prefill_initial_state_indices(
        prefill_state_indices,
        block_idx_last_computed,
        num_decodes,
    )
    output = cca._select_prefill_output_state_indices(
        prefill_state_indices,
        block_idx_last_scheduled,
        num_decodes,
    )

    assert initial.tolist() == [40, 52]
    assert output.tolist() == [41, 53]


def test_prefill_boundary_writes_newly_completed_full_blocks():
    block_size = 4
    total_padding = 3
    channels = 2
    recurrent_dim = 2
    num_computed_tokens = 4
    num_scheduled = 8
    cca = _make_cca(block_size=block_size, total_padding=total_padding)

    state_indices = torch.tensor([[100, 101, 102, 103]], dtype=torch.int64)
    conv_states = torch.full(
        (104, channels, total_padding), -1.0, dtype=torch.float32)
    recurrent_states = torch.full(
        (104, recurrent_dim), -1.0, dtype=torch.float32)
    qk_source = torch.arange(
        channels * (total_padding + num_scheduled), dtype=torch.float32
    ).view(1, channels, total_padding + num_scheduled)
    delayed_v = torch.arange(
        num_scheduled * recurrent_dim, dtype=torch.float32
    ).view(num_scheduled, 1, recurrent_dim)

    cca._write_prefill_boundary_states(
        conv_states,
        recurrent_states,
        state_indices,
        0,
        qk_source,
        delayed_v,
        torch.tensor(num_computed_tokens),
    )

    torch.testing.assert_close(conv_states[101], qk_source[0, :, 4:7])
    torch.testing.assert_close(recurrent_states[101], delayed_v[3, 0, :])
    torch.testing.assert_close(conv_states[102], qk_source[0, :, 8:11])
    torch.testing.assert_close(recurrent_states[102], delayed_v[7, 0, :])

    torch.testing.assert_close(
        conv_states[100], torch.full_like(conv_states[100], -1.0))
    torch.testing.assert_close(
        conv_states[103], torch.full_like(conv_states[103], -1.0))


def test_prefill_boundary_does_not_write_partial_tail_as_full_block():
    block_size = 4
    total_padding = 2
    channels = 1
    recurrent_dim = 1
    num_computed_tokens = 6
    num_scheduled = 3
    cca = _make_cca(block_size=block_size, total_padding=total_padding)

    state_indices = torch.tensor([[200, 201, 202]], dtype=torch.int64)
    conv_states = torch.full(
        (203, channels, total_padding), -1.0, dtype=torch.float32)
    recurrent_states = torch.full(
        (203, recurrent_dim), -1.0, dtype=torch.float32)
    qk_source = torch.arange(
        channels * (total_padding + num_scheduled), dtype=torch.float32
    ).view(1, channels, total_padding + num_scheduled)
    delayed_v = torch.arange(
        num_scheduled * recurrent_dim, dtype=torch.float32
    ).view(num_scheduled, 1, recurrent_dim)

    cca._write_prefill_boundary_states(
        conv_states,
        recurrent_states,
        state_indices,
        0,
        qk_source,
        delayed_v,
        torch.tensor(num_computed_tokens),
    )

    torch.testing.assert_close(conv_states[201], qk_source[0, :, 2:4])
    torch.testing.assert_close(recurrent_states[201], delayed_v[1, 0, :])
    torch.testing.assert_close(
        conv_states[202], torch.full_like(conv_states[202], -1.0))


def test_prefill_boundary_uses_pre_conv_qk_tail_for_total_padding():
    for total_padding in (1, 3, 4):
        block_size = 4
        channels = 2
        recurrent_dim = 1
        num_computed_tokens = 0
        num_scheduled = 4
        cca = _make_cca(block_size=block_size, total_padding=total_padding)

        state_indices = torch.tensor([[7]], dtype=torch.int64)
        conv_states = torch.full(
            (8, channels, total_padding), -1.0, dtype=torch.float32)
        recurrent_states = torch.full(
            (8, recurrent_dim), -1.0, dtype=torch.float32)
        qk_source = torch.arange(
            channels * (total_padding + num_scheduled), dtype=torch.float32
        ).view(1, channels, total_padding + num_scheduled)
        delayed_v = torch.arange(
            num_scheduled * recurrent_dim, dtype=torch.float32
        ).view(num_scheduled, 1, recurrent_dim)

        cca._write_prefill_boundary_states(
            conv_states,
            recurrent_states,
            state_indices,
            0,
            qk_source,
            delayed_v,
            torch.tensor(num_computed_tokens),
        )

        expected_start = block_size
        expected_end = block_size + total_padding
        torch.testing.assert_close(
            conv_states[7], qk_source[0, :, expected_start:expected_end])
        torch.testing.assert_close(recurrent_states[7], delayed_v[3, 0, :])
