# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm._custom_ops as ops

QK_WIDTH = 1280
VALUE_WIDTH = 128
OUTPUT_WIDTH = 1536
SLOTS = 8

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="requires a ROCm GPU",
)


def _packed_states(*, overlap: bool) -> tuple[torch.Tensor, torch.Tensor]:
    page = torch.empty((SLOTS, 16384), dtype=torch.uint8, device="cuda")
    conv_state = page[:, :10240].view(torch.float32).view(SLOTS, QK_WIDTH, 2)
    recurrent_offset = 10236 if overlap else 10240
    recurrent_state = (
        page[:, recurrent_offset : recurrent_offset + 512]
        .view(torch.float32)
        .view(SLOTS, VALUE_WIDTH)
    )
    return conv_state, recurrent_state


def _inputs():
    generator = torch.Generator(device="cuda").manual_seed(20260727)
    qk0 = torch.randn((1, QK_WIDTH), generator=generator, device="cuda").bfloat16()
    v_current = torch.randn(
        (1, VALUE_WIDTH), generator=generator, device="cuda"
    ).bfloat16()
    v_delayed = torch.randn(
        (1, VALUE_WIDTH), generator=generator, device="cuda"
    ).bfloat16()
    state_idx = torch.tensor([2], dtype=torch.int32, device="cuda")
    conv_window = torch.empty((1, QK_WIDTH, 3), dtype=torch.bfloat16, device="cuda")
    conv_tail = torch.empty((1, QK_WIDTH), dtype=torch.float32, device="cuda")
    qkv_out = torch.randn(
        (1, OUTPUT_WIDTH), generator=generator, device="cuda"
    ).bfloat16()
    return (
        qk0,
        v_current,
        v_delayed,
        state_idx,
        conv_window,
        conv_tail,
        qkv_out,
    )


def test_bind_kv_cache_equivalent_disjoint_state_views_are_exact():
    conv_state, recurrent_state = _packed_states(overlap=False)
    assert conv_state.stride(0) * conv_state.element_size() == 16384
    assert recurrent_state.stride(0) * recurrent_state.element_size() == 16384
    generator = torch.Generator(device="cuda").manual_seed(20260728)
    conv_state.copy_(torch.randn(conv_state.shape, generator=generator, device="cuda"))
    recurrent_state.copy_(
        torch.randn(recurrent_state.shape, generator=generator, device="cuda")
    )
    (
        qk0,
        v_current,
        v_delayed,
        state_idx,
        conv_window,
        conv_tail,
        qkv_out,
    ) = _inputs()
    initial_conv = conv_state.clone()
    initial_recurrent = recurrent_state.clone()
    initial_qkv = qkv_out.clone()

    ops.cca_decode_state_prepare(
        qk0,
        v_current,
        conv_state,
        recurrent_state,
        state_idx,
        conv_window,
        conv_tail,
        qkv_out,
    )
    assert torch.equal(conv_window[0, :, 0], initial_conv[2, :, 0].bfloat16())
    assert torch.equal(conv_window[0, :, 1], initial_conv[2, :, 1].bfloat16())
    assert torch.equal(conv_window[0, :, 2], qk0[0])
    assert torch.equal(conv_tail[0], initial_conv[2, :, 1])
    assert torch.equal(qkv_out[0, :QK_WIDTH], initial_qkv[0, :QK_WIDTH])
    assert torch.equal(qkv_out[0, QK_WIDTH : QK_WIDTH + VALUE_WIDTH], v_current[0])
    assert torch.equal(
        qkv_out[0, QK_WIDTH + VALUE_WIDTH :],
        initial_recurrent[2].bfloat16(),
    )

    ops.cca_decode_state_commit(
        qk0,
        v_delayed,
        state_idx,
        conv_tail,
        conv_state,
        recurrent_state,
        qkv_out,
    )
    torch.cuda.synchronize()
    assert torch.equal(conv_state[2, :, 0], initial_conv[2, :, 1])
    assert torch.equal(conv_state[2, :, 1], qk0[0].float())
    assert torch.equal(recurrent_state[2], v_delayed[0].float())


def test_overlapping_packed_state_views_fail_closed():
    conv_state, recurrent_state = _packed_states(overlap=True)
    (
        qk0,
        _,
        v_delayed,
        state_idx,
        _,
        conv_tail,
        qkv_out,
    ) = _inputs()
    with pytest.raises(
        RuntimeError,
        match="mutable outputs must not share storage with inputs or each other",
    ):
        ops.cca_decode_state_commit(
            qk0,
            v_delayed,
            state_idx,
            conv_tail,
            conv_state,
            recurrent_state,
            qkv_out,
        )
