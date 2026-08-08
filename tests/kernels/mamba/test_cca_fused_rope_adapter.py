# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
CCA_PATH = ROOT / "vllm/model_executor/layers/mamba/cca.py"
ZAYA_PATH = ROOT / "vllm/model_executor/models/zaya.py"
PAD = -1
_RUNTIME = SimpleNamespace(context=None, zk=None, zk_prefill=None)
_Metadata = type("_Metadata", (), {})


class _ZkFallback(Exception):
    pass


def _source(path):
    return path.read_text()


def _production_class(path, class_name, methods, namespace):
    tree = ast.parse(_source(path))
    source_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    result = type(f"Executable{class_name}", (), {})
    for node in source_class.body:
        if isinstance(node, ast.FunctionDef) and node.name in methods:
            module = ast.fix_missing_locations(ast.Module([node], type_ignores=[]))
            exec(compile(module, str(path), "exec"), namespace)
            setattr(result, node.name, namespace[node.name])
    return result


_CCA = _production_class(
    CCA_PATH,
    "CCA",
    {
        "_add_grouped_qk_means_inplace",
        "_apply_rope_to_output",
        "_finalize_qk",
        "_forward_no_cache",
        "_forward_zk_decode",
        "_get_zk_rope_cache",
        "_rms_normalize_qk",
        "forward_cuda",
    },
    {
        "AttentionMetadata": object,
        "CCAAttentionMetadata": _Metadata,
        "F": F,
        "PAD_SLOT_ID": PAD,
        "_USE_ZK_CCA": True,
        "_ZK_CCA_DECODE_IMPORTED": True,
        "_ZK_CCA_PREFILL_IMPORTED": True,
        "_ZkFallback": _ZkFallback,
        "get_forward_context": lambda: _RUNTIME.context,
        "_zk_cca_decode": lambda *args, **kwargs: _RUNTIME.zk(*args, **kwargs),
        "_zk_cca_decode_available": lambda: _RUNTIME.zk is not None,
        "_zk_cca_prefill": lambda *args, **kwargs: _RUNTIME.zk_prefill(
            *args, **kwargs
        ),
        "_zk_cca_prefill_available": lambda: _RUNTIME.zk_prefill is not None,
        "logger": SimpleNamespace(
            exception=lambda *args, **kwargs: None,
            info_once=lambda *args, **kwargs: None,
            warning=lambda *args, **kwargs: None,
        ),
        "nn": nn,
        "torch": torch,
    },
)
_Attention = _production_class(
    ZAYA_PATH, "ZayaAttention", {"forward"}, {"torch": torch}
)


class _Call:
    def __init__(self, function, **attributes):
        self.function = function
        self.calls = []
        self.__dict__.update(attributes)

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.function(*args, **kwargs)


def _zk_op(*args):
    first, state, indices = args[0], args[3], args[4]
    output = -first[:, :, 0]
    for row, index in enumerate(indices.tolist()):
        if index == PAD:
            output[row].zero_()
        else:
            state[index, :, 0] = state[index, :, 1]
            state[index, :, 1] = first[row, :, 0].to(state.dtype)
    return output


def _zk_prefill(*args, **kwargs):
    delayed_v, recurrent_state = args[0], args[2]
    query_start, has_initial, state_indices = args[4:7]
    shifted = torch.empty_like(delayed_v)
    for request_index in range(len(query_start) - 1):
        start = int(query_start[request_index])
        end = int(query_start[request_index + 1])
        state_index = int(state_indices[request_index])
        shifted[start] = (
            recurrent_state[state_index]
            if has_initial[request_index]
            else torch.zeros_like(delayed_v[start])
        )
        shifted[start + 1 : end] = delayed_v[start : end - 1]
        recurrent_state[state_index] = delayed_v[end - 1]
    return shifted, -args[1]


def _raise_zk_fallback(*args, **kwargs):
    raise _ZkFallback("unsupported test input")


def _raise_prefill_error(*args, **kwargs):
    raise RuntimeError("test prefill failure")


def _make_cca(num_q_heads=8, state_dtype=torch.bfloat16):
    cca = _CCA()
    head_dim = 128
    latent_q, latent_k = num_q_heads * head_dim, 2 * head_dim
    recurrent, in_out = latent_k // 2, latent_q + latent_k
    rope = _Call(
        lambda positions, query, key: (-query, -key),
        rotary_dim=64,
        cos_sin_cache=torch.zeros(32, 64, dtype=torch.bfloat16),
    )
    rope._match_cos_sin_cache_dtype = lambda reference: rope.cos_sin_cache.to(
        reference.dtype
    )
    conv = lambda: _Call(
        lambda value: value[..., 1:],
        weight=torch.ones(in_out, 1, 2, dtype=torch.bfloat16),
        bias=torch.zeros(in_out, dtype=torch.bfloat16),
    )
    cca.__dict__.update(
        num_q_heads=num_q_heads,
        num_k_heads=2,
        head_dim=head_dim,
        gqa_groups=num_q_heads // 2,
        latent_q_dim=latent_q,
        latent_k_dim=latent_k,
        recurrent_v_dim=recurrent,
        in_out_ch=in_out,
        hidden_size=4,
        total_padding=2,
        sqrt_head_dim=head_dim**0.5,
        prefix="layer",
        config=SimpleNamespace(attention_bias=False, clamp_temp=False),
        rotary_emb=rope,
        q_proj=_Call(
            lambda hidden: hidden[..., :1]
            + torch.linspace(0, 1, latent_q, dtype=hidden.dtype)
        ),
        k_proj=_Call(
            lambda hidden: hidden[..., :1]
            + torch.linspace(1, 2, latent_k, dtype=hidden.dtype)
        ),
        v_proj_current=_Call(
            lambda hidden: hidden[..., :1]
            + torch.linspace(2, 3, recurrent, dtype=hidden.dtype)
        ),
        v_proj_delayed=_Call(
            lambda hidden: hidden[..., :1]
            + torch.linspace(3, 4, recurrent, dtype=hidden.dtype)
        ),
        conv_qk_depthwise=conv(),
        conv_qk_grouped=conv(),
        _zk_grouped_weight=torch.ones(
            num_q_heads + 2, 2 * head_dim, head_dim, dtype=torch.bfloat16
        ),
        temp=torch.ones(2, dtype=torch.float32),
    )
    stride = in_out * 2 + 8
    backing = torch.zeros(4, stride, dtype=state_dtype)
    conv_state = torch.as_strided(backing, (4, in_out, 2), (stride, 2, 1))
    cca.kv_cache = (
        conv_state,
        torch.zeros(4, recurrent, dtype=state_dtype),
    )
    cca.kv_cache[1][1].fill_(3)
    cca._conv_qk_decode = lambda value: value[..., -1:]
    return cca


def _metadata(decodes, prefills, indices, prefill_has_initial=False):
    if decodes is None:
        return None
    metadata = _Metadata()
    metadata.__dict__.update(
        num_prefills=int(prefills > 0),
        num_decode_tokens=decodes,
        num_prefill_tokens=prefills,
        state_indices_tensor_d=indices,
        state_indices_tensor_p=(
            torch.tensor([2], dtype=torch.int32) if prefills else None
        ),
        has_initial_states_p=(
            torch.tensor([prefill_has_initial]) if prefills else None
        ),
        query_start_loc_p=(
            torch.tensor([0, prefills], dtype=torch.int32) if prefills else None
        ),
    )
    return metadata


def _run(cca, decodes, prefills, indices=None, prefill_has_initial=False):
    tokens = 2 if decodes is None else decodes + prefills
    metadata = _metadata(
        decodes,
        prefills,
        indices,
        prefill_has_initial=prefill_has_initial,
    )
    _RUNTIME.context = SimpleNamespace(
        attn_metadata=None if metadata is None else {cca.prefix: metadata}
    )
    qkv = _Call(
        lambda hidden, output, positions: cca.forward_cuda(hidden, output, positions)
    )
    attention = _Attention()
    attention.__dict__.update(
        qkv_proj=qkv,
        q_dim=cca.latent_q_dim,
        k_dim=cca.latent_k_dim,
        v_dim=cca.latent_k_dim,
        qkv_dim=cca.in_out_ch + cca.latent_k_dim,
        attn=_Call(lambda query, key, value: query),
        o_proj=_Call(lambda value: value),
    )
    positions = torch.arange(tokens, dtype=torch.int64)
    attention.forward(torch.ones(tokens, 4, dtype=torch.bfloat16), positions)
    return qkv.calls[0][0][1], positions


def test_reference_env_gate_and_independent_kernel_imports():
    source = _source(CCA_PATH)
    assert 'os.environ.get("VLLM_USE_ZK_CCA", "0")' in source
    assert "if _USE_ZK_CCA:" in source
    assert "_ZK_CCA_PREFILL_IMPORTED" in source
    assert "_ZK_CCA_DECODE_IMPORTED" in source
    assert "VLLM_CCA_ZK_DECODE" not in source


@pytest.mark.parametrize("num_q_heads", [8, 16])
def test_pure_decode_uses_fused_rope_once(num_q_heads):
    cca = _make_cca(num_q_heads)
    _RUNTIME.zk = _Call(_zk_op)
    output, _ = _run(cca, 2, 0, torch.tensor([1, PAD], dtype=torch.int32))
    args, kwargs = _RUNTIME.zk.calls[0]
    assert (args[11], args[12], args[13]) == (num_q_heads, 128, num_q_heads // 2)
    assert args[0].shape == (2, cca.in_out_ch, 1)
    assert kwargs == {}
    assert cca.rotary_emb.calls == []
    assert (output[0].count_nonzero() > 0) and (output[1].count_nonzero() == 0)
    assert (cca.kv_cache[0][1].count_nonzero() > 0) and (
        cca.kv_cache[0][0].count_nonzero() == 0
    )


def test_mismatched_decode_state_dtype_falls_back():
    cca = _make_cca(state_dtype=torch.float32)
    _RUNTIME.zk = _Call(_zk_op)
    output, positions = _run(cca, 1, 0, torch.tensor([1], dtype=torch.int32))
    assert _RUNTIME.zk.calls == []
    assert len(cca.rotary_emb.calls) == 1
    assert torch.equal(cca.rotary_emb.calls[0][0][0], positions)
    assert output.count_nonzero() > 0


def test_decode_kernel_decline_falls_back_before_state_mutation():
    cca = _make_cca()
    _RUNTIME.zk = _Call(_raise_zk_fallback)
    output, positions = _run(cca, 1, 0, torch.tensor([1], dtype=torch.int32))
    assert len(_RUNTIME.zk.calls) == 1
    assert len(cca.rotary_emb.calls) == 1
    assert torch.equal(cca.rotary_emb.calls[0][0][0], positions)
    assert output.count_nonzero() > 0


@pytest.mark.parametrize("has_initial", [False, True])
def test_pure_prefill_fuses_rope_and_shifts_delayed_values(has_initial):
    cca = _make_cca()
    cca.kv_cache[1][2].fill_(7)
    initial_state = cca.kv_cache[1][2].clone()
    _RUNTIME.zk = _Call(_zk_op)
    _RUNTIME.zk_prefill = _Call(_zk_prefill)
    output, _ = _run(cca, 0, 2, prefill_has_initial=has_initial)
    assert _RUNTIME.zk.calls == []
    assert len(_RUNTIME.zk_prefill.calls) == 1
    args, kwargs = _RUNTIME.zk_prefill.calls[0]
    assert args[0].shape == (2, cca.recurrent_v_dim)
    assert args[2] is cca.kv_cache[1]
    assert kwargs["fuse_rope"] is True
    assert len(cca.v_proj_delayed.calls) == 1
    assert torch.equal(cca.kv_cache[1][2], args[0][-1])
    delayed_value = output[:, cca.in_out_ch + cca.recurrent_v_dim :]
    if has_initial:
        assert torch.equal(delayed_value[0], initial_state)
    else:
        assert delayed_value[0].count_nonzero() == 0
    assert torch.equal(delayed_value[1], args[0][0])
    assert cca.rotary_emb.calls == []
    assert output.count_nonzero() > 0


def test_mismatched_prefill_recurrent_dtype_falls_back():
    cca = _make_cca(state_dtype=torch.float32)
    _RUNTIME.zk = None
    _RUNTIME.zk_prefill = _Call(_zk_prefill)
    output, positions = _run(cca, 0, 2)
    assert _RUNTIME.zk_prefill.calls == []
    assert len(cca.rotary_emb.calls) == 1
    assert torch.equal(cca.rotary_emb.calls[0][0][0], positions)
    assert output.count_nonzero() > 0


def test_prefill_kernel_error_falls_back():
    cca = _make_cca()
    _RUNTIME.zk = None
    _RUNTIME.zk_prefill = _Call(_raise_prefill_error)
    output, positions = _run(cca, 0, 2)
    assert len(_RUNTIME.zk_prefill.calls) == 1
    assert len(cca.rotary_emb.calls) == 1
    assert torch.equal(cca.rotary_emb.calls[0][0][0], positions)
    assert output.count_nonzero() > 0


def test_mixed_batch_uses_both_kernels_and_ropes_prefill_once():
    cca = _make_cca()
    _RUNTIME.zk = _Call(_zk_op)
    _RUNTIME.zk_prefill = _Call(_zk_prefill)
    output, positions = _run(cca, 1, 2, torch.tensor([PAD], dtype=torch.int32))
    assert len(_RUNTIME.zk.calls) == 1
    assert len(_RUNTIME.zk_prefill.calls) == 1
    assert _RUNTIME.zk_prefill.calls[0][1]["fuse_rope"] is False
    assert len(cca.rotary_emb.calls) == 1
    assert torch.equal(cca.rotary_emb.calls[0][0][0], positions[1:])
    assert output[0].count_nonzero() == 0
    assert output[1:].count_nonzero() > 0


@pytest.mark.parametrize(
    ("decode_available", "prefill_available"),
    [(True, False), (False, True)],
)
def test_mixed_batch_checks_kernel_availability_independently(
    decode_available, prefill_available
):
    cca = _make_cca()
    _RUNTIME.zk = _Call(_zk_op) if decode_available else None
    _RUNTIME.zk_prefill = _Call(_zk_prefill) if prefill_available else None
    output, _ = _run(cca, 1, 2, torch.tensor([1], dtype=torch.int32))
    if decode_available:
        assert len(_RUNTIME.zk.calls) == 1
    if prefill_available:
        assert len(_RUNTIME.zk_prefill.calls) == 1
    assert output.count_nonzero() > 0


def test_unavailable_kernels_fall_back_and_apply_rope():
    cca = _make_cca()
    _RUNTIME.zk = None
    _RUNTIME.zk_prefill = None
    output, positions = _run(cca, 1, 2, torch.tensor([1], dtype=torch.int32))
    assert len(cca.rotary_emb.calls) == 2
    assert torch.equal(cca.rotary_emb.calls[0][0][0], positions[:1])
    assert torch.equal(cca.rotary_emb.calls[1][0][0], positions[1:])
    assert output.count_nonzero() > 0


def test_no_cache_applies_rope_once_inside_cca():
    cca = _make_cca()
    _RUNTIME.zk = None
    _RUNTIME.zk_prefill = None
    output, positions = _run(cca, None, 0)
    assert len(cca.rotary_emb.calls) == 1
    assert torch.equal(cca.rotary_emb.calls[0][0][0], positions)
    assert output.count_nonzero() > 0
