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
ENVS_PATH = ROOT / "vllm/envs.py"
PAD = -1
_RUNTIME = SimpleNamespace(context=None, zk=None, zk_prefill=None)
_Metadata = type("_Metadata", (), {})


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
        "get_forward_context": lambda: _RUNTIME.context,
        "_get_zk_cca_ops": lambda: (_RUNTIME.zk, _RUNTIME.zk_prefill),
        "logger": SimpleNamespace(info_once=lambda *args: None),
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


def _zk_op(*args, out):
    first, state, indices = args[0], args[3], args[4]
    out.copy_(-first)
    for row, index in enumerate(indices.tolist()):
        if index == PAD:
            out[row].zero_()
        else:
            state[index, :, 0] = state[index, :, 1]
            state[index, :, 1] = first[row].to(state.dtype)
    return out


def _zk_prefill(*args, **kwargs):
    return torch.empty(0), -args[1]


def _make_cca(num_q_heads=8, enabled=True, state_dtype=torch.float32):
    cca = _CCA()
    head_dim = 128
    latent_q, latent_k = num_q_heads * head_dim, 2 * head_dim
    recurrent, in_out = latent_k // 2, latent_q + latent_k
    rope = _Call(
        lambda positions, query, key: (-query, -key),
        rotary_dim=64,
        cache=torch.zeros(32, 64, dtype=torch.bfloat16),
    )
    rope._match_cos_sin_cache_dtype = lambda reference: rope.cache.to(reference.dtype)
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
        _zk_cca_decode_enabled=enabled,
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


def _metadata(decodes, prefills, indices):
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
        has_initial_states_p=torch.tensor([False]) if prefills else None,
        query_start_loc_p=(
            torch.tensor([0, prefills], dtype=torch.int32) if prefills else None
        ),
    )
    return metadata


def _run(cca, decodes, prefills, indices=None):
    tokens = 2 if decodes is None else decodes + prefills
    metadata = _metadata(decodes, prefills, indices)
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
        _cca_returns_rotated_qk=cca._zk_cca_decode_enabled,
        attn=_Call(lambda query, key, value: query),
        o_proj=_Call(lambda value: value),
    )
    if not cca._zk_cca_decode_enabled:
        attention.rotary_emb = cca.rotary_emb
    positions = torch.arange(tokens, dtype=torch.int64)
    attention.forward(torch.ones(tokens, 4, dtype=torch.bfloat16), positions)
    return qkv.calls[0][0][1], positions


def test_lazy_default_off_and_compile_factor_wiring():
    source = _source(CCA_PATH)
    assert "from zyphra_kernels.cca" not in source.split("def _get_zk_cca_ops")[0]
    assert "envs.VLLM_CCA_ZK_DECODE" in source
    assert '"VLLM_CCA_ZK_DECODE": lambda: bool' in _source(ENVS_PATH)


@pytest.mark.parametrize("num_q_heads", [8, 16])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_pure_decode_uses_caller_output_and_fused_rope_once(num_q_heads, state_dtype):
    cca = _make_cca(num_q_heads, state_dtype=state_dtype)
    _RUNTIME.zk = _Call(_zk_op)
    output, _ = _run(cca, 2, 0, torch.tensor([1, PAD], dtype=torch.int32))
    args, kwargs = _RUNTIME.zk.calls[0]
    assert (args[11], args[12], args[13]) == (num_q_heads, 128, num_q_heads // 2)
    assert (kwargs["out"].data_ptr(), kwargs["out"].is_contiguous()) == (
        output.data_ptr(),
        False,
    )
    assert cca.rotary_emb.calls == []
    assert (output[0].count_nonzero() > 0) and (output[1].count_nonzero() == 0)
    assert (cca.kv_cache[0][1].count_nonzero() > 0) and (
        cca.kv_cache[0][0].count_nonzero() == 0
    )


@pytest.mark.parametrize(
    ("decodes", "prefills", "indices"),
    [(0, 2, None), (1, 2, torch.tensor([PAD], dtype=torch.int32))],
)
def test_prefill_and_mixed_fallback_apply_rope_once(decodes, prefills, indices):
    cca = _make_cca()
    _RUNTIME.zk = _Call(_zk_op)
    _RUNTIME.zk_prefill = _Call(_zk_prefill)
    output, positions = _run(cca, decodes, prefills, indices)
    assert _RUNTIME.zk.calls == []
    assert len(_RUNTIME.zk_prefill.calls) == (decodes == 0)
    assert len(cca.rotary_emb.calls) == 1
    assert torch.equal(cca.rotary_emb.calls[0][0][0], positions)
    if decodes:
        assert output[0].count_nonzero() == 0
    assert output[decodes:].count_nonzero() > 0


@pytest.mark.parametrize("enabled", [False, True])
def test_no_cache_applies_rope_once_inside_or_outside_cca(enabled):
    cca = _make_cca(enabled=enabled)
    _RUNTIME.zk = None
    output, positions = _run(cca, None, 0)
    assert len(cca.rotary_emb.calls) == 1
    assert torch.equal(cca.rotary_emb.calls[0][0][0], positions)
    assert output.count_nonzero() > 0
