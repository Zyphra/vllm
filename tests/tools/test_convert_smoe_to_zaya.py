# SPDX-License-Identifier: Apache-2.0

import pytest

from tools.model_converters.convert_smoe_to_zaya import (
    build_plan,
    convert_config,
    map_weight,
)


def _config() -> dict:
    return {
        "smoe_layers": ["a", 2, "a", 2],
        "cca_num_q_heads": [8, 0, 8, 0],
        "ffn_hidden_size_list": [0, 4096, 0, 4096],
        "smoe_mlp_expansion": [0, 256, 0, 256],
        "swa_layers": [4096, 0, 0, 0],
        "vocab_size": 1024,
        "hidden_size": 2048,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "moe_router_topk": 1,
        "max_position_embeddings": 8192,
        "norm_epsilon": 1e-5,
        "rotary_base": 5_000_000,
        "swa_rotary_base": 10_000,
    }


def test_convert_config_collapses_attention_moe_pairs():
    converted = convert_config(_config())
    assert converted["architectures"] == ["ZayaForCausalLM"]
    assert converted["model_type"] == "zaya"
    assert converted["num_hidden_layers"] == 2
    assert converted["num_experts"] == 2
    assert converted["num_attention_heads"] == 8
    assert converted["head_dim"] == 128
    assert converted["moe_intermediate_size"] == 2048
    assert converted["layer_types"] == ["hybrid_sliding", "hybrid"]


@pytest.mark.parametrize(
    ("source", "target"),
    [
        (
            "model.layers.0.res_scale.hidden_states_scale",
            "model.input_hidden_states_scale",
        ),
        (
            "model.layers.0.input_norm.weight",
            "model.layers.0.input_layernorm.weight",
        ),
        (
            "model.layers.0.self_attn.qkv.linear_q.weight",
            "model.layers.0.self_attn.qkv_proj.q_proj.weight",
        ),
        (
            "model.layers.1.res_scale.residual_scale",
            "model.layers.0.post_attention_residual_scale.residual_scale",
        ),
        (
            "model.layers.1.input_norm.weight",
            "model.layers.0.post_attention_layernorm.weight",
        ),
        (
            "model.layers.1.smoe_block.router.rmsnorm_eda.weight",
            "model.layers.0.mlp.gate.router_mlp.norm.weight",
        ),
        (
            "model.layers.2.res_scale.residual_bias",
            "model.layers.0.post_mlp_residual_scale.residual_bias",
        ),
        (
            "model.res_scale.hidden_states_bias",
            "model.layers.1.post_mlp_residual_scale.hidden_states_bias",
        ),
        ("model.final_norm.weight", "model.norm.weight"),
    ],
)
def test_map_weight_preserves_combined_block_semantics(source, target):
    assert map_weight(source, num_blocks=2) == target


def test_build_plan_requires_every_expert():
    weights = {
        "model.embed_tokens.weight": "globals.safetensors",
        "model.final_norm.weight": "globals.safetensors",
    }
    for source_layer in (1, 3):
        for expert in range(2):
            for projection in ("linear_fc1", "linear_fc2"):
                weights[
                    f"model.layers.{source_layer}.smoe_block.experts."
                    f"local_experts.{expert}.{projection}.weight"
                ] = f"layer-{source_layer}.safetensors"

    mapped, experts = build_plan(weights, _config())
    assert mapped["model.final_norm.weight"] == "model.norm.weight"
    assert len(experts) == 4

    weights.pop("model.layers.3.smoe_block.experts.local_experts.1.linear_fc2.weight")
    with pytest.raises(ValueError, match="does not contain experts"):
        build_plan(weights, _config())
