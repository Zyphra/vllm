# SPDX-License-Identifier: Apache-2.0
"""Convert legacy alternating-layer SMoE checkpoints to Zaya layout."""

import argparse
import json
import re
import shutil
from contextlib import ExitStack
from pathlib import Path
from typing import Any

_LAYER = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
_EXPERT = re.compile(
    r"^smoe_block\.experts\.local_experts\.(\d+)\."
    r"(linear_fc1|linear_fc2)\.weight$"
)


def _consistent_layer_value(
    config: dict[str, Any],
    key: str,
    layer_indices: range,
) -> int:
    """Read one scalar value from a scalar or source-layer-sized list."""
    value = config[key]
    if not isinstance(value, list):
        return int(value)

    source_layer_count = len(config["smoe_layers"])
    if len(value) < source_layer_count:
        raise ValueError(
            f"{key} must be scalar or contain at least {source_layer_count} "
            f"source-layer entries, got {len(value)}"
        )
    selected = {int(value[index]) for index in layer_indices}
    if len(selected) != 1:
        raise ValueError(f"converted layers must have one consistent {key}: {selected}")
    return selected.pop()


def convert_config(config: dict[str, Any]) -> dict[str, Any]:
    """Translate a 2N-layer attention/MoE config into N Zaya blocks."""
    layers = config["smoe_layers"]
    if len(layers) % 2 or any(
        layers[i] != "a" or not isinstance(layers[i + 1], int)
        for i in range(0, len(layers), 2)
    ):
        raise ValueError("smoe_layers must alternate attention and MoE layers")
    experts = {layers[i] for i in range(1, len(layers), 2)}
    if len(experts) != 1:
        raise ValueError("all converted MoE layers must have the same expert count")

    attention_layers = range(0, len(layers), 2)
    moe_layers = range(1, len(layers), 2)
    q_heads = _consistent_layer_value(config, "cca_num_q_heads", attention_layers)
    ffn_size = _consistent_layer_value(config, "ffn_hidden_size_list", moe_layers)
    if ffn_size % 2:
        raise ValueError("all MoE layers must have one even fused gate/up size")
    router_size = _consistent_layer_value(config, "smoe_mlp_expansion", moe_layers)

    kv_heads_value = config.get("num_key_value_heads")
    if kv_heads_value is None:
        kv_heads = _consistent_layer_value(
            config, "num_query_groups_list", attention_layers
        )
    elif isinstance(kv_heads_value, list):
        kv_heads = _consistent_layer_value(
            config, "num_key_value_heads", attention_layers
        )
    else:
        kv_heads = int(kv_heads_value)
    if "num_query_groups_list" in config:
        listed_kv_heads = _consistent_layer_value(
            config, "num_query_groups_list", attention_layers
        )
        if listed_kv_heads != kv_heads:
            raise ValueError(
                "num_key_value_heads disagrees with attention-layer "
                f"num_query_groups_list: {kv_heads} != {listed_kv_heads}"
            )

    hidden_size = int(config["hidden_size"])
    source_attention_heads = int(config["num_attention_heads"])
    if hidden_size % source_attention_heads:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
    head_dim = hidden_size // source_attention_heads

    swa_values = config.get("swa_layers", [0] * len(layers))
    if not isinstance(swa_values, list) or len(swa_values) != len(layers):
        raise ValueError(
            f"swa_layers must contain {len(layers)} source-layer entries"
        )
    swa = [int(swa_values[index]) for index in attention_layers]
    layer_types = ["hybrid_sliding" if window else "hybrid" for window in swa]
    partial_rotary_factor = config.get("rope_pct", 0.5)
    return {
        "architectures": ["ZayaForCausalLM"],
        "model_type": "zaya",
        "dtype": config.get("dtype", "bfloat16"),
        "use_cache": config.get("use_cache", True),
        "attention_bias": config.get("attention_bias", False),
        "attention_dropout": config.get("attention_dropout", 0.0),
        "lm_head_bias": config.get("lm_head_bias", False),
        "vocab_size": config["vocab_size"],
        "hidden_size": hidden_size,
        "num_hidden_layers": len(layers) // 2,
        "num_experts": experts.pop(),
        "num_attention_heads": q_heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "hidden_act": "silu",
        "moe_intermediate_size": ffn_size // 2,
        "router_hidden_size": router_size,
        "num_experts_per_tok": config["moe_router_topk"],
        "max_position_embeddings": config["max_position_embeddings"],
        "initializer_range": config.get("initializer_range", 0.02),
        "rms_norm_eps": config["norm_epsilon"],
        "pad_token_id": config.get("pad_token_id", 0),
        "bos_token_id": config.get("bos_token_id", 2),
        "eos_token_id": config.get("eos_token_id", 106),
        "tie_word_embeddings": True,
        "output_router_logits": False,
        "layer_types": layer_types,
        "sliding_window": max(swa, default=0) or None,
        "partial_rotary_factor": partial_rotary_factor,
        "rope_parameters": {
            "hybrid": {
                "rope_type": "default",
                "rope_theta": config["rotary_base"],
                "partial_rotary_factor": partial_rotary_factor,
            },
            "hybrid_sliding": {
                "rope_type": "default",
                "rope_theta": config.get(
                    "swa_rotary_base", config["rotary_base"]
                ),
                "partial_rotary_factor": partial_rotary_factor,
            },
        },
        "cca_time0": config.get("cca_time0", 2),
        "cca_time1": config.get("cca_time1", 2),
        "transformers_version": config.get("transformers_version"),
    }


def map_weight(name: str, num_blocks: int) -> str | None:
    """Return the Zaya key, or None for expert weights that require stacking."""
    global_names = {
        "model.embed_tokens.weight": "model.embed_tokens.weight",
        "model.final_norm.weight": "model.norm.weight",
    }
    if name in global_names:
        return global_names[name]
    if name.startswith("model.res_scale."):
        suffix = name.removeprefix("model.res_scale.")
        return f"model.layers.{num_blocks - 1}.post_mlp_residual_scale.{suffix}"

    match = _LAYER.match(name)
    if not match:
        raise ValueError(f"unsupported checkpoint weight: {name}")
    source_layer, suffix = int(match.group(1)), match.group(2)
    block, is_moe = divmod(source_layer, 2)

    if not is_moe:
        if suffix.startswith("res_scale."):
            scale = suffix.removeprefix("res_scale.")
            if block == 0:
                if scale.startswith("residual_"):
                    raise ValueError("first attention layer has residual-scale weights")
                return f"model.input_{scale}"
            return f"model.layers.{block - 1}.post_mlp_residual_scale.{scale}"
        if suffix == "input_norm.weight":
            return f"model.layers.{block}.input_layernorm.weight"
        replacements = (
            ("self_attn.qkv.linear_q.", "self_attn.qkv_proj.q_proj."),
            ("self_attn.qkv.linear_k.", "self_attn.qkv_proj.k_proj."),
            ("self_attn.qkv.val_proj1.", "self_attn.qkv_proj.v_proj_current."),
            ("self_attn.qkv.val_proj2.", "self_attn.qkv_proj.v_proj_delayed."),
            (
                "self_attn.qkv.conv_qk.0.",
                "self_attn.qkv_proj.conv_qk_depthwise.",
            ),
            (
                "self_attn.qkv.conv_qk.1.",
                "self_attn.qkv_proj.conv_qk_grouped.",
            ),
            ("self_attn.qkv.temp", "self_attn.qk_norm.temp"),
        )
        for source, target in replacements:
            if suffix.startswith(source):
                suffix = target + suffix[len(source) :]
                break
        return f"model.layers.{block}.{suffix}"

    if suffix.startswith("res_scale."):
        suffix = "post_attention_residual_scale." + suffix.removeprefix("res_scale.")
    elif suffix == "input_norm.weight":
        suffix = "post_attention_layernorm.weight"
    elif _EXPERT.match(suffix):
        return None
    elif suffix.startswith("smoe_block.router."):
        suffix = "mlp.gate." + suffix.removeprefix("smoe_block.router.")
        suffix = suffix.replace("rmsnorm_eda.", "router_mlp.norm.")
        suffix = suffix.replace("router_mlp.0.", "router_mlp.fc1.")
        suffix = suffix.replace("router_mlp.2.", "router_mlp.fc2.")
        suffix = suffix.replace("router_mlp.4.", "router_mlp.out_proj.")
    else:
        raise ValueError(f"unsupported MoE weight: {name}")
    return f"model.layers.{block}.{suffix}"


def build_plan(
    weight_map: dict[str, str], config: dict[str, Any]
) -> tuple[dict[str, str], dict[tuple[int, str], list[str]]]:
    """Validate a bijective non-expert map and complete expert groups."""
    num_blocks = len(config["smoe_layers"]) // 2
    mapped: dict[str, str] = {}
    experts: dict[tuple[int, str], list[tuple[int, str]]] = {}
    targets: set[str] = set()

    for source in weight_map:
        target = map_weight(source, num_blocks)
        if target is not None:
            if target in targets:
                raise ValueError(f"multiple weights map to {target}")
            targets.add(target)
            mapped[source] = target
            continue
        match = _LAYER.match(source)
        assert match is not None
        block = int(match.group(1)) // 2
        expert_match = _EXPERT.match(match.group(2))
        assert expert_match is not None
        expert_id = int(expert_match.group(1))
        projection = expert_match.group(2)
        experts.setdefault((block, projection), []).append((expert_id, source))

    expected_experts = config["smoe_layers"][1]
    ordered: dict[tuple[int, str], list[str]] = {}
    for block in range(num_blocks):
        for projection in ("linear_fc1", "linear_fc2"):
            group = sorted(experts.get((block, projection), []))
            if [expert_id for expert_id, _ in group] != list(range(expected_experts)):
                raise ValueError(
                    f"block {block} {projection} does not contain experts "
                    f"0..{expected_experts - 1}"
                )
            ordered[(block, projection)] = [source for _, source in group]
    return mapped, ordered


def _convert(source: Path, destination: Path) -> None:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"destination must be empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    config = json.loads((source / "config.json").read_text())
    index = json.loads((source / "model.safetensors.index.json").read_text())
    weight_map: dict[str, str] = index["weight_map"]
    mapped, expert_groups = build_plan(weight_map, config)
    num_blocks = len(config["smoe_layers"]) // 2
    output_map: dict[str, str] = {}
    total_size = 0

    def write_shard(
        filename: str,
        direct: dict[str, str],
        expert: dict[str, list[str]],
    ) -> None:
        nonlocal total_size
        source_files = {weight_map[name] for name in direct} | {
            weight_map[name] for names in expert.values() for name in names
        }
        with ExitStack() as stack:
            handles = {
                filename: stack.enter_context(
                    safe_open(source / filename, framework="pt", device="cpu")
                )
                for filename in source_files
            }

            def load(name: str) -> torch.Tensor:
                return handles[weight_map[name]].get_tensor(name)

            tensors = {
                target: load(source_name) for source_name, target in direct.items()
            }
            tensors.update(
                {
                    target: torch.stack([load(name) for name in names])
                    for target, names in expert.items()
                }
            )
            save_file(tensors, destination / filename, metadata={"format": "pt"})
        for target, tensor in tensors.items():
            output_map[target] = filename
            total_size += tensor.numel() * tensor.element_size()

    shard_count = num_blocks + 1
    for block in range(num_blocks):
        filename = f"model-{block + 1:05d}-of-{shard_count:05d}.safetensors"
        prefix = f"model.layers.{block}."
        direct = {
            source_name: target
            for source_name, target in mapped.items()
            if target.startswith(prefix)
        }
        expert = {
            f"{prefix}mlp.experts.gate_up_proj": expert_groups[(block, "linear_fc1")],
            f"{prefix}mlp.experts.down_proj": expert_groups[(block, "linear_fc2")],
        }
        write_shard(filename, direct, expert)

    filename = f"model-{shard_count:05d}-of-{shard_count:05d}.safetensors"
    direct = {
        source_name: target
        for source_name, target in mapped.items()
        if not target.startswith("model.layers.")
    }
    write_shard(filename, direct, {})

    converted_config = convert_config(config)
    (destination / "config.json").write_text(
        json.dumps(converted_config, indent=2, sort_keys=True) + "\n"
    )
    (destination / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": total_size}, "weight_map": output_map},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    excluded = {
        "config.json",
        "model.safetensors.index.json",
        *set(weight_map.values()),
    }
    for path in source.iterdir():
        if path.is_file() and path.name not in excluded:
            shutil.copy2(path, destination / path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path, nargs="?")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate config and the complete key mapping without reading tensors",
    )
    args = parser.parse_args()

    config = json.loads((args.source / "config.json").read_text())
    index = json.loads((args.source / "model.safetensors.index.json").read_text())
    mapped, expert_groups = build_plan(index["weight_map"], config)
    convert_config(config)
    if args.validate_only:
        target_count = len(mapped) + len(expert_groups)
        print(
            f"validated {len(index['weight_map'])} source weights "
            f"to {target_count} Zaya tensors"
        )
        return
    if args.destination is None:
        parser.error("destination is required unless --validate-only is used")
    _convert(args.source, args.destination)


if __name__ == "__main__":
    main()
