# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from typing import Any


def validate_dspark_load_format(
    dspark_markov_enabled: bool,
    load_format: str,
) -> None:
    """Reject random initialization of checkpoint-only DSpark heads."""
    dspark_active = os.environ.get("VLLM_TIDAR_DSPARK", "1") != "0"
    if dspark_markov_enabled and dspark_active and load_format == "dummy":
        raise ValueError(
            "DSpark cannot run with load_format='dummy': dummy loading "
            "random-initializes the checkpoint-only diffusion_output_layer "
            "and diffusion_markov_head weights, and Megatron parameter sync "
            "does not provide those vLLM-only tensors. Use "
            "load_format='auto' (or another checkpoint-backed loader)."
        )


def validate_dspark_target_contract(
    target_model: Any,
    *,
    retained_target_model: Any | None = None,
) -> None:
    """Validate that target verification and DSpark drafting stay separate.

    The shared SMoE transformer is intentionally reused by both passes. Only
    ``diffusion_output_layer`` and ``diffusion_markov_head`` are draft-only.
    This runs once after checkpoint loading, before warmup or graph capture.
    """
    if (
        retained_target_model is not None
        and retained_target_model is not target_model
    ):
        raise RuntimeError(
            "TiDAR AR-verifier contract failed: speculator does not retain "
            "the exact target model object."
        )

    try:
        config = target_model.config
        lm_head_weight = target_model.lm_head.weight
        embedding_weight = target_model.model.embed_tokens.weight
        draft_weight = target_model.diffusion_output_layer.weight
        markov_w1 = target_model.diffusion_markov_head.w1
        markov_w2 = target_model.diffusion_markov_head.w2
    except AttributeError as exc:
        raise RuntimeError(
            "TiDAR AR-verifier contract failed: loaded DSpark target is "
            f"missing a required AR/draft head ({exc})."
        ) from exc

    if not bool(getattr(config, "tie_word_embeddings", False)):
        raise RuntimeError(
            "TiDAR AR-verifier contract failed: the live DSpark target "
            "requires tie_word_embeddings=True."
        )
    if lm_head_weight is not embedding_weight:
        raise RuntimeError(
            "TiDAR AR-verifier contract failed: regular lm_head.weight is "
            "not the same Parameter as model.embed_tokens.weight."
        )

    draft_parameters = {
        "diffusion_output_layer.weight": draft_weight,
        "diffusion_markov_head.w1": markov_w1,
        "diffusion_markov_head.w2": markov_w2,
    }
    for name, parameter in draft_parameters.items():
        if parameter is lm_head_weight:
            raise RuntimeError(
                "TiDAR AR-verifier contract failed: draft-only "
                f"{name} aliases regular lm_head.weight."
            )
        if bool(getattr(parameter, "requires_grad", True)):
            raise RuntimeError(
                "TiDAR AR-verifier contract failed: draft-only "
                f"{name} must be frozen."
            )

    vocab_size = int(config.vocab_size)
    hidden_size = int(config.hidden_size)
    expected_draft_shape = (vocab_size, hidden_size)
    if tuple(draft_weight.shape) != expected_draft_shape:
        raise RuntimeError(
            "TiDAR AR-verifier contract failed: "
            f"diffusion_output_layer.weight shape={tuple(draft_weight.shape)} "
            f"expected={expected_draft_shape}."
        )
    if (
        markov_w1.ndim != 2
        or markov_w2.ndim != 2
        or int(markov_w1.shape[0]) != vocab_size
        or int(markov_w2.shape[1]) != vocab_size
        or int(markov_w1.shape[1]) != int(markov_w2.shape[0])
    ):
        raise RuntimeError(
            "TiDAR AR-verifier contract failed: incompatible Markov head "
            f"shapes w1={tuple(markov_w1.shape)} "
            f"w2={tuple(markov_w2.shape)} vocab_size={vocab_size}."
        )
