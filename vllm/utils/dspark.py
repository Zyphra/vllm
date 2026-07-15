# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
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


def validate_tidar_temperature_contract(
    *,
    draft_temperature: float,
    ar_temperature: float,
    configured_ar_temperature: float | str | None,
) -> None:
    """Require draft, verifier, and exported-score temperatures to match."""

    values: dict[str, float] = {
        "tidar_diff_temperature": float(draft_temperature),
        "tidar_ar_temperature": float(ar_temperature),
    }
    if configured_ar_temperature is None or not str(
        configured_ar_temperature
    ).strip():
        raise ValueError(
            "TiDAR temperature contract failed: "
            "VLLM_TIDAR_AR_TEMPERATURE is required in strict mode."
        )
    try:
        values["VLLM_TIDAR_AR_TEMPERATURE"] = float(
            configured_ar_temperature
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "TiDAR temperature contract failed: "
            "VLLM_TIDAR_AR_TEMPERATURE must be a finite positive float."
        ) from exc

    invalid = {
        name: value
        for name, value in values.items()
        if not math.isfinite(value) or value <= 0
    }
    if invalid:
        raise ValueError(
            "TiDAR temperature contract failed: all temperatures must be "
            f"finite and positive; invalid={invalid}."
        )

    reference = values["VLLM_TIDAR_AR_TEMPERATURE"]
    mismatched = {
        name: value
        for name, value in values.items()
        if not math.isclose(value, reference, rel_tol=1e-6, abs_tol=1e-8)
    }
    if mismatched:
        raise ValueError(
            "TiDAR temperature contract failed: draft, AR verifier, and "
            f"returned-score temperatures must match; values={values}."
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

    compute_logits = getattr(target_model, "compute_logits", None)
    compute_logits_function = getattr(
        compute_logits, "__func__", compute_logits
    )
    compute_logits_code = getattr(compute_logits_function, "__code__", None)
    compute_logits_names = (
        set(compute_logits_code.co_names) if compute_logits_code is not None else set()
    )
    required_compute_names = {"logits_processor", "lm_head"}
    forbidden_compute_names = {
        "diffusion_output_layer",
        "diffusion_markov_head",
    }
    if not required_compute_names.issubset(compute_logits_names) or (
        forbidden_compute_names & compute_logits_names
    ):
        raise RuntimeError(
            "TiDAR AR-verifier contract failed: target compute_logits must "
            "use logits_processor with the regular lm_head and may not read "
            "DSpark/Markov draft heads; "
            f"observed_names={sorted(compute_logits_names)}."
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


def enforce_dspark_target_contract(
    target_model: Any,
    *,
    retained_target_model: Any | None = None,
    dspark_active: bool,
    required: bool,
) -> None:
    """Fail closed when the live TiDAR run requires the DSpark contract."""
    if not required:
        return
    if not dspark_active:
        raise RuntimeError(
            "TiDAR AR-verifier contract failed: "
            "VLLM_TIDAR_REQUIRE_AR_VERIFIER_CONTRACT=1 requires the loaded "
            "target checkpoint to activate the DSpark/Markov draft heads."
        )
    validate_dspark_target_contract(
        target_model,
        retained_target_model=retained_target_model,
    )
