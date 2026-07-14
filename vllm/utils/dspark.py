# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os


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
