"""Optional exact-geometry fused CCA decode boundary.

This module is deliberately default-off.  It owns only extension loading and
the narrow call wrapper; model semantics and cache ownership remain in
``mamba.cca``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_enabled = os.environ.get("VLLM_CCA_DECODE_FUSED_ENABLED", "0").lower() in (
    "1",
    "true",
    "yes",
)
_fn = None
_mixed_state_fn = None

if _enabled and torch.version.hip is not None:
    try:
        from torch.utils.cpp_extension import load

        source_path = os.environ.get(
            "VLLM_CCA_DECODE_FUSED_KERNEL_PATH",
            str(Path(__file__).resolve().parents[5] / "csrc/rocm/cca_decode_norm_fused.cu"),
        )
        module = load(
            name="zaya_cca_decode_norm_fused_v1",
            sources=[source_path],
            extra_cuda_cflags=["-O3", "-DUSE_ROCM"],
            verbose=False,
        )
        _fn = module.cca_decode_fused
        _mixed_state_fn = module.cca_decode_fused_mixed_state
        logger.info("Loaded fused CCA decode kernel from %s", source_path)
    except Exception:
        logger.exception("Fused CCA decode requested but failed to load")


def available() -> bool:
    return _fn is not None


def requested() -> bool:
    return _enabled


def decode(
    first_input: torch.Tensor,
    dw_weight: torch.Tensor,
    dw_bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    state_indices: torch.Tensor,
    gw_weight: torch.Tensor,
    gw_bias: torch.Tensor | None,
    temp: torch.Tensor,
    num_q_heads: int,
    head_dim: int,
    gqa_groups: int,
    pad_slot_id: int,
    sqrt_head_dim: float,
    clamp_temp: bool,
    out: torch.Tensor,
) -> torch.Tensor:
    if _fn is None:
        raise RuntimeError("fused CCA decode kernel is not loaded")
    return _fn(
        first_input.contiguous(),
        dw_weight.contiguous(),
        None if dw_bias is None else dw_bias.contiguous(),
        conv_states,
        state_indices.to(torch.int64).contiguous(),
        gw_weight.contiguous(),
        None if gw_bias is None else gw_bias.contiguous(),
        None,
        temp.float().contiguous(),
        int(num_q_heads),
        int(head_dim),
        int(gqa_groups),
        int(pad_slot_id),
        float(sqrt_head_dim),
        bool(clamp_temp),
        True,
        out,
    )


def decode_mixed_state(
    first_input: torch.Tensor,
    dw_weight: torch.Tensor,
    dw_bias: torch.Tensor | None,
    conv_states: torch.Tensor,
    state_indices: torch.Tensor,
    gw_weight: torch.Tensor,
    gw_bias: torch.Tensor | None,
    temp: torch.Tensor,
    num_q_heads: int,
    head_dim: int,
    gqa_groups: int,
    pad_slot_id: int,
    sqrt_head_dim: float,
    clamp_temp: bool,
    out: torch.Tensor,
) -> torch.Tensor:
    if _mixed_state_fn is None:
        raise RuntimeError("mixed-state CCA decode kernel is not loaded")
    return _mixed_state_fn(
        first_input.contiguous(),
        dw_weight.contiguous(),
        None if dw_bias is None else dw_bias.contiguous(),
        conv_states,
        state_indices.to(torch.int64).contiguous(),
        gw_weight.contiguous(),
        None if gw_bias is None else gw_bias.contiguous(),
        temp.float().contiguous(),
        int(num_q_heads),
        int(head_dim),
        int(gqa_groups),
        int(pad_slot_id),
        float(sqrt_head_dim),
        bool(clamp_temp),
        out,
    )
