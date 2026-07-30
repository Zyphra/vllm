# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.attention.backend import AttentionBackend

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from vllm import envs
from vllm.config import CacheConfig, ModelConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.attention.backends.cca_attn import CCAAttentionMetadata
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

logger = init_logger(__name__)
@cache
def _get_zk_cca_ops():
    try:
        from zyphra_kernels.cca import (
            cca_decode_fused_rope_auto,
            cca_prefill_norm_fused,
            cca_prefill_norm_fused_available,
        )
        from zyphra_kernels.cca.ops import cca_decode_rope_available
    except ImportError as exc:
        raise RuntimeError("Zyphra CCA kernels were requested but unavailable") from exc
    if not cca_decode_rope_available() or not cca_prefill_norm_fused_available():
        raise RuntimeError("Zyphra CCA kernels were requested but unavailable")
    return cca_decode_fused_rope_auto, cca_prefill_norm_fused


@CustomOp.register("cca")
class CCA(MambaBase, CustomOp):
    def __init__(
        self,
        config,
        cca_num_k_heads: int = 2,
        cca_num_q_heads: int = 8,
        hidden_size: int | None = None,
        head_dim: int = 128,
        cca_time0: int = 2,
        cca_time1: int = 2,
        layer_number: int = 0,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        rotary_emb: nn.Module | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.model_config = model_config
        self.cache_config = cache_config
        self.layer_number = layer_number
        self.prefix = prefix

        # Use the model's true hidden size unless explicitly overridden.
        # (In Megatron this is the lane's hidden_size_in.)
        self.hidden_size = int(hidden_size or config.hidden_size)

        self.cca_time0 = cca_time0
        self.cca_time1 = cca_time1
        self.padding0 = cca_time0 - 1
        self.padding1 = cca_time1 - 1
        self.total_padding = self.padding0 + self.padding1

        self.num_k_heads = int(cca_num_k_heads)
        self.num_q_heads = int(cca_num_q_heads)

        # Geometry
        self.head_dim = int(head_dim)
        self.latent_k_dim = self.num_k_heads * self.head_dim
        self.latent_q_dim = self.num_q_heads * self.head_dim
        self.recurrent_v_dim = self.latent_k_dim // 2
        self.sqrt_head_dim = np.sqrt(self.head_dim)
        self.gqa_groups = self.num_q_heads // self.num_k_heads
        assert self.num_q_heads % self.num_k_heads == 0, (
            "q_heads must be a multiple of k_heads"
        )
        assert (self.latent_k_dim + self.latent_q_dim) == (
            self.num_k_heads + self.num_q_heads
        ) * self.head_dim
        self._zk_cca_decode_enabled = envs.VLLM_CCA_ZK_DECODE
        if self._zk_cca_decode_enabled:
            if (
                self.num_q_heads not in (8, 16)
                or self.num_k_heads != 2
                or self.head_dim != 128
                or self.gqa_groups not in (4, 8)
                or self.cca_time0 != 2
                or self.cca_time1 != 2
            ):
                raise RuntimeError(
                    "Zyphra CCA decode requires packed Q8/K2 or Q16/K2 "
                    "D128 with K0=K1=2"
                )
            if rotary_emb is None:
                raise RuntimeError(
                    "Zyphra CCA decode requires a per-layer RoPE module"
                )
            self.rotary_emb = rotary_emb
            _get_zk_cca_ops()

        # Projections
        self.q_proj = ReplicatedLinear(
            self.hidden_size,
            self.latent_q_dim,
            bias=self.config.attention_bias,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.q_proj",
        )
        self.k_proj = ReplicatedLinear(
            self.hidden_size,
            self.latent_k_dim,
            bias=self.config.attention_bias,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.k_proj",
        )
        self.v_proj_current = ReplicatedLinear(
            self.hidden_size,
            self.latent_k_dim // 2,
            bias=self.config.attention_bias,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.v_proj_current",
        )
        self.v_proj_delayed = ReplicatedLinear(
            self.hidden_size,
            self.latent_k_dim // 2,
            bias=self.config.attention_bias,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.v_proj_delayed",
        )

        # Depthwise + grouped conv along sequence (exactly like Megatron)
        in_out_ch = self.latent_k_dim + self.latent_q_dim
        self.in_out_ch = in_out_ch
        self.conv_qk_depthwise = nn.Conv1d(
            in_channels=in_out_ch,
            out_channels=in_out_ch,
            kernel_size=self.cca_time0,
            groups=in_out_ch,
            padding=0,
            stride=1,
        )
        self.conv_qk_grouped = nn.Conv1d(
            in_channels=in_out_ch,
            out_channels=in_out_ch,
            kernel_size=self.cca_time1,
            groups=(self.num_k_heads + self.num_q_heads),
            padding=0,
            stride=1,
        )
        if self._zk_cca_decode_enabled:
            # The decode ABI consumes this transposed, contiguous layout.
            grouped_weight = self.conv_qk_grouped.weight
            self.register_buffer(
                "_zk_grouped_weight",
                grouped_weight.new_empty(
                    self.num_q_heads + self.num_k_heads,
                    self.head_dim * self.cca_time1,
                    self.head_dim,
                ),
                persistent=False,
            )
            set_weight_attrs(
                grouped_weight,
                {"post_weight_update": self._refresh_zk_grouped_weight},
            )

        # Per-k head temperature (Megatron: shape [num_k_heads])
        self.temp = nn.Parameter(torch.zeros(self.num_k_heads, dtype=torch.float32))

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        self.kv_cache = (torch.tensor([]), torch.tensor([]))

    @torch.no_grad()
    def _refresh_zk_grouped_weight(self) -> None:
        groups = self.num_q_heads + self.num_k_heads
        source = self.conv_qk_grouped.weight.view(
            groups, self.head_dim, self.head_dim, self.cca_time1
        )
        self._zk_grouped_weight.view(
            groups, self.head_dim, self.cca_time1, self.head_dim
        ).copy_(source.permute(0, 2, 3, 1))

    def _get_zk_rope_cache(self, reference: torch.Tensor) -> torch.Tensor:
        cache = self.rotary_emb._match_cos_sin_cache_dtype(reference)
        if not cache.is_contiguous():
            raise RuntimeError("Zyphra CCA decode requires a contiguous RoPE cache")
        return cache

    def _apply_rope_to_output(
        self,
        position_ids: torch.Tensor,
        output: torch.Tensor,
        num_tokens: int,
    ) -> None:
        positions = position_ids.reshape(-1)[:num_tokens]
        if positions.numel() != num_tokens:
            raise RuntimeError("CCA position metadata does not cover all tokens")
        query = output[:num_tokens, : self.latent_q_dim]
        key = output[
            :num_tokens,
            self.latent_q_dim : self.latent_q_dim + self.latent_k_dim,
        ]
        rotated_query, rotated_key = self.rotary_emb(positions, query, key)
        query.copy_(rotated_query)
        key.copy_(rotated_key)

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        position_ids: torch.Tensor,
    ):
        self._forward_no_cache(hidden_states, output, position_ids)

    @property
    def returns_rotated_qk(self) -> bool:
        return self._zk_cca_decode_enabled

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        position_ids: torch.Tensor,
    ):
        torch.ops.vllm.cca(
            hidden_states,
            output,
            position_ids,
            self.prefix,
        )

    def _forward_no_cache(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> None:
        """Project an uncached contiguous token sequence into q/k/v."""
        num_tokens = hidden_states.shape[0]
        hs = hidden_states.unsqueeze(1)  # [S, 1, H]

        q = self.q_proj(hs)
        k = self.k_proj(hs)
        qk_packed0 = torch.cat([q, k], dim=-1)
        del q
        del k

        query_pre = qk_packed0[..., :self.latent_q_dim].view(
            *qk_packed0.shape[:2], self.num_q_heads, self.head_dim)
        key_base = qk_packed0[..., self.latent_q_dim:].view(
            *qk_packed0.shape[:2], self.num_k_heads, self.head_dim)

        qk_packed2 = F.pad(qk_packed0.permute(1, 2, 0),
                           (self.total_padding, 0))
        qk_packed3 = self.conv_qk_grouped(
            self.conv_qk_depthwise(qk_packed2)).permute(2, 0, 1)

        query = qk_packed3[..., :self.latent_q_dim].view(
            *qk_packed3.shape[:2], self.num_q_heads, self.head_dim)
        key = qk_packed3[..., self.latent_q_dim:].view(
            *qk_packed3.shape[:2], self.num_k_heads, self.head_dim)
        query, key = self._add_grouped_qk_means_inplace(
            query, key, query_pre, key_base)
        query, key = self._rms_normalize_qk(query.contiguous(),
                                            key.contiguous())

        value_current = self.v_proj_current(hs)
        delayed_v_state = self.v_proj_delayed(hs)
        zero_delayed = self.v_proj_delayed(
            hidden_states.new_zeros(1, 1, self.hidden_size))
        value_delayed = torch.cat([zero_delayed, delayed_v_state[:-1]], dim=0)
        value = torch.cat([value_current, value_delayed], dim=-1).contiguous()
        value = value.view(num_tokens, 1, self.num_k_heads, self.head_dim)

        q_end = self.latent_q_dim
        k_end = q_end + self.latent_k_dim
        output[:num_tokens, :q_end] = query.reshape(num_tokens,
                                                    self.latent_q_dim)
        output[:num_tokens, q_end:k_end] = key.reshape(num_tokens,
                                                       self.latent_k_dim)
        output[:num_tokens, k_end:] = value.reshape(num_tokens,
                                                    self.latent_k_dim)
        if self._zk_cca_decode_enabled:
            assert position_ids is not None
            self._apply_rope_to_output(position_ids, output, num_tokens)

    def _rms_normalize_qk(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Equivalent to RMSNorm with unit weights and eps=1e-12/head_dim.
        # Normalize one tensor at a time in fp32 to reduce peak memory versus
        # the custom rms_norm op, which materializes an additional fp32 output.
        eps = 1e-12
        sqrt_head_dim = float(self.sqrt_head_dim)

        query_fp32 = query.to(torch.float32)
        q_norm = torch.linalg.vector_norm(query_fp32, ord=2, dim=-1, keepdim=True)
        query_fp32.mul_(torch.rsqrt(q_norm * q_norm + eps))
        query_fp32.mul_(sqrt_head_dim)
        query.copy_(query_fp32)

        key_fp32 = key.to(torch.float32)
        k_norm = torch.linalg.vector_norm(key_fp32, ord=2, dim=-1, keepdim=True)
        key_fp32.mul_(torch.rsqrt(k_norm * k_norm + eps))
        key_fp32.mul_(sqrt_head_dim)
        temp = self.temp.to(torch.float32).view(1, 1, self.num_k_heads, 1)
        if self.config.clamp_temp:
            temp = torch.exp(torch.clamp(temp, 1e-7, 2.0))
        key_fp32.mul_(temp)
        key.copy_(key_fp32)
        return query, key

    def _add_grouped_qk_means_inplace(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        query_pre: torch.Tensor,
        key_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_k_heads = key_base.shape[2]
        key_base_fp32 = key_base.float()
        query_pre_grouped = query_pre.view(
            *query_pre.shape[:2], num_k_heads, self.gqa_groups, query_pre.shape[-1]
        )
        query_out_grouped = query.view_as(query_pre_grouped)
        query_out_grouped.add_(query_pre_grouped, alpha=0.5)
        query_out_grouped.add_(key_base_fp32.unsqueeze(-2), alpha=0.5)

        query_pre_mean = torch.mean(query_pre_grouped, dim=-2, dtype=torch.float32)
        key.add_(query_pre_mean, alpha=0.5)
        key.add_(key_base_fp32, alpha=0.5)
        return query, key

    def _conv_qk_decode(self, x: torch.Tensor) -> torch.Tensor:
        """Manual conv_qk for decode-sized inputs.

        Decode uses tiny sequence windows (currently total_padding + 1), so the
        generic conv path can spend a disproportionate amount of time on layout
        transforms and kernel setup. This manual implementation preserves the
        two-stage depthwise+grouped conv math while operating directly on the
        compact decode tensor.

        Input:  [N, C, S]
        Output: [N, C, S_out]
        """
        # Stage 1: depthwise conv over sequence.
        w0 = self.conv_qk_depthwise.weight.squeeze(1)  # [C, K0]
        b0 = self.conv_qk_depthwise.bias  # [C] or None

        x = x.to(w0.dtype)
        k0 = w0.shape[1]
        x_windows = x.unfold(-1, k0, 1)  # [N, C, L_mid, K0]
        mid = (x_windows * w0[:, None, :]).sum(dim=-1)  # [N, C, L_mid]
        if b0 is not None:
            mid = mid + b0[None, :, None]

        # Stage 2: grouped conv over the depthwise output.
        w1 = self.conv_qk_grouped.weight  # [C, D, K1]
        b1 = self.conv_qk_grouped.bias  # [C] or None
        g = self.num_k_heads + self.num_q_heads
        d = self.head_dim
        k1 = w1.shape[2]
        mid_windows = mid.view(mid.shape[0], g, d, mid.shape[-1]).unfold(-1, k1, 1)
        w1_grouped = w1.view(g, d, d, k1)
        out = torch.einsum("godk,sgdtk->sgot", w1_grouped, mid_windows)
        if b1 is not None:
            out = out + b1.view(1, g, d, 1)
        return out.reshape(x.shape[0], g * d, out.shape[-1])

    def _forward_zk_decode(
        self,
        qk_packed0: torch.Tensor,
        value_current: torch.Tensor,
        delayed_v_state: torch.Tensor,
        output: torch.Tensor,
        position_ids: torch.Tensor,
        conv_states: torch.Tensor,
        recurrent_states: torch.Tensor,
        state_indices: torch.Tensor,
    ) -> None:
        num_tokens = qk_packed0.shape[0]
        qk_output = output[:num_tokens, : self.in_out_ch]
        positions = position_ids.reshape(-1)[:num_tokens]
        rope_cache = self._get_zk_rope_cache(qk_packed0)
        grouped_bias = self.conv_qk_grouped.bias
        _get_zk_cca_ops()[0](
            qk_packed0[:, 0, :],
            self.conv_qk_depthwise.weight.squeeze(1),
            self.conv_qk_depthwise.bias,
            conv_states,
            state_indices,
            self._zk_grouped_weight,
            None
            if grouped_bias is None
            else grouped_bias.view(
                self.num_q_heads + self.num_k_heads, self.head_dim
            ),
            self.temp,
            rope_cache,
            positions,
            int(self.rotary_emb.rotary_dim),
            self.num_q_heads,
            self.head_dim,
            self.gqa_groups,
            PAD_SLOT_ID,
            float(self.sqrt_head_dim),
            self.config.clamp_temp,
            out=qk_output,
        )
        logger.info_once("Using Zyphra fused CCA decode and RoPE")

        is_pad = state_indices == PAD_SLOT_ID
        safe_indices = torch.where(
            is_pad, torch.zeros_like(state_indices), state_indices
        )
        value_delayed = recurrent_states[safe_indices].unsqueeze(1)
        value_delayed = torch.where(
            is_pad.view(-1, 1, 1),
            value_delayed.new_zeros(()),
            value_delayed,
        )
        if value_delayed.dtype != value_current.dtype:
            value_delayed = value_delayed.to(value_current.dtype)
        new_recurrent_state = delayed_v_state[:, 0, :]
        if new_recurrent_state.dtype != recurrent_states.dtype:
            new_recurrent_state = new_recurrent_state.to(recurrent_states.dtype)
        new_recurrent_state = torch.where(
            is_pad.view(-1, 1),
            new_recurrent_state.new_zeros(()),
            new_recurrent_state,
        )
        recurrent_states[safe_indices] = new_recurrent_state

        value = torch.cat([value_current, value_delayed], dim=-1).reshape(
            num_tokens, self.latent_k_dim
        )
        output[:num_tokens, self.in_out_ch :] = value
        decode_output = output[:num_tokens]
        output[:num_tokens] = torch.where(
            is_pad.view(-1, 1),
            decode_output.new_zeros(()),
            decode_output,
        )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        position_ids: torch.Tensor,
    ):
        forward_context = get_forward_context()

        attn_metadata: AttentionMetadata = forward_context.attn_metadata
        if attn_metadata is not None:
            assert isinstance(attn_metadata, dict)
            attn_metadata = attn_metadata[self.prefix]
            assert isinstance(attn_metadata, CCAAttentionMetadata)
            conv_states = self.kv_cache[0]
            recurrent_states = self.kv_cache[1]
            state_indices_tensor_p = attn_metadata.state_indices_tensor_p
            state_indices_tensor_d = attn_metadata.state_indices_tensor_d
            if state_indices_tensor_d is not None and state_indices_tensor_d.dim() > 1:
                state_indices_tensor_d = state_indices_tensor_d[:, 0]
            has_initial_states_p = attn_metadata.has_initial_states_p
            query_start_loc_p = attn_metadata.query_start_loc_p

        if attn_metadata is None:
            # V1 profile run
            self._forward_no_cache(hidden_states, output, position_ids)
            return

        num_prefills = attn_metadata.num_prefills  # request count
        num_decodes = attn_metadata.num_decode_tokens  # token count (=request)
        num_prefill_tokens = attn_metadata.num_prefill_tokens  # token count
        has_prefill = num_prefills > 0
        has_decode = num_decodes > 0
        use_zk_cca_decode = (
            self._zk_cca_decode_enabled and has_decode and not has_prefill
        )
        use_zk_cca_prefill = (
            self._zk_cca_decode_enabled and has_prefill and not has_decode
        )
        num_actual_tokens = num_decodes + num_prefill_tokens

        hidden_states = hidden_states[:num_actual_tokens]

        # Batch size is effectively 1 in this path, so insert the singleton
        # dimension directly instead of transposing and materializing a copy.
        hs = hidden_states.unsqueeze(1)  # [S, 1, H]
        batch_size = hs.shape[1]

        q = self.q_proj(hs)  # [S, B, latent_q_dim]
        k = self.k_proj(hs)  # [S, B, latent_k_dim]
        qk_packed0 = torch.cat([q, k], dim=-1)  # [S, B, latent_q + latent_k]
        del q
        del k
        if use_zk_cca_decode:
            assert state_indices_tensor_d is not None
            self._forward_zk_decode(
                qk_packed0,
                self.v_proj_current(hs),
                self.v_proj_delayed(hs),
                output,
                position_ids,
                conv_states,
                recurrent_states,
                state_indices_tensor_d,
            )
            return

        # NOTE: V1 puts decode before prefill
        # Separate prefill and decode by splitting varlen input
        # Split along token dimension
        qk_packed0_d, qk_packed0_p = torch.split(
            qk_packed0[:num_actual_tokens],
            [num_decodes, num_prefill_tokens],
            dim=0,
        )
        delayed_v_state = self.v_proj_delayed(hs[:num_actual_tokens])
        delayed_v_state_d, delayed_v_state_p = torch.split(
            delayed_v_state,
            [num_decodes, num_prefill_tokens],
            dim=0,
        )

        qk_packed3 = None
        if not use_zk_cca_prefill:
            query_pre = qk_packed0[..., : self.latent_q_dim].view(
                *qk_packed0.shape[:2], self.num_q_heads, self.head_dim
            )
            key_base = qk_packed0[..., self.latent_q_dim :].view(
                *qk_packed0.shape[:2], self.num_k_heads, self.head_dim
            )
            qk_packed3 = torch.empty(
                (num_actual_tokens, batch_size, self.in_out_ch),
                device=hs.device,
                dtype=hs.dtype,
            )
        value_delayed = torch.empty(
            (num_actual_tokens, batch_size, self.recurrent_v_dim),
            device=hs.device,
            dtype=hs.dtype,
        )
        decode_is_pad: torch.Tensor | None = None
        if has_prefill:
            assert state_indices_tensor_p is not None
            assert has_initial_states_p is not None
            assert query_start_loc_p is not None
            # Prefill
            prefill_slice = slice(num_decodes, num_decodes + num_prefill_tokens)
            value_delayed_prefill = value_delayed[prefill_slice]
            if use_zk_cca_prefill:
                request_indices = torch.repeat_interleave(
                    torch.arange(
                        num_prefills, device=hs.device, dtype=torch.int32
                    ),
                    torch.diff(query_start_loc_p).to(torch.int32),
                )
                grouped_bias = self.conv_qk_grouped.bias
                _, qk_prefill = _get_zk_cca_ops()[1](
                    hidden_states[num_decodes:num_actual_tokens],
                    qk_packed0_p[:, 0],
                    None,
                    conv_states,
                    query_start_loc_p,
                    has_initial_states_p,
                    state_indices_tensor_p,
                    request_indices,
                    self.conv_qk_depthwise.weight.squeeze(1),
                    self.conv_qk_depthwise.bias,
                    self._zk_grouped_weight,
                    None
                    if grouped_bias is None
                    else grouped_bias.view(
                        self.num_q_heads + self.num_k_heads, self.head_dim
                    ),
                    self.temp,
                    num_query_heads=self.num_q_heads,
                    head_dim=self.head_dim,
                    gqa_groups=self.gqa_groups,
                    sqrt_head_dim=float(self.sqrt_head_dim),
                    clamp_temp=self.config.clamp_temp,
                )
                logger.info_once("Using Zyphra fused CCA prefill")
            else:
                assert qk_packed3 is not None
                qk_packed3_prefill = qk_packed3[prefill_slice]
            for i in range(len(query_start_loc_p) - 1):
                start_i, end_i = query_start_loc_p[i], query_start_loc_p[i + 1]
                delayed_v_state_cur = delayed_v_state_p[start_i:end_i]

                if has_initial_states_p[i]:
                    value_delayed_cached = recurrent_states[
                        state_indices_tensor_p[i]].unsqueeze(0).unsqueeze(0)
                    if value_delayed_cached.dtype != value_delayed.dtype:
                        value_delayed_cached = value_delayed_cached.to(
                            value_delayed.dtype)
                else:
                    value_delayed_cached = self.v_proj_delayed(
                        hs.new_zeros(1, 1, self.hidden_size))

                value_delayed_prefill[start_i:end_i] = torch.cat(
                    [value_delayed_cached, delayed_v_state_cur[:-1]], dim=0)

                if not use_zk_cca_prefill:
                    qk_packed1_cur = qk_packed0_p[
                        start_i:end_i
                    ].permute(1, 2, 0)
                    if has_initial_states_p[i]:
                        qk_packed0_cached = conv_states[
                            state_indices_tensor_p[i]
                        ].unsqueeze(0)
                        if qk_packed0_cached.dtype != qk_packed1_cur.dtype:
                            qk_packed0_cached = qk_packed0_cached.to(
                                qk_packed1_cur.dtype
                            )
                        qk_packed2_cur = torch.cat(
                            [qk_packed0_cached, qk_packed1_cur], dim=-1
                        )
                    else:
                        qk_packed2_cur = F.pad(
                            qk_packed1_cur, (self.total_padding, 0)
                        )
                    conv_states_cur = nn.functional.pad(
                        qk_packed2_cur,
                        (self.total_padding - qk_packed2_cur.shape[-1], 0),
                    )
                    conv_states[state_indices_tensor_p[i]] = conv_states_cur.to(
                        device=conv_states.device, dtype=conv_states.dtype
                    )
                    qk_packed3_cur = self.conv_qk_grouped(
                        self.conv_qk_depthwise(qk_packed2_cur)
                    ).permute(2, 0, 1)
                    qk_packed3_prefill[start_i:end_i] = qk_packed3_cur

            recurrent_states[state_indices_tensor_p] = delayed_v_state_p[
                query_start_loc_p[1:] - 1, 0, :].to(
                    device=recurrent_states.device,
                    dtype=recurrent_states.dtype)

        if has_decode:
            assert state_indices_tensor_d is not None
            assert qk_packed3 is not None
            # Generation
            # In generation B and S are actually the same in meaning
            # That's why we don't need to transpose qk_packed0
            # qk_packed0_d [S, 1, H]
            decode_is_pad = state_indices_tensor_d == PAD_SLOT_ID
            # block_id=0 reserved
            # Zvllm/vllm/v1/core/block_pool.py
            safe_decode_indices = torch.where(
                decode_is_pad,
                torch.zeros_like(state_indices_tensor_d),
                state_indices_tensor_d,
            )
            qk_packed0_d = torch.where(
                decode_is_pad.view(-1, 1, 1),
                qk_packed0_d.new_zeros(()),
                qk_packed0_d,
            )
            qk_packed0_cached = conv_states[
                safe_decode_indices
            ]  # [S, H, total_padding]
            qk_packed0_cached = torch.where(
                decode_is_pad.view(-1, 1, 1),
                qk_packed0_cached.new_zeros(()),
                qk_packed0_cached,
            )
            qk_packed0_cached_for_compute = qk_packed0_cached
            decode_qk_dtype = qk_packed0_d.dtype
            if qk_packed0_cached_for_compute.dtype != decode_qk_dtype:
                qk_packed0_cached_for_compute = qk_packed0_cached_for_compute.to(
                    decode_qk_dtype
                )
            qk_packed0_cat = torch.cat(
                [qk_packed0_cached_for_compute, qk_packed0_d.transpose(1, 2)], dim=-1
            )  # [S, H, total_padding + 1]
            qk_packed3_d = self._conv_qk_decode(qk_packed0_cat).transpose(
                1, 2
            )  # [S, 1, E]
            qk_packed3[:num_decodes] = qk_packed3_d

            new_qk_packed0_cache = qk_packed0_cached.roll(shifts=-1, dims=-1)
            new_qk_packed0_cache[..., -1] = qk_packed0_d[:, 0, :].to(
                new_qk_packed0_cache.dtype
            )
            new_qk_packed0_cache = torch.where(
                decode_is_pad.view(-1, 1, 1),
                new_qk_packed0_cache.new_zeros(()),
                new_qk_packed0_cache,
            )
            conv_states[safe_decode_indices] = new_qk_packed0_cache.to(
                device=conv_states.device, dtype=conv_states.dtype
            )

            value_delayed_decode = recurrent_states[safe_decode_indices].unsqueeze(1)
            value_delayed_decode = torch.where(
                decode_is_pad.view(-1, 1, 1),
                value_delayed_decode.new_zeros(()),
                value_delayed_decode,
            )
            if value_delayed_decode.dtype != value_delayed.dtype:
                value_delayed_decode = value_delayed_decode.to(value_delayed.dtype)
            value_delayed[:num_decodes] = value_delayed_decode
            new_recurrent_state = delayed_v_state_d[:, 0, :].to(
                recurrent_states.dtype)
            new_recurrent_state = torch.where(
                decode_is_pad.view(-1, 1),
                new_recurrent_state.new_zeros(()),
                new_recurrent_state,
            )
            recurrent_states[safe_decode_indices] = new_recurrent_state.to(
                device=recurrent_states.device,
                dtype=recurrent_states.dtype)

        del qk_packed0_d
        del qk_packed0_p
        del delayed_v_state_d
        del delayed_v_state_p

        # Values from the two time streams
        v1 = self.v_proj_current(hs)  # [S, B, latent_k_dim/2]
        value = torch.cat([v1, value_delayed], dim=-1).contiguous()
        value = value.view(
            num_actual_tokens, batch_size, self.num_k_heads, self.head_dim
        )  # [S, B, kh, dh]
        del value_delayed

        # Build queries/keys from conv output + means
        qk_result = (
            qk_prefill.unsqueeze(1) if use_zk_cca_prefill else qk_packed3
        )
        assert qk_result is not None
        query = qk_result[..., : self.latent_q_dim].view(
            num_actual_tokens, batch_size, self.num_q_heads, self.head_dim
        )
        key = qk_result[..., self.latent_q_dim :].view(
            num_actual_tokens, batch_size, self.num_k_heads, self.head_dim
        )
        if not use_zk_cca_prefill:
            query, key = self._add_grouped_qk_means_inplace(
                query.float(), key.float(), query_pre, key_base
            )
            del query_pre
            del key_base
        del qk_packed0
        del qk_result

        if not use_zk_cca_prefill:
            query, key = self._rms_normalize_qk(
                query.contiguous(), key.contiguous()
            )
        # Flatten the singleton batch dimension without transpose/cat copies and
        # write directly into the preallocated output buffer.
        query = query.reshape(num_actual_tokens, self.latent_q_dim)
        key = key.reshape(num_actual_tokens, self.latent_k_dim)
        value = value.reshape(num_actual_tokens, self.latent_k_dim)
        q_end = self.latent_q_dim
        k_end = q_end + self.latent_k_dim
        output[:num_actual_tokens, :q_end] = query
        output[:num_actual_tokens, q_end:k_end] = key
        output[:num_actual_tokens, k_end:] = value
        if decode_is_pad is not None:
            decode_output = output[:num_decodes]
            output[:num_decodes] = torch.where(
                decode_is_pad.view(-1, 1),
                decode_output.new_zeros(()),
                decode_output,
            )
        if self._zk_cca_decode_enabled:
            self._apply_rope_to_output(
                position_ids, output, num_actual_tokens
            )

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        assert self.model_config is not None
        assert self.cache_config is not None
        return MambaStateDtypeCalculator.cca_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
        )

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.cca_state_shape(
            tp_world_size=get_tensor_model_parallel_world_size(),
            conv_kernel_size=self.total_padding,
            num_k_heads=self.num_k_heads,
            num_q_heads=self.num_q_heads,
            head_dim=self.head_dim,
            recurrent_state_size=self.recurrent_v_dim,
        )

    @property
    def mamba_type(self) -> str:
        return "cca"

    def get_attn_backend(self) -> type["AttentionBackend"]:
        from vllm.v1.attention.backends.cca_attn import CCAAttentionBackend

        return CCAAttentionBackend


def cca(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    position_ids: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self.forward_cuda(
        hidden_states=hidden_states,
        output=output,
        position_ids=position_ids,
    )


def cca_fake(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    position_ids: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="cca",
    op_func=cca,
    mutates_args=["output"],
    fake_impl=cca_fake,
)
