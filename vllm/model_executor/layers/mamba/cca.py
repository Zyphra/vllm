# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from vllm.v1.attention.backend import AttentionBackend

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.config import CacheConfig, ModelConfig, get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator, MambaStateShapeCalculator)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.cca_attn import (
    CCAAttentionMetadata)

from vllm.model_executor.layers.mamba.ops import (
    run_causal_conv1d_update, grouped_conv1d_decode,
    cca_decode_fused_available, cca_decode_fused,
    cca_prefill_fused,
    cca_prefill_fused_hip_available, cca_prefill_fused_hip,
    fused_pad_gather_scatter,
    fused_qk_mean,
)

_CCA_FUSED_ENABLED = os.environ.get(
    "VLLM_CCA_FUSED_ENABLED", "0").lower() in ("1", "true", "yes")

_CCA_TRITON_FUSION_ENABLED = os.environ.get(
    "VLLM_CCA_TRITON_FUSION_ENABLED", "0").lower() in ("1", "true", "yes")

@CustomOp.register("cca")
class CCA(MambaBase, CustomOp):
    def __init__(
            self,
            config,
            cca_num_k_heads: int = 2,
            cca_num_q_heads: int = 8,
            cca_num_heads: int = 16,
            hidden_size: Optional[int] = None,
            cca_time0: int = 2,
            cca_time1: int = 2,
            layer_number: int = 0,
            model_config: Optional[ModelConfig] = None,
            cache_config: Optional[CacheConfig] = None,
            quant_config: Optional[QuantizationConfig] = None,
            use_triton: bool = False,
            prefix: str = ""
        ):
        super().__init__()
        self.config       = config
        self.model_config = model_config
        self.cache_config = cache_config
        self.layer_number = layer_number
        self.prefix = prefix
        self.use_triton = use_triton

        # Use the model's true hidden size unless explicitly overridden.
        # (In Megatron this is the lane's hidden_size_in.)
        self.hidden_size  = int(hidden_size or config.hidden_size)

        self.cca_time0     = cca_time0
        self.cca_time1     = cca_time1
        self.padding0      = cca_time0 - 1
        self.padding1      = cca_time1 - 1
        self.total_padding = self.padding0 + self.padding1

        self.num_k_heads = int(cca_num_k_heads)
        self.num_q_heads = int(cca_num_q_heads)
        self.num_heads   = int(cca_num_heads)

        # Geometry
        self.head_dim      = self.hidden_size // self.num_heads
        self.latent_k_dim  = self.num_k_heads * self.head_dim
        self.latent_q_dim  = self.num_q_heads * self.head_dim
        self.sqrt_head_dim = np.sqrt(self.head_dim)
        self.gqa_groups    = self.num_q_heads // self.num_k_heads       
        assert self.num_q_heads % self.num_k_heads == 0, "q_heads must be a multiple of k_heads"
        assert (self.latent_k_dim + self.latent_q_dim) == (self.num_k_heads + self.num_q_heads) * self.head_dim

        # Projections
        self.linear_q  = ReplicatedLinear(self.hidden_size,
                                          self.latent_q_dim,
                                          bias=self.config.attention_bias,
                                          quant_config=quant_config,
                                          return_bias=False,
                                          prefix=f"{prefix}.linear_q")
        self.linear_k  = ReplicatedLinear(self.hidden_size,
                                          self.latent_k_dim,
                                          bias=self.config.attention_bias,
                                          quant_config=quant_config,
                                          return_bias=False,
                                          prefix=f"{prefix}.linear_k")
        self.val_proj1 = ReplicatedLinear(self.hidden_size,
                                          self.latent_k_dim // 2,
                                          bias=self.config.attention_bias,
                                          quant_config=quant_config,
                                          return_bias=False,
                                          prefix=f"{prefix}.val_proj1")
        self.val_proj2 = ReplicatedLinear(self.hidden_size,
                                          self.latent_k_dim // 2,
                                          bias=self.config.attention_bias,
                                          quant_config=quant_config,
                                          return_bias=False,
                                          prefix=f"{prefix}.val_proj2")

        # Depthwise + grouped conv along sequence (exactly like Megatron)
        in_out_ch = self.latent_k_dim + self.latent_q_dim
        self.in_out_ch = in_out_ch
        self.conv_qk = nn.Sequential(
            nn.Conv1d(
                in_channels=in_out_ch, out_channels=in_out_ch,
                kernel_size=self.cca_time0, groups=in_out_ch, padding=0, stride=1
            ),
            nn.Conv1d(
                in_channels=in_out_ch, out_channels=in_out_ch,
                kernel_size=self.cca_time1, groups=(self.num_k_heads + self.num_q_heads),
                padding=0, stride=1
            ),
        )


        dim, _, kernel_width = self.conv_qk[0].weight.shape
        groups = self.num_k_heads + self.num_q_heads
        head_dim = dim // groups

        self.dw_weight_flat = self.conv_qk[0].weight.reshape(dim, kernel_width).contiguous()
        self.gw_weight_flat = self.conv_qk[1].weight.reshape(groups, head_dim, -1, kernel_width).contiguous()
        self.gw_bias_flat = self.conv_qk[1].bias.reshape(groups, -1).contiguous()

        self._gw_weight_T = None

        # Per-k head temperature (Megatron: shape [num_k_heads])
        self.temp = nn.Parameter(torch.zeros(self.num_k_heads))

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        self.kv_cache = (torch.tensor([]), torch.tensor([]))

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        return

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        torch.ops.vllm.cca(
            hidden_states,
            output,
            self.prefix,
        )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):

        forward_context = get_forward_context()

        attn_metadata: AttentionMetadata = forward_context.attn_metadata
        if attn_metadata is not None:
            assert isinstance(attn_metadata, dict)
            attn_metadata = attn_metadata[self.prefix]
            assert isinstance(attn_metadata, CCAAttentionMetadata)
            self_kv_cache = self.kv_cache[forward_context.virtual_engine]
            conv_states = self_kv_cache[0]
            prev_hs = self_kv_cache[1]
            state_indices_tensor = attn_metadata.state_indices_tensor
            has_initial_states_p = attn_metadata.has_initial_states_p
            query_start_loc_p = attn_metadata.query_start_loc_p

        if attn_metadata is None:
            # V1 profile run
            hs = hidden_states.unsqueeze(0).transpose(0, 1).contiguous()
            hs_d = F.pad(hs[:-1], pad=(0, 0, 0, 0, 1, 0))    # [S, B, H]
            q = self.linear_q(hs)  # [S, B, latent_q_dim]
            k = self.linear_k(hs)  # [S, B, latent_k_dim]
            qk_packed0 = torch.cat([q, k], dim=-1)  # [S, B, latent_q + latent_k]

            # Pre-mean tensors in head form (for "qk_mean_{q,k}" calc)
            query_pre = qk_packed0[..., :self.latent_q_dim].view(
                *qk_packed0.shape[:2], self.num_q_heads, self.head_dim
            )  # [S, B, qh, dh]

            key_pre = qk_packed0[..., self.latent_q_dim:].view(
                *qk_packed0.shape[:2], self.num_k_heads, self.head_dim
            )  # [S, B, kh, dh]
            key_pre = key_pre.unsqueeze(-2).repeat(1, 1, 1, self.gqa_groups, 1) \
                            .view(*qk_packed0.shape[:2], self.num_q_heads, self.head_dim)  # [S, B, qh, dh]

            # Means for residual mixing (identical to Megatron)
            qk_mean_q = (query_pre + key_pre) / 2
            qk_mean_k = qk_mean_q.view(*qk_mean_q.shape[:2], self.num_k_heads, self.gqa_groups, -1).mean(dim=-2)

            qk_packed1 = qk_packed0.permute(1, 2, 0)             # [B, E, S]
            qk_packed2 = F.pad(qk_packed1, (self.total_padding, 0))
            qk_packed3 = self.conv_qk(qk_packed2).permute(2, 0, 1)  # [S, B, E]

            # Build queries/keys from conv output + means
            query = qk_packed3[..., :self.latent_q_dim].view(
                *qk_packed3.shape[:2], self.num_q_heads, self.head_dim
            ) + qk_mean_q  # [S, B, qh, dh]

            key = qk_packed3[..., self.latent_q_dim:].view(
                *qk_packed3.shape[:2], self.num_k_heads, self.head_dim
            ) + qk_mean_k  # [S, B, kh, dh]

            # Values from the two time streams
            v1 = self.val_proj1(hs)   # [S, B, latent_k_dim/2]
            v2 = self.val_proj2(hs_d) # [S, B, latent_k_dim/2]
            value = torch.cat([v1, v2], dim=-1).contiguous() \
                        .view(*hs.shape[:2], self.num_k_heads, self.head_dim)  # [S, B, kh, dh]

            # L2-normalize per head, then scale like Megatron
            query_norm = query.norm(p=2, dim=-1, keepdim=True)
            key_norm   = key.norm(p=2,   dim=-1, keepdim=True)    
            # temp: [kh] -> [1,1,kh,1]
            key = (key * (self.sqrt_head_dim / key_norm)) * self.temp[None, None].unsqueeze(-1)
            query = query * (self.sqrt_head_dim / query_norm)
            
            return hs

        num_prefills = attn_metadata.num_prefills  # request count
        num_decodes = attn_metadata.num_decode_tokens  # token count (=request)
        num_prefill_tokens = attn_metadata.num_prefill_tokens  # token count
        has_prefill = num_prefills > 0
        has_decode = num_decodes > 0
        num_actual_tokens = num_decodes + num_prefill_tokens

        num_input_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states[:num_actual_tokens]

        hidden_states = hidden_states.unsqueeze(0)  # [B, S, E]
        batch_size, _, _ = hidden_states.shape

        # ---- Switch to Megatron's layout: [S, B, H] ----
        hs = hidden_states.transpose(0, 1).contiguous()  # [S, B, H]

        q = self.linear_q(hs)  # [S, B, latent_q_dim]
        k = self.linear_k(hs)  # [S, B, latent_k_dim]
        qk_packed0 = torch.cat([q, k], dim=-1)  # [S, B, latent_q + latent_k]

        # Pre-mean tensors in head form (for "qk_mean_{q,k}" calc)
        query_pre = qk_packed0[..., :self.latent_q_dim].view(
            *qk_packed0.shape[:2], self.num_q_heads, self.head_dim
        )  # [S, B, qh, dh]

        key_pre = qk_packed0[..., self.latent_q_dim:].view(
            *qk_packed0.shape[:2], self.num_k_heads, self.head_dim
        )  # [S, B, kh, dh]
        key_pre = key_pre.unsqueeze(-2).repeat(1, 1, 1, self.gqa_groups, 1) \
                          .view(*qk_packed0.shape[:2], self.num_q_heads, self.head_dim)  # [S, B, qh, dh]

        # Means for residual mixing (identical to Megatron)
        qk_mean_q = (query_pre + key_pre) / 2
        qk_mean_k = qk_mean_q.view(*qk_mean_q.shape[:2], self.num_k_heads, self.gqa_groups, -1).mean(dim=-2)

        # NOTE: V1 puts decode before prefill
        # Separate prefill and decode by splitting varlen input
        # Split along token dimension
        qk_packed0_d, qk_packed0_p = torch.split(
            qk_packed0[:num_actual_tokens],
            [num_decodes, num_prefill_tokens],
            dim=0,
        )
        hs_d, hs_p = torch.split(
            hs[:num_actual_tokens],
            [num_decodes, num_prefill_tokens],
            dim=0,
        )

        # Split along batch dimension
        state_indices_tensor_d, state_indices_tensor_p = torch.split(
            state_indices_tensor[:num_actual_tokens],
            [num_decodes, num_prefills],
            dim=0,
        )

        # TODO: allocate memory for output tensors
        qk_packed3_output_list = []
        hs2_output_list = []
        qk_packed3_p = None
        _fused_query_d = None
        _fused_key_d = None
        decode_is_pad: Optional[torch.Tensor] = None

        if has_prefill:
            # Prefill
            hs2 = torch.zeros((num_prefill_tokens, batch_size, self.hidden_size), device=hs.device, dtype=hs.dtype)
            qk_packed3_p = torch.zeros((num_prefill_tokens, batch_size, self.in_out_ch), device=hs.device, dtype=hs.dtype)
            for i in range(len(query_start_loc_p) - 1):
                start_i, end_i = query_start_loc_p[i], query_start_loc_p[i + 1]
                hs2_cur = hs_p[start_i:end_i, :, :] # [S_cur, B, H]
                qk_packed0_cur = qk_packed0_p[start_i:end_i, :, :]  # [S_cur, B, H]    
                qk_packed1_cur = qk_packed0_cur.permute(1, 2, 0)  # [1, H, S_cur]
                
                if has_initial_states_p[i]:
                    hs2_cached = prev_hs[state_indices_tensor_p[i]].unsqueeze(0).unsqueeze(0) # [1, 1, H]
                    hs2_cur = torch.cat([hs2_cached, hs2_cur[:-1]], dim=0) # [S_cur, 1, H]
                    qk_packed0_cached = conv_states[state_indices_tensor_p[i]].unsqueeze(0) # [1, H, total_padding]
                    qk_packed2_cur = torch.cat([qk_packed0_cached, qk_packed1_cur], dim=-1) # [1, H, S_cur + total_padding]
                else:
                    hs2_cur = F.pad(hs2_cur[:-1], pad=(0, 0, 0, 0, 1, 0))
                    qk_packed2_cur = F.pad(qk_packed1_cur, (self.total_padding, 0))
                    
                hs2[start_i:end_i] = hs2_cur

                conv_states_cur = nn.functional.pad(qk_packed2_cur, (self.cca_time0 - qk_packed2_cur.shape[-1], 0))
                conv_states[state_indices_tensor_p[i]] = conv_states_cur.to(conv_states.device)
                
                # Computing conv
                qk_packed3_cur = self.conv_qk(qk_packed2_cur).permute(2, 0, 1)  # [S, B, E]
                qk_packed3_p[start_i:end_i] = qk_packed3_cur

            qk_packed3_output_list.append(qk_packed3_p)
            hs2_output_list.append(hs2)
            prev_hs[state_indices_tensor_p] = hs_p[query_start_loc_p[1:] - 1, 0, :].to(prev_hs.device)

        if has_decode:
            # Generation
            # In generation B and S are actually the same in meaning
            # That's why we don't need to transpose qk_packed0
            # qk_packed0_d [S, 1, H]

            # Handle PAD_SLOT_ID with torch.where for CUDA graph compatibility
            decode_is_pad = (state_indices_tensor_d == PAD_SLOT_ID)
            # block_id=0 reserved for safe indexing (see vllm/v1/core/block_pool.py)
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
            hs_d = torch.where(
                decode_is_pad.view(-1, 1, 1),
                hs_d.new_zeros(()),
                hs_d,
            )


            qk_packed0_cached = conv_states[safe_decode_indices]  # [S, H, total_padding]
            qk_packed0_cached = torch.where(
                decode_is_pad.view(-1, 1, 1),
                qk_packed0_cached.new_zeros(()),
                qk_packed0_cached,
            )


            qk_packed0_cat = torch.cat([qk_packed0_cached, qk_packed0_d.transpose(1, 2)], dim=-1) # [S, H, total_padding + 1]
            qk_packed3_d = self.conv_qk(qk_packed0_cat).transpose(1, 2)  # [S, 1, E]
            # Zero out conv output for padded positions (conv bias adds non-zero values even with zero input)
            qk_packed3_d = torch.where(
                decode_is_pad.view(-1, 1, 1),
                qk_packed3_d.new_zeros(()),
                qk_packed3_d,
            )
            qk_packed3_output_list.insert(0, qk_packed3_d)
            
            new_qk_packed0_cache = qk_packed0_cached.roll(shifts=-1, dims=-1)
            new_qk_packed0_cache[..., -1] = qk_packed0_d[:, 0, :]
            new_qk_packed0_cache = torch.where(
                decode_is_pad.view(-1, 1, 1),
                new_qk_packed0_cache.new_zeros(()),
                new_qk_packed0_cache,
            )
            conv_states[safe_decode_indices] = new_qk_packed0_cache.to(
                device=conv_states.device, dtype=conv_states.dtype)

            hs2 = prev_hs[safe_decode_indices].unsqueeze(1) # [S, 1, H]
            hs2 = torch.where(
                decode_is_pad.view(-1, 1, 1),
                hs2.new_zeros(()),
                hs2,
            )
            hs2_output_list.insert(0, hs2)
            new_prev_hs = hs_d[:, 0, :].to(prev_hs.dtype)
            new_prev_hs = torch.where(
                decode_is_pad.view(-1, 1),
                new_prev_hs.new_zeros(()),
                new_prev_hs,
            )
            prev_hs[safe_decode_indices] = new_prev_hs.to(
                device=prev_hs.device, dtype=prev_hs.dtype)


        qk_packed3 = torch.vstack(qk_packed3_output_list)[:num_actual_tokens]
        hs2 = torch.vstack(hs2_output_list)[:num_actual_tokens]

        # Values from the two time streams
        v1 = self.val_proj1(hs)   # [S, B, latent_k_dim/2]
        v2 = self.val_proj2(hs2)
        value = torch.cat([v1, v2], dim=-1).contiguous()
        value = value.view(num_actual_tokens, batch_size, self.num_k_heads, self.head_dim)  # [S, B, kh, dh]

        # Build queries/keys from conv output + means
        query = qk_packed3[..., :self.latent_q_dim].view(
            num_actual_tokens, batch_size, self.num_q_heads, self.head_dim
        ) + qk_mean_q  # [S, B, qh, dh]

        key = qk_packed3[..., self.latent_q_dim:].view(
            num_actual_tokens, batch_size, self.num_k_heads, self.head_dim
        ) + qk_mean_k  # [S, B, kh, dh]

        # L2-normalize per head, then scale like Megatron
        query_norm = query.norm(p=2, dim=-1, keepdim=True)
        key_norm   = key.norm(p=2,   dim=-1, keepdim=True)

        # temp: [kh] -> [1,1,kh,1]
        if self.config.clamp_temp:
            key = (key * (self.sqrt_head_dim / key_norm)) * torch.exp(torch.clamp(self.temp[None, None].unsqueeze(-1),0.0000001,2))
        else:
            key = (key * (self.sqrt_head_dim / key_norm)) * self.temp[None, None].unsqueeze(-1)
        query = query * (self.sqrt_head_dim / query_norm)
        
        # Flatten head axis, then return to vllm layout [S, ...]
        query = query.view(num_actual_tokens, batch_size, self.num_q_heads * self.head_dim).transpose(0, 1).contiguous().squeeze(0)
        key   = key  .view(num_actual_tokens, batch_size, self.num_k_heads * self.head_dim).transpose(0, 1).contiguous().squeeze(0)
        value = value.view(num_actual_tokens, batch_size, self.num_k_heads * self.head_dim).transpose(0, 1).contiguous().squeeze(0)
        qkv = torch.cat([query, key, value], dim=1)
        output[:num_actual_tokens] = qkv

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        assert self.model_config is not None
        assert self.cache_config is not None
        return MambaStateDtypeCalculator.cca_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
        )

    def get_state_shape(self) -> tuple[tuple[int, ...]]:
        return MambaStateShapeCalculator.cca_state_shape(
            tp_world_size=get_tensor_model_parallel_world_size(),
            conv_kernel_size=self.total_padding,
            num_k_heads=self.num_k_heads,
            num_q_heads=self.num_q_heads,
            head_dim=self.head_dim,
            hidden_size=self.hidden_size,
        )

    def forward_triton(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        forward_context = get_forward_context()
        attn_metadata: AttentionMetadata = forward_context.attn_metadata
        if attn_metadata is not None:
            assert isinstance(attn_metadata, dict)
            attn_metadata = attn_metadata[self.prefix]
            assert isinstance(attn_metadata, CCAAttentionMetadata)
            self_kv_cache = self.kv_cache[forward_context.virtual_engine]
            conv_states = self_kv_cache[0]
            prev_hs = self_kv_cache[1]
            state_indices_tensor = attn_metadata.state_indices_tensor
            has_initial_states_p = attn_metadata.has_initial_states_p
            query_start_loc_p = attn_metadata.query_start_loc_p
        if attn_metadata is None:
            # V1 profile run
            hs = hidden_states.unsqueeze(0).transpose(0, 1).contiguous()
            hs_d = F.pad(hs[:-1], pad=(0, 0, 0, 0, 1, 0))    # [S, B, H]
            q = self.linear_q(hs)  # [S, B, latent_q_dim]
            k = self.linear_k(hs)  # [S, B, latent_k_dim]
            qk_packed0 = torch.cat([q, k], dim=-1)  # [S, B, latent_q + latent_k]

            # Pre-mean tensors in head form (for "qk_mean_{q,k}" calc)
            query_pre = qk_packed0[..., :self.latent_q_dim].view(
                *qk_packed0.shape[:2], self.num_q_heads, self.head_dim
            )  # [S, B, qh, dh]

            key_pre = qk_packed0[..., self.latent_q_dim:].view(
                *qk_packed0.shape[:2], self.num_k_heads, self.head_dim
            )  # [S, B, kh, dh]
            key_pre = key_pre.unsqueeze(-2).repeat(1, 1, 1, self.gqa_groups, 1) \
                            .view(*qk_packed0.shape[:2], self.num_q_heads, self.head_dim)  # [S, B, qh, dh]

            # Means for residual mixing (identical to Megatron)
            qk_mean_q = (query_pre + key_pre) / 2
            qk_mean_k = qk_mean_q.view(*qk_mean_q.shape[:2], self.num_k_heads, self.gqa_groups, -1).mean(dim=-2)

            qk_packed1 = qk_packed0.permute(1, 2, 0)             # [B, E, S]
            qk_packed2 = F.pad(qk_packed1, (self.total_padding, 0))
            qk_packed3 = self.conv_qk(qk_packed2).permute(2, 0, 1)  # [S, B, E]

            # Build queries/keys from conv output + means
            query = qk_packed3[..., :self.latent_q_dim].view(
                *qk_packed3.shape[:2], self.num_q_heads, self.head_dim
            ) + qk_mean_q  # [S, B, qh, dh]

            key = qk_packed3[..., self.latent_q_dim:].view(
                *qk_packed3.shape[:2], self.num_k_heads, self.head_dim
            ) + qk_mean_k  # [S, B, kh, dh]

            # Values from the two time streams
            v1 = self.val_proj1(hs)   # [S, B, latent_k_dim/2]
            v2 = self.val_proj2(hs_d) # [S, B, latent_k_dim/2]
            value = torch.cat([v1, v2], dim=-1).contiguous() \
                        .view(*hs.shape[:2], self.num_k_heads, self.head_dim)  # [S, B, kh, dh]

            # L2-normalize per head, then scale like Megatron
            query_norm = query.norm(p=2, dim=-1, keepdim=True)
            key_norm   = key.norm(p=2,   dim=-1, keepdim=True)    
            # temp: [kh] -> [1,1,kh,1]
            key = (key * (self.sqrt_head_dim / key_norm)) * self.temp[None, None].unsqueeze(-1)
            query = query * (self.sqrt_head_dim / query_norm)
            
            return hs

        num_prefills = attn_metadata.num_prefills  # request count
        num_decodes = attn_metadata.num_decode_tokens  # token count (=request)
        num_prefill_tokens = attn_metadata.num_prefill_tokens  # token count
        has_prefill = num_prefills > 0
        has_decode = num_decodes > 0
        num_actual_tokens = num_decodes + num_prefill_tokens

        num_input_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states[:num_actual_tokens]

        hidden_states = hidden_states.unsqueeze(0)  # [B, S, E]
        batch_size, _, _ = hidden_states.shape

        # ---- Switch to Megatron's layout: [S, B, H] ----
        hs = hidden_states.transpose(0, 1).contiguous()  # [S, B, H]

        q = self.linear_q(hs)  # [S, B, latent_q_dim]
        k = self.linear_k(hs)  # [S, B, latent_k_dim]
        qk_packed0 = torch.cat([q, k], dim=-1)  # [S, B, latent_q + latent_k]

        if _CCA_TRITON_FUSION_ENABLED:
            qk_mean_q, qk_mean_k = fused_qk_mean(
                qk_packed0, self.num_q_heads, self.num_k_heads,
                self.head_dim, self.gqa_groups,
            )
        else:
            query_pre = qk_packed0[..., :self.latent_q_dim].view(
                *qk_packed0.shape[:2], self.num_q_heads, self.head_dim
            )
            key_pre = qk_packed0[..., self.latent_q_dim:].view(
                *qk_packed0.shape[:2], self.num_k_heads, self.head_dim
            )
            key_pre = key_pre.unsqueeze(-2).repeat(1, 1, 1, self.gqa_groups, 1) \
                              .view(*qk_packed0.shape[:2], self.num_q_heads, self.head_dim)
            qk_mean_q = (query_pre + key_pre) / 2
            qk_mean_k = qk_mean_q.view(
                *qk_mean_q.shape[:2], self.num_k_heads, self.gqa_groups, -1
            ).mean(dim=-2)

        # # NOTE: V1 puts decode before prefill
        # # Separate prefill and decode by splitting varlen input
        # # Split along token dimension
        qk_packed0_d, qk_packed0_p = torch.split(
            qk_packed0[:num_actual_tokens],
            [num_decodes, num_prefill_tokens],
            dim=0,
        )
        hs_d, hs_p = torch.split(
            hs[:num_actual_tokens],
            [num_decodes, num_prefill_tokens],
            dim=0,
        )

        # Split along batch dimension
        state_indices_tensor_d, state_indices_tensor_p = torch.split(
            state_indices_tensor[:num_actual_tokens],
            [num_decodes, num_prefills],
            dim=0,
        )

        # print(f"MAX INDEX: {state_indices_tensor_d.max()}")

        # TODO: allocate memory for output tensors
        qk_packed3_output_list = []
        hs2_output_list = []
        decode_is_pad: Optional[torch.Tensor] = None

        if has_prefill:
            if _CCA_FUSED_ENABLED:
                _prefill_args = (
                    hs_p, qk_packed0_p,
                    prev_hs, conv_states,
                    query_start_loc_p, has_initial_states_p,
                    state_indices_tensor_p,
                    self.dw_weight_flat,
                    self.conv_qk[0].bias,
                    self.gw_weight_flat,
                    self.gw_bias_flat,
                )
                if cca_prefill_fused_hip_available():
                    hs2, qk_packed3_p = cca_prefill_fused_hip(
                        *_prefill_args)
                else:
                    hs2, qk_packed3_p = cca_prefill_fused(*_prefill_args)
                qk_packed3_output_list.append(qk_packed3_p)
                hs2_output_list.append(hs2)
            else:
                hs2 = torch.zeros((num_prefill_tokens, batch_size, self.hidden_size), device=hs.device, dtype=hs.dtype)
                qk_packed3_p = torch.zeros((num_prefill_tokens, batch_size, self.in_out_ch), device=hs.device, dtype=hs.dtype)
                for i in range(len(query_start_loc_p) - 1):
                    start_i, end_i = query_start_loc_p[i], query_start_loc_p[i + 1]
                    hs2_cur = hs_p[start_i:end_i, :, :] # [S_cur, B, H]
                    qk_packed0_cur = qk_packed0_p[start_i:end_i, :, :]  # [S_cur, B, H]    
                    qk_packed1_cur = qk_packed0_cur.permute(1, 2, 0)  # [1, H, S_cur]
                    
                    if has_initial_states_p[i]:
                        hs2_cached = prev_hs[state_indices_tensor_p[i]].unsqueeze(0).unsqueeze(0) # [1, 1, H]
                        hs2_cur = torch.cat([hs2_cached, hs2_cur[:-1]], dim=0) # [S_cur, 1, H]
                        qk_packed0_cached = conv_states[state_indices_tensor_p[i]].unsqueeze(0) # [1, H, total_padding]
                        qk_packed2_cur = torch.cat([qk_packed0_cached, qk_packed1_cur], dim=-1) # [1, H, S_cur + total_padding]
                    else:
                        hs2_cur = F.pad(hs2_cur[:-1], pad=(0, 0, 0, 0, 1, 0))
                        qk_packed2_cur = F.pad(qk_packed1_cur, (self.total_padding, 0))
                        
                    hs2[start_i:end_i] = hs2_cur

                    conv_states_cur = nn.functional.pad(qk_packed2_cur, (self.cca_time0 - qk_packed2_cur.shape[-1], 0))
                    conv_states[state_indices_tensor_p[i]] = conv_states_cur.to(conv_states.device)
                    
                    # Computing conv
                    qk_packed3_cur = self.conv_qk(qk_packed2_cur).permute(2, 0, 1)  # [S, B, E]
                    qk_packed3_p[start_i:end_i] = qk_packed3_cur
                qk_packed3_output_list.append(qk_packed3_p)
                hs2_output_list.append(hs2)
                prev_hs[state_indices_tensor_p] = hs_p[query_start_loc_p[1:] - 1, 0, :].to(prev_hs.device)

        _fused_decode_active = False
        if has_decode:
            # Generation
            # In generation B and S are actually the same in meaning
            # That's why we don't need to transpose qk_packed0
            # qk_packed0_d [S, 1, H]

            if _CCA_TRITON_FUSION_ENABLED:
                # Fused: pad detection + prev_hs gather/scatter
                hs2_d, decode_is_pad = fused_pad_gather_scatter(
                    state_indices_tensor_d, hs_d, prev_hs,
                )
            else:
                # Handle PAD_SLOT_ID with torch.where for CUDA graph compatibility
                decode_is_pad = (state_indices_tensor_d == PAD_SLOT_ID)
                # block_id=0 reserved for safe indexing (see vllm/v1/core/block_pool.py)
                safe_decode_indices = torch.where(
                    decode_is_pad,
                    torch.zeros_like(state_indices_tensor_d),
                    state_indices_tensor_d,
                )
                hs_d = torch.where(
                    decode_is_pad.view(-1, 1, 1),
                    hs_d.new_zeros(()),
                    hs_d,
                )
                hs2_d = prev_hs[safe_decode_indices].unsqueeze(1) # [S, 1, H]
                hs2_d = torch.where(
                    decode_is_pad.view(-1, 1, 1),
                    hs2_d.new_zeros(()),
                    hs2_d,
                )
                prev_hs[state_indices_tensor_d] = hs_d[:, 0, :].to(prev_hs.device)
            hs2_output_list.insert(0, hs2_d)

            if cca_decode_fused_available():
                # ── Fused decode path ──
                # Single kernel: depthwise conv1d update + grouped conv + qk_mean
                #                + L2 norm + scale + temperature
                b = num_decodes
                dim, _, kernel_width = self.conv_qk[0].weight.shape
                groups = self.num_k_heads + self.num_q_heads
                head_dim = dim // groups

                first_input = qk_packed0_d.transpose(1, 2).contiguous()  # [B, E, 1]

                dw_weight = self.dw_weight_flat
                if self._gw_weight_T is None:
                    gw = self.gw_weight_flat
                    self._gw_weight_T = gw.permute(0, 2, 3, 1).contiguous().view(
                        gw.shape[0], gw.shape[2] * gw.shape[3], gw.shape[1])
                gw_weight = self._gw_weight_T
                gw_bias = self.gw_bias_flat

                # Pack qk_mean for decode tokens: [B, qh+kh, dh]
                qk_mean_q_d = qk_mean_q[:num_decodes].squeeze(1)  # [B, qh, dh]
                qk_mean_k_d = qk_mean_k[:num_decodes].squeeze(1)  # [B, kh, dh]
                qk_mean_packed = torch.cat(
                    [qk_mean_q_d, qk_mean_k_d], dim=1).contiguous()

                fused_out = cca_decode_fused(
                    first_input,
                    dw_weight,
                    self.conv_qk[0].bias,
                    conv_states,
                    state_indices_tensor_d,
                    gw_weight,
                    gw_bias,
                    qk_mean_packed,
                    self.temp.data,
                    self.num_q_heads,
                    self.sqrt_head_dim,
                    self.config.clamp_temp,
                )  # [B, G*dh]

                _fused_query_d = fused_out[:, :self.latent_q_dim]   # [B, qh*dh]
                _fused_key_d   = fused_out[:, self.latent_q_dim:]   # [B, kh*dh]
                _fused_decode_active = True

            else:
                # ── Original unfused decode path ──
                dim, _, kernel_width = self.conv_qk[0].weight.shape
                weights = self.conv_qk[0].weight.reshape(dim, kernel_width).contiguous()

                first_input = qk_packed0_d.transpose(1, 2).contiguous()
                qk_packed3_old = run_causal_conv1d_update(
                    first_input,
                    conv_states,
                    weights,
                    self.conv_qk[0].bias,
                    state_indices_tensor_d,
                    seqlen=1
                )
                groups = self.num_k_heads + self.num_q_heads
                b, d, w = qk_packed3_old.shape
                second_conv_input = qk_packed3_old.reshape(
                    b, groups, d // groups, w).contiguous()
                second_weights = self.conv_qk[1].weight.reshape(
                    groups, d // groups, -1, w).contiguous()
                second_bias = self.conv_qk[1].bias.reshape(groups, -1).contiguous()
                qk_packed3_d = grouped_conv1d_decode(
                    second_conv_input,
                    second_weights,
                    second_bias,
                )
                qk_packed3_d = qk_packed3_d.reshape(b, 1, -1).contiguous()
               # Zero out conv output for padded positions (conv bias adds non-zero values even with zero input)
                qk_packed3_d = torch.where(
                    decode_is_pad.view(-1, 1, 1),
                    qk_packed3_d.new_zeros(()),
                    qk_packed3_d,
                )
                qk_packed3_output_list.insert(0, qk_packed3_d)
        hs2 = torch.vstack(hs2_output_list)[:num_actual_tokens]

        # Values from the two time streams
        v1 = self.val_proj1(hs)   # [S, B, latent_k_dim/2]
        v2 = self.val_proj2(hs2)
        value = torch.cat([v1, v2], dim=-1).contiguous()
        value = value.view(num_actual_tokens, batch_size, self.num_k_heads, self.head_dim)  # [S, B, kh, dh]

        query = torch.empty(
            (num_actual_tokens, self.latent_q_dim),
            device=hs.device,
            dtype=hs.dtype,
        )
        key = torch.empty(
            (num_actual_tokens, self.latent_k_dim),
            device=hs.device,
            dtype=hs.dtype,
        )

        if _fused_decode_active:
            # Fused decode kernel already includes qk_mean, L2 normalization,
            # scaling and temperature. Keep decode path kernel-minimal.
            query[:num_decodes] = _fused_query_d
            key[:num_decodes] = _fused_key_d

            if has_prefill:
                # Only prefill tokens still need Python-side post-processing.
                qk_packed3_pf = qk_packed3_p[..., :].contiguous()
                prefill_query = qk_packed3_pf[..., :self.latent_q_dim].view(
                    num_prefill_tokens, batch_size, self.num_q_heads, self.head_dim
                ) + qk_mean_q[num_decodes:]
                prefill_key = qk_packed3_pf[..., self.latent_q_dim:].view(
                    num_prefill_tokens, batch_size, self.num_k_heads, self.head_dim
                ) + qk_mean_k[num_decodes:]

                prefill_query_norm = prefill_query.norm(p=2, dim=-1, keepdim=True)
                prefill_key_norm = prefill_key.norm(p=2, dim=-1, keepdim=True)

                if self.config.clamp_temp:
                    prefill_key = (
                        prefill_key * (self.sqrt_head_dim / prefill_key_norm)
                    ) * torch.exp(torch.clamp(self.temp[None, None].unsqueeze(-1),
                                              0.0000001, 2))
                else:
                    prefill_key = (
                        prefill_key * (self.sqrt_head_dim / prefill_key_norm)
                    ) * self.temp[None, None].unsqueeze(-1)
                prefill_query = prefill_query * (self.sqrt_head_dim / prefill_query_norm)

                query[num_decodes:] = prefill_query.view(
                    num_prefill_tokens, self.latent_q_dim)
                key[num_decodes:] = prefill_key.view(
                    num_prefill_tokens, self.latent_k_dim)
        else:
            qk_packed3 = torch.vstack(qk_packed3_output_list)[:num_actual_tokens]
            # Build queries/keys from conv output + means
            query_h = qk_packed3[..., :self.latent_q_dim].view(
                num_actual_tokens, batch_size, self.num_q_heads, self.head_dim
            ) + qk_mean_q  # [S, B, qh, dh]
            key_h = qk_packed3[..., self.latent_q_dim:].view(
                num_actual_tokens, batch_size, self.num_k_heads, self.head_dim
            ) + qk_mean_k  # [S, B, kh, dh]

            # L2-normalize per head, then scale like Megatron
            query_norm = query_h.norm(p=2, dim=-1, keepdim=True)
            key_norm = key_h.norm(p=2, dim=-1, keepdim=True)

            # temp: [kh] -> [1,1,kh,1]
            if self.config.clamp_temp:
                key_h = (key_h * (self.sqrt_head_dim / key_norm)) * torch.exp(
                    torch.clamp(self.temp[None, None].unsqueeze(-1), 0.0000001, 2))
            else:
                key_h = (key_h * (self.sqrt_head_dim / key_norm)) * self.temp[
                    None, None].unsqueeze(-1)
            query_h = query_h * (self.sqrt_head_dim / query_norm)

            query = query_h.view(num_actual_tokens, self.latent_q_dim)
            key = key_h.view(num_actual_tokens, self.latent_k_dim)

        value = value.view(num_actual_tokens, batch_size, self.num_k_heads * self.head_dim).transpose(0, 1).contiguous().squeeze(0)

        qkv = torch.cat([query, key, value], dim=1)
        output[:num_actual_tokens] = qkv

    @property
    def mamba_type(self) -> str:
        return "cca"

    def get_attn_backend(self) -> type["AttentionBackend"]:
        from vllm.v1.attention.backends.cca_attn import (
            CCAAttentionBackend)
        return CCAAttentionBackend


def cca(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    if self.use_triton:
        self.forward_triton(hidden_states=hidden_states, output=output)
    else:
        self.forward_cuda(hidden_states=hidden_states, output=output)


def cca_fake(
    hidden_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="cca",
    op_func=cca,
    mutates_args=["output"],
    fake_impl=cca_fake,
)
