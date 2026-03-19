# SPDX-License-Identifier: Apache-2.0
"""Inference-only SMoE model."""
import torch

from vllm.distributed import (get_dp_group,
                              get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)

import os
VLLM_DISABLE_CCA_QKV = os.getenv("VLLM_DISABLE_CCA_QKV", "0") == "1"
VLLM_CCA_TRITON = os.getenv("VLLM_CCA_TRITON", "1") == "1"

from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import torch
from torch import nn

from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.mamba.cca import CCA
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.mamba.ops import cca_decode_fused_available
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors
from vllm.model_executor.layers.layernorm import RMSNorm

from .interfaces import HasInnerState, IsHybrid
from .utils import (make_empty_intermediate_tensors_factory, maybe_prefix)
from vllm.transformers_utils.configs import SMoEConfig

import logging
logger = logging.getLogger(__name__)

class ResidualScaling(nn.Module):
    def __init__(
        self,
        config,
        layer_n,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",        
    ):
        super().__init__()
        self.config          = config
        self.not_first_layer = (layer_n != 0)
        self.hidden_states_scale = torch.nn.Parameter(torch.ones(self.config.hidden_size))
        self.hidden_states_bias = torch.nn.Parameter(torch.zeros(self.config.hidden_size))
        
        if self.not_first_layer:
            self.residual_scale = torch.nn.Parameter(torch.ones(self.config.hidden_size))
            self.residual_bias = torch.nn.Parameter(torch.zeros(self.config.hidden_size))

    def forward(self, residual: torch.Tensor, hidden_states: torch.Tensor):
        hidden_states = (hidden_states + self.hidden_states_bias.expand(1, -1)) * self.hidden_states_scale.expand(1, -1)
        if self.not_first_layer:
                residual = (residual + self.residual_bias.expand(1, -1)) * self.residual_scale.expand(1, -1)
        return residual, hidden_states


class SMoEAttention(nn.Module):
    def __init__(
        self,
        config: SMoEConfig,
        layer_idx,
        layer_n,
        prefix_name: str = "",
        model_config: Optional[ModelConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):

        super().__init__()
        # tp_size = get_tensor_model_parallel_world_size()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_n = layer_n
        self.hidden_size = config.hidden_size
        # self.total_num_heads = config.num_attention_heads
        # self.num_key_value_heads = config.num_key_value_heads
        # self.num_key_value_groups = self.num_heads // self.num_key_value_heads        
        # assert self.total_num_heads % tp_size == 0
        # self.head_dim = self.hidden_size // self.total_num_heads
        # self.num_heads = self.total_num_heads // tp_size
        # self.qkv_size = self.hidden_size // tp_size
        self.attention_dropout = config.attention_dropout

        # TODO: clean up the config
        self.cca_num_k_heads = 2
        self.cca_num_q_heads = 8
        self.cca_num_heads = 16
        self.cca_time0 = 2
        self.cca_time1 = 2
        self.head_dim = self.hidden_size // self.cca_num_heads
        self.scale = self.head_dim**-0.5

        # if (self.head_dim * self.total_num_heads) != self.hidden_size:
        #     raise ValueError(
        #         f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
        #         f" and `num_heads`: {self.total_num_heads})."
        #     )
        
        self.qkv = CCA(
            config=config,
            cca_num_k_heads=self.cca_num_k_heads,
            cca_num_q_heads=self.cca_num_q_heads,
            cca_num_heads=self.cca_num_heads,
            hidden_size=self.hidden_size,
            cca_time0=self.cca_time0,
            cca_time1=self.cca_time1,
            layer_number=layer_n,
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            use_triton=VLLM_CCA_TRITON,
            prefix=f"{prefix_name}.cca",
        )
        self.o_proj = ReplicatedLinear(self.cca_num_q_heads * self.head_dim,
                                       self.hidden_size,
                                       bias=self.config.attention_bias,
                                       quant_config=quant_config,
                                       return_bias=False,
                                       prefix=f"{prefix_name}.o_proj")            

        self.attn = Attention(
            self.cca_num_q_heads,
            self.head_dim,
            self.scale,
            self.cca_num_k_heads,
            cache_config=cache_config,
            prefix=f"{prefix_name}.attn",
        )

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            is_neox_style=True,
            rope_parameters={
                "rope_theta": config.rotary_base,
                "rope_type": "default",
                "partial_rotary_factor": 0.5,
            },
        )

        self.q_dim = self.cca_num_q_heads * self.head_dim
        self.k_dim = self.cca_num_k_heads * self.head_dim
        self.v_dim = self.cca_num_k_heads * self.head_dim
        self.qkv_dim = self.q_dim + self.k_dim + self.v_dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        output_qkv = torch.zeros((hidden_states.shape[0], self.qkv_dim), device=hidden_states.device, dtype=hidden_states.dtype)
        if not VLLM_DISABLE_CCA_QKV:
            self.qkv(hidden_states, output_qkv)
        q, k, v = output_qkv.split([self.q_dim, self.k_dim, self.v_dim], dim=-1)
        q, k = self.rotary_emb(position_ids, q, k)
        attn_output = self.attn(q, k, v)     
        attn_output = self.o_proj(attn_output)     

        return attn_output

class SMoEDecoderATTLayer(nn.Module):
    def __init__(
        self,
        config: SMoEConfig,
        layer_idx: str,
        layer_n: int,
        prefix_name = "",
        model_config: Optional[ModelConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,      
    ):

        super().__init__()
        self.config    = config
        self.layer_n   = layer_n
        self.training  = self.training
        self.self_attn = SMoEAttention(
            config,
            layer_idx,
            layer_n,
            prefix_name,
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
        )

        if (config.normalization == "RMSNorm"):
            self.input_norm = RMSNorm(self.config.hidden_size, eps=config.norm_epsilon)
        elif (config.normalization == "LayerNorm"):
            self.input_norm = nn.LayerNorm(self.config.hidden_size, eps=config.norm_epsilon)
        else:
            raise TypeError("Normalization not supported.")

        if self.config.scale_residual_merge:
            self.res_scale = ResidualScaling(config, layer_n)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        position_ids: torch.LongTensor,
        layer_n: int,
        prev_router_hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        if self.config.scale_residual_merge:
            residual, hidden_states = self.res_scale(residual, hidden_states)      
        if residual is not None:
            residual += hidden_states
            hidden_states = residual
        else:
            residual = hidden_states
        hidden_states = self.input_norm(residual)
        
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
        )

        return hidden_states, residual, prev_router_hidden_states


class SMoERouter(nn.Module):
    """
    Hugging Face/Transformers-style port of the Megatron `SMoETopKRouter`.

    Key features replicated from Megatron code:
      - Down-projection to `mlp_expansion` then RMSNorm.
      - Optional EDA (depth-wise averaging) via `router_states_scale` and prior `router_states`.
      - Three-layer MLP with GELU producing `num_experts` logits per token.
      - Top-k expert selection with balancing biases and MOD (skip expert) handling.

    Returns (route_prob, expert_choice_t, router_hidden_states_next) where:
      - route_prob:  (batch*seq, topk) gathered probabilities of chosen experts
      - expert_choice_t: (batch*seq, topk) chosen expert indices (with MOD post-processing)
      - router_hidden_states_next: the pre-norm router hidden states (B, S, mlp_expansion),
        for feeding forward to the MoE layer as in Megatron.
    """

    def __init__(
        self,
        config,
        layer_n: int,
        num_moe_experts: int,
        moe_router_topk: int,
        mlp_expansion: int,
        hidden_size: Optional[int] = None,
        layer_number: Optional[int] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        # ---- Config / shape ----
        self.config = config
        self.layer_n = layer_n
        self.hidden_size = int(hidden_size or getattr(config, "hidden_size"))
        self.layer_number = layer_number if layer_number is not None else 0

        # MOD 
        self.use_mod = bool(getattr(config, "smoe_use_mod", False))
        self.mod_per = int(getattr(config, "smoe_mod_per", 0))
        if (self.mod_per == 0) and (num_moe_experts == 1):
            raise ValueError(
                "ERROR! The only way in which we can have a single expert is if MOD is enabled."
            )

        # Expert counts (extra 'skip' expert if MOD)
        self.num_experts = (num_moe_experts + 1) if self.use_mod else num_moe_experts
        self.topk = int(moe_router_topk)

        # Router hidden dim
        self.mlp_expansion = int(mlp_expansion)

        # ---- Layers ----
        self.down_proj = ReplicatedLinear(self.hidden_size, self.mlp_expansion, bias=True, quant_config=quant_config, return_bias=False)

        # EDA (depth-wise averaging) 
        smoe_first_layer = 1
        use_eda_cfg = bool(getattr(config, "smoe_use_eda", False))
        self.use_eda = use_eda_cfg and (smoe_first_layer is not None) and (self.layer_number != smoe_first_layer)

        ln_eps = float(getattr(config, "layernorm_epsilon", 1e-6))
        self.rmsnorm_eda = RMSNorm(self.mlp_expansion, eps=ln_eps)
        if self.use_eda:
            # eda
            self.router_states_scale = nn.Parameter(torch.ones(self.mlp_expansion))

        #routermlp
        D = self.mlp_expansion
        E = self.num_experts
        self.non_linearity = nn.GELU()
        self.router_mlp = nn.Sequential(
            ReplicatedLinear(D, D, bias=True, quant_config=quant_config, return_bias=False),
            self.non_linearity,
            ReplicatedLinear(D, D, bias=True, quant_config=quant_config, return_bias=False),
            self.non_linearity,
            ReplicatedLinear(D, E, bias=False, quant_config=quant_config, return_bias=False),
        )

        # Balancing biases
        self.register_buffer("balancing_biases", torch.zeros(self.num_experts, dtype=torch.float32))
        if self.use_mod:
            self.balancing_biases[-1] = -1.0


    def forward(
        self,
        hidden_states: torch.Tensor,               # (B, S, H)
        prev_router_hidden_states: Optional[torch.Tensor] = None,  # (B, S, D) previous router states for EDA
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute per-token expert probabilities and choose top-k experts.

        Args:
            hidden_states: (batch, seq, hidden_size)
            router_states: (batch, seq, mlp_expansion) from prior step/layer (for EDA). Optional.
            eos_mask: kept for API compatibility; not used.

        Returns:
            route_prob: (batch*seq, topk)
            expert_choice_t: (batch*seq, topk) int64
            router_hidden_states_next: (batch, seq, mlp_expansion)
        """
        S, _ = hidden_states.shape
        device = hidden_states.device

        # eda
        hs = self.down_proj(hidden_states)
        if self.use_eda and (prev_router_hidden_states is not None):
            hs = hs + prev_router_hidden_states * self.router_states_scale

        # Stash the pre-norm states for the caller (this is what Megatron returns)
        router_hidden_states_next = hs[-S:].clone()

        # 2) RMSNorm eda
        hs_norm = self.rmsnorm_eda(hs)

        # 3) Expert probability distribution
        logits = self.router_mlp(hs_norm)
        expert_prob = torch.softmax(logits, dim=-1)

        # 4) Expert choice with balancing biases (biases affect choice only, not probabilities)
        biased = expert_prob.detach().to(torch.float32) + self.balancing_biases
        _, expert_choice_t = torch.topk(biased, self.topk, dim=-1)  # (S, topk)

        # 5) If MOD and topk>1, once skip expert is selected, force all subsequent choices to skip as well, but this never happens since we use topk=1
        if (self.topk > 1) and self.use_mod:
            skip_idx = self.num_experts - 1
            n_mask = (expert_choice_t == skip_idx)
            cumsum_mask = torch.cumsum(n_mask, dim=-1)
            expert_choice_t = expert_choice_t.masked_fill(cumsum_mask > 0, skip_idx)

        # Gather the probabilities for the selected experts
        route_prob = torch.gather(expert_prob, dim=1, index=expert_choice_t)
        
        expert_choice_flat = expert_choice_t.reshape(-1, self.topk)
        route_prob_flat = route_prob.reshape(-1, self.topk)

        return route_prob_flat, expert_choice_flat, router_hidden_states_next


class SMoEBlock(nn.Module):

    def __init__(
        self,
        config: SMoEConfig,
        layer_idx: int,
        mlp_expansion: int,
        ffn_hidden_size: int,
        layer_n: int,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):

        super().__init__()
        self.config = config
        self.layer_n = layer_n
        self.hidden_dim = config.hidden_size
        self.num_moe_experts = layer_idx
        self.mlp_expansion = mlp_expansion

        assert config.activation_func == "swiglu", "Only SwiGLU activation is supported"
        assert config.gated_linear_unit, "gated_linear_unit must be True"
        assert not config.add_bias_linear, "add_bias_linear must be False"

        self.router = SMoERouter(
            config=self.config,
            layer_n=layer_n,
            num_moe_experts=self.num_moe_experts,
            moe_router_topk=getattr(self.config, "moe_router_topk", 1),
            mlp_expansion=self.mlp_expansion,
            hidden_size=self.hidden_dim,
            layer_number=layer_n,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.router",
        )

        self.topk = getattr(self.config, "moe_router_topk", 1)

        self.tp_size = get_tensor_model_parallel_world_size()
        if self.tp_size > self.num_moe_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        def _custom_routing_fn(hidden_states, gating_output, topk, renormalize):
            # Routing results are packed into gating_output by forward():
            # columns [:topk] = weights (float), columns [topk:] = ids (float-cast)
            topk_weights = gating_output[:, :topk]
            topk_ids = gating_output[:, topk:2 * topk].to(torch.int64)
            return topk_weights, topk_ids

        self.experts = FusedMoE(
            num_experts=self.num_moe_experts,
            top_k=self.topk,
            hidden_size=config.hidden_size,
            intermediate_size=ffn_hidden_size // 2,
            reduce_results=False,
            renormalize=False,
            custom_routing_function=_custom_routing_fn,
            activation="silu",
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_router_hidden_states: Optional[torch.Tensor] = None,
    ):

        probs, indices, router_hidden_states_out = self.router(
            hidden_states,
            prev_router_hidden_states=prev_router_hidden_states,
        )

        if self.config.smoe_use_mod and not getattr(self.config, "ignore_mod_in_smoe_block", False):
            clamped_indices = torch.clamp(indices,
                                          min=0,
                                          max=self.num_moe_experts - 1)
            packed_logits = torch.cat([
                probs, clamped_indices.to(probs.dtype)], dim=-1)
            hidden_states_experts = self.experts(
                hidden_states, packed_logits)
            hidden_states_mod = hidden_states * probs
            mod_mask = (indices != self.num_moe_experts)
            hidden_states = (mod_mask * hidden_states_experts) + ((~mod_mask) * hidden_states_mod)
        else:
            packed_logits = torch.cat([
                probs, indices.to(probs.dtype)], dim=-1)
            hidden_states = self.experts(
                hidden_states, packed_logits)

        if self.tp_size > 1:
            hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(hidden_states)

        return hidden_states, router_hidden_states_out


class SMoEDecoderMLPLayer(nn.Module):
    def __init__(
        self,
        config: SMoEConfig,
        layer_idx: int,
        mlp_expansion: int,
        ffn_hidden_size: int,
        layer_n: int,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",        
    ):

        super().__init__()
        self.config     = config
        self.layer_n    = layer_n
        self.smoe_block = SMoEBlock(
            config,
            layer_idx,
            mlp_expansion,
            ffn_hidden_size,
            layer_n,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,            
        )

        if (config.normalization == "RMSNorm"):
            self.input_norm = RMSNorm(self.config.hidden_size, eps=config.norm_epsilon)
        elif (config.normalization == "LayerNorm"):
            self.input_norm = nn.LayerNorm(self.config.hidden_size, eps=config.norm_epsilon)
        else:
            raise TypeError("Normalization not supported.")

        if self.config.scale_residual_merge:
            self.res_scale = ResidualScaling(config, layer_n)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,      
        position_ids: torch.LongTensor,
        layer_n: int,
        prev_router_hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        if self.config.scale_residual_merge:
            residual, hidden_states = self.res_scale(residual, hidden_states)               
        if residual is not None:
            residual += hidden_states
            hidden_states = residual
        else:
            residual = hidden_states
        hidden_states = self.input_norm(hidden_states)  

        hidden_states, prev_router_hidden_states = self.smoe_block(hidden_states, prev_router_hidden_states)

        return hidden_states, residual, prev_router_hidden_states

@support_torch_compile
class SMoEModel(nn.Module):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer can be an attention layer CMoEDecoderATTLayer or an MLP layer CMoEDecoderMLPLayer.
    Args:
        config: CMoEConfig
    """
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:

        super().__init__()

        config: SMoEConfig = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        is_lora_enabled = bool(lora_config)
        assert not is_lora_enabled

        self.config = config
        lora_vocab = ((lora_config.lora_extra_vocab_size *
                       (lora_config.max_loras or 1)) if lora_config else 0)
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size

        self.config               = config
        self.padding_idx          = config.pad_token_id
        self.vocab_size           = config.vocab_size
        self.layers               = []

        # Initialize token embeddings
        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )

        for layer_n in range(len(config.smoe_layers)):
            if isinstance(config.smoe_layers[layer_n], int):
                prefix_name = f"{prefix}.layers.{layer_n}.moe"
                self.layers.append(
                    SMoEDecoderMLPLayer(
                        config,
                        config.smoe_layers[layer_n],
                        config.smoe_mlp_expansion[layer_n],
                        config.ffn_hidden_size_list[layer_n],
                        layer_n,
                        cache_config=cache_config,
                        quant_config=quant_config,
                        prefix=prefix_name,
                    )
                )
            else:
                prefix_name = f"{prefix}.layers.{layer_n}.self_attn"
                self.layers.append(
                    SMoEDecoderATTLayer(
                        config,
                        config.smoe_layers[layer_n],
                        layer_n,
                        prefix_name,
                        model_config=model_config,
                        cache_config=cache_config,
                        quant_config=quant_config,
                    )
                )
        self.layers = nn.ModuleList(self.layers)

        if self.config.scale_residual_merge:
            self.res_scale = ResidualScaling(config, len(config.smoe_layers))

        if (config.normalization == "RMSNorm"):
            self.final_norm = RMSNorm(self.config.hidden_size, eps=config.norm_epsilon)
        elif (config.normalization == "LayerNorm"):
            self.final_norm = nn.LayerNorm(self.config.hidden_size, eps=config.norm_epsilon)
        else:
            raise TypeError("Normalization not supported.")

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))


    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convert input token IDs to embeddings.
        
        Args:
            input_ids: Tensor of input token IDs
            
        Returns:
            Embedded representation of the input tokens
        """
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        residual = None
        hidden_states = inputs_embeds
        prev_router_hidden_states = None

        for layer_n, decoder_layer in enumerate(self.layers):
            hidden_states, residual, prev_router_hidden_states = decoder_layer(
                hidden_states,
                residual,
                positions,
                layer_n,
                prev_router_hidden_states,
            )

        if self.config.scale_residual_merge:
            residual, hidden_states = self.res_scale(residual, hidden_states)
        if residual is not None:
            hidden_states += residual
        hidden_states = self.final_norm(hidden_states)
    
        return hidden_states


class SMoEForCausalLM(nn.Module, HasInnerState, IsHybrid):

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.cca_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_config

        # TODO: make these configurable
        conv_kernel_size = 2
        num_k_heads = 2
        num_q_heads = 8
        head_dim = 128
        hidden_size = hf_config.hidden_size

        return MambaStateShapeCalculator.cca_state_shape(
            tp_world_size=parallel_config.tensor_parallel_size,
            conv_kernel_size=conv_kernel_size,
            num_k_heads=num_k_heads,
            num_q_heads=num_q_heads,
            head_dim=head_dim,
            hidden_size=hidden_size,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.cca_state_copy_func()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        lora_config = vllm_config.lora_config
        scheduler_config = vllm_config.scheduler_config
        assert config.moe_router_topk == 1, "Only topk=1 is supported in SMoE!"
        if config.moe_router_topk != 1:
            raise NotImplementedError(
                "SMoE currently only supports topk=1, "
                "please use '--moe-router-topk=1' instead"
            )
        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "SMoE currently does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )

        if VLLM_CCA_TRITON:
            logger.info("[SMoE]: Using Triton CCA backend")
            if cca_decode_fused_available():
                logger.info("[SMoE]: Using CCA fused for decode")
        else:
            logger.info("[SMoE]: Using pytorch-native CCA backend")

        if VLLM_DISABLE_CCA_QKV:
            logger.warning(
                "[SMoE]: WARNING! Skipping QKV computation in CCA, "
                "QKV will be just filled with zeros. This is only used "
                "for debugging/benchmarking purposes."
            )

        super().__init__()
        self.config = config
        self.lora_config = lora_config
        self.scheduler_config = scheduler_config
        self.vllm_config = vllm_config
        self.quant_config = vllm_config.quant_config
        self.model_config = vllm_config.model_config
        if config.smoe_use_mod and getattr(config, "ignore_mod_in_smoe_block", False):
            logger.warning(
                "[SMoE]: WARNING! SMoE is using MoD but ignoring it in SMoE blocks! "
                "Please, double check if this is intended. "
                "Some SMoE checkpoints technically have MoD, but MoD bias is set to -1, "
                "so the MoD expert is never selected. In this case, setting "
                "ignore_mod_in_smoe_block=True can improve inference speed."
            )

        self.model = SMoEModel(vllm_config=vllm_config,
                               prefix=maybe_prefix(prefix, "model"))
        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=(
                DEFAULT_VOCAB_PADDING_SIZE
                # We need bigger padding if using lora for kernel
                # compatibility
                if not lora_config else lora_config.lora_vocab_padding_size),
            quant_config=None,
            bias=config.lm_head_bias,
        )
        # Tie weights with input embeddings if using same dimensions
        if self.config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)

        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, IntermediateTensors]:

        hidden_states = self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Compute logits for next token prediction.

        Args:
            hidden_states: Hidden states from model forward pass

        Returns:
            Logits for next token prediction
        """
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:

        params_dict = dict(self.named_parameters())
        buffers_dict = dict(self.named_buffers())
        for key, buffer in buffers_dict.items():
            if "cos_sin_cache" in key:
                continue
            params_dict[key] = buffer

        weights_dict = {}
        for key, loaded_weight in weights:
            if "lora" in key:
                if "_A.weight" in key:
                    key = key.replace("_A.weight", ".A.weight")
                elif "_B.weight" in key:
                    key = key.replace("_B.weight", ".B.weight")
            weights_dict[key] = loaded_weight

        # Build a map from prefix → FusedMoE module for expert weight loading
        fused_moe_modules: Dict[str, FusedMoE] = {}
        for name, module in self.named_modules():
            if isinstance(module, FusedMoE):
                fused_moe_modules[name] = module

        loaded_params: Set[str] = set()
        import tqdm
        import re
        tp_rank = get_tensor_model_parallel_rank()
        disable_tqdm = tp_rank != 0
        for chkpt_weight_name, loaded_weight in tqdm.tqdm(weights_dict.items(), desc="Loading weights", unit_scale=True, unit="weights", disable=disable_tqdm):
            if "local_experts" in chkpt_weight_name:
                parts = chkpt_weight_name.split(".")

                m = re.search(r"\.local_experts\.(\d+)\.", chkpt_weight_name)
                if not m:
                    raise ValueError(f"Could not parse expert id from {chkpt_weight_name}")
                expert_id = int(m.group(1))

                # Determine FusedMoE param name and shard_id
                # linear_fc1 = merged gate+up → w13_weight (split into w1, w3)
                # linear_fc2 = down proj → w2_weight (shard_id w2)
                fused_moe_prefix = ".".join(parts[:5])
                fused_moe_module = fused_moe_modules.get(fused_moe_prefix)
                if fused_moe_module is None:
                    logger.warning("No FusedMoE module found at %s, skipping %s",
                                   fused_moe_prefix, chkpt_weight_name)
                    continue

                if parts[-2] == "linear_fc1":
                    param_name = f"{fused_moe_prefix}.w13_weight"
                    param = params_dict[param_name]
                    half = loaded_weight.shape[0] // 2
                    gate_weight = loaded_weight[:half, :]
                    up_weight = loaded_weight[half:, :]
                    fused_moe_module.weight_loader(
                        param, gate_weight, chkpt_weight_name, "w1", expert_id)
                    fused_moe_module.weight_loader(
                        param, up_weight, chkpt_weight_name, "w3", expert_id)
                    loaded_params.add(param_name)
                elif parts[-2] == "linear_fc2":
                    param_name = f"{fused_moe_prefix}.w2_weight"
                    param = params_dict[param_name]
                    fused_moe_module.weight_loader(
                        param, loaded_weight, chkpt_weight_name, "w2", expert_id)
                    loaded_params.add(param_name)
                else:
                    logger.warning("Unknown expert weight kind in %s", chkpt_weight_name)
                continue
               
            # Loading other parameters
            if chkpt_weight_name not in params_dict:
                logger.info(f"WARNING: key {chkpt_weight_name} not in params! Skipping loading")
                continue
            param = params_dict[chkpt_weight_name]
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(chkpt_weight_name)            
        return loaded_params
