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

_CCA_AMD_CONV_UNFOLD_ENABLED = os.environ.get(
    "VLLM_CCA_AMD_CONV_UNFOLD", "0").lower() in ("1", "true", "yes")

_CCA_DIM_PRESERVE_CONV_ENABLED = os.environ.get(
    "CCA_DIM_PRESERVE_CONV", "0").lower() in ("1", "true", "yes")

@CustomOp.register("cca")
class CCA(MambaBase, CustomOp):
    # Monotonic per-engine-step sequence number, bumped by the model runner
    # before each execute_model. The eager prefill loop tags its sparse
    # stash with the current value so the runner can tell a fresh stash
    # from a stale one (class attr: one write per step covers all layers).
    _tidar_step_seq: int = -1

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

        self._gw_weight_T = None
        self._conv_qk_fp32_cache: list[
            tuple[torch.Tensor, torch.Tensor | None]] | None = None
        self._conv_qk_fp32_versions: tuple[int, ...] | None = None
        self.refresh_runtime_weight_views()

        # Per-k head temperature (Megatron: shape [num_k_heads])
        self.temp = nn.Parameter(torch.zeros(self.num_k_heads))
        self._temp_fp32_cache: torch.Tensor | None = None
        self._temp_fp32_key: tuple[int, int] | None = None

        if _CCA_DIM_PRESERVE_CONV_ENABLED:
            self._decode_conv = self.conv_qk_dim_preserve
        else:
            self._decode_conv = self._conv_qk_apply

        vllm_cfg = get_current_vllm_config()
        compilation_config = vllm_cfg.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        self.kv_cache = (torch.tensor([]), torch.tensor([]))

        # === TiDAR spec-decode stash buffers ====================
        # Pre-allocated GPU buffers for K+1 candidate (conv_state, hs)
        # pairs per request. The forward path writes via pure GPU ops
        # so cudagraph replay sees fresh values; commit_spec_decode_state
        # consumes them outside the captured graph after rejection
        # sampling, writing the post-acceptance candidate back to
        # conv_states / prev_hs.
        spec = vllm_cfg.speculative_config
        if (spec is not None and getattr(spec, "use_tidar", lambda: False)()
                and spec.num_speculative_tokens is not None):
            self._spec_max_P = vllm_cfg.scheduler_config.max_num_seqs
            self._spec_max_S = spec.num_speculative_tokens + 1
            stash_dtype = vllm_cfg.model_config.dtype
            stash_device = torch.device(
                f"cuda:{torch.cuda.current_device()}"
                if torch.cuda.is_available() else "cpu")
            self._spec_stash_conv = torch.zeros(
                (self._spec_max_P, self._spec_max_S, self.in_out_ch,
                 self.cca_time0),
                dtype=stash_dtype, device=stash_device)
            self._spec_stash_hs = torch.zeros(
                (self._spec_max_P, self._spec_max_S, self.hidden_size),
                dtype=stash_dtype, device=stash_device)
            self._spec_stash_slots = torch.zeros(
                (self._spec_max_P, ), dtype=torch.int64,
                device=stash_device)
        else:
            self._spec_max_P = 0
            self._spec_max_S = 0
            self._spec_stash_conv = None
            self._spec_stash_hs = None
            self._spec_stash_slots = None

        # Eager-fallback stash bookkeeping (mixed prefill/decode steps).
        # When the prefill segment is NOT a uniform K+1 verify batch (e.g.
        # a cold prompt prefill co-scheduled with TiDAR verify rows), the
        # vectorized path is skipped and the Python prefill loop stashes
        # spec-sized rows sparsely instead. These attributes record which
        # step (runner-provided ``CCA._tidar_step_seq``) the sparse stash
        # belongs to and which prefill-segment rows were stashed, so the
        # runner can (a) detect staleness and (b) pair stash rows with the
        # right requests. Python attrs are safe here: the eager loop never
        # runs inside a captured cudagraph.
        self._spec_stash_eager_seq = -1
        self._spec_stash_eager_rows: list = []

        # Whether the captured-friendly vectorized prefill path is REQUIRED
        # this run. True when:
        #   (a) FULL_DECODE_ONLY (or FULL_AND_PIECEWISE) cudagraph mode is
        #       enabled — the eager Python loop is cudagraph-unsafe, OR
        #   (b) TiDAR is active — the vectorized path stashes K+1 conv
        #       state candidates per req, which commit_spec_decode_state
        #       needs to write the post-acceptance state back to AR.
        #       The Python loop doesn't stash, so under TF mode (where SF
        #       inflation doesn't run) the state would be over-advanced by
        #       (K - num_accepted) positions and the next forward gets
        #       garbage → token loop degeneration after ~25 tokens.
        # Under PIECEWISE non-spec the eager loop is still faster (no
        # stash overhead) and self._spec_stash_conv is None there anyway.
        from vllm.config import CUDAGraphMode
        _cg_mode = getattr(compilation_config, "cudagraph_mode", None)
        _full_active = (_cg_mode is not None
                        and _cg_mode != CUDAGraphMode.NONE
                        and _cg_mode.decode_mode() == CUDAGraphMode.FULL)
        self._use_capture_vectorized = bool(_full_active)
        # _spec_stash_conv is non-None whenever TiDAR is active (SF or TF).
        # We need vectorized path whenever stash is needed — that's any
        # TiDAR run, because commit_spec_decode_state always wants the
        # stashed candidate state.
        self._use_spec_vectorized = bool(
            self._spec_stash_conv is not None
        )

        if os.environ.get("PATCH_PROBE_LAYER_HASH", "0").lower() in (
                "1", "true", "yes"):
            probe_rows = int(os.environ.get("PATCH_PROBE_LAYER_ROWS", "17"))
            probe_dtype = (model_config.dtype if model_config is not None
                           else torch.get_default_dtype())
            for name, width in (
                ("_patch_probe_qk_linear", self.in_out_ch),
                ("_patch_probe_qk_conv", self.in_out_ch),
                ("_patch_probe_value", self.latent_k_dim),
                ("_patch_probe_qk_cached",
                 self.in_out_ch * self.total_padding),
                ("_patch_probe_hs_cached", self.hidden_size),
            ):
                self.register_buffer(
                    name,
                    torch.empty(probe_rows, width, dtype=probe_dtype),
                    persistent=False,
                )

    def _conv_qk_apply(self, x: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.to(torch.float32)

        versions = tuple(
            value
            for module in self.conv_qk
            for value in (
                module.weight.data_ptr(),
                module.weight._version,
                -1 if module.bias is None else module.bias.data_ptr(),
                -1 if module.bias is None else module.bias._version,
            )
        )
        if (self._conv_qk_fp32_cache is None
                or self._conv_qk_fp32_versions != versions):
            self._conv_qk_fp32_cache = [
                (
                    module.weight.detach().to(torch.float32).contiguous(),
                    None if module.bias is None else
                    module.bias.detach().to(torch.float32).contiguous(),
                )
                for module in self.conv_qk
            ]
            self._conv_qk_fp32_versions = versions

        def _conv1d_fp32(
            module: nn.Conv1d,
            params: tuple[torch.Tensor, torch.Tensor | None],
            inp: torch.Tensor,
        ) -> torch.Tensor:
            weight, bias = params
            return F.conv1d(
                inp,
                weight,
                bias,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
            )

        assert self._conv_qk_fp32_cache is not None
        return _conv1d_fp32(
            self.conv_qk[1],
            self._conv_qk_fp32_cache[1],
            _conv1d_fp32(
                self.conv_qk[0], self._conv_qk_fp32_cache[0], x_fp32),
        )

    @torch.no_grad()
    def refresh_runtime_weight_views(self) -> None:
        """Refresh cached conv views after in-place runtime weight reloads."""
        dim, _, kernel_width = self.conv_qk[0].weight.shape
        groups = self.num_k_heads + self.num_q_heads
        head_dim = dim // groups
        self.dw_weight_flat = self.conv_qk[0].weight.reshape(
            dim, kernel_width
        ).contiguous()
        self.gw_weight_flat = self.conv_qk[1].weight.reshape(
            groups, head_dim, -1, kernel_width
        ).contiguous()
        self.gw_bias_flat = (
            None
            if self.conv_qk[1].bias is None
            else self.conv_qk[1].bias.reshape(groups, -1).contiguous()
        )
        self._gw_weight_T = None
        self._conv_qk_fp32_cache = None
        self._conv_qk_fp32_versions = None

    def _rms_normalize_qk(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        target_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eps = 1e-12
        sqrt_head_dim = float(self.sqrt_head_dim)

        query_fp32 = query.to(torch.float32)
        q_norm = query_fp32.norm(p=2, dim=-1, keepdim=True)
        query_out = (
            query_fp32 * torch.rsqrt(q_norm * q_norm + eps) * sqrt_head_dim
        ).to(target_dtype)

        key_fp32 = key.to(torch.float32)
        k_norm = key_fp32.norm(p=2, dim=-1, keepdim=True)
        temp_key = (self.temp.data_ptr(), self.temp._version)
        if self._temp_fp32_cache is None or self._temp_fp32_key != temp_key:
            self._temp_fp32_cache = (
                self.temp.detach().to(torch.float32).contiguous())
            self._temp_fp32_key = temp_key
        temp = self._temp_fp32_cache.view(1, self.num_k_heads, 1)
        if self.config.clamp_temp:
            temp = torch.exp(torch.clamp(temp, 1e-7, 2.0))
        key_out = (
            key_fp32 * torch.rsqrt(k_norm * k_norm + eps) * sqrt_head_dim * temp
        ).to(target_dtype)

        return query_out, key_out

    def conv_qk_dim_preserve(self, x: torch.Tensor) -> torch.Tensor:
        """Decode-time conv path that avoids layout-heavy cuDNN transforms."""
        w0 = self.conv_qk[0].weight.squeeze(1)
        b0 = self.conv_qk[0].bias

        kernel0 = w0.shape[1]
        seq = x.shape[2]
        mid_len = seq - kernel0 + 1

        mid = x[:, :, :mid_len] * w0[:, 0:1]
        for k in range(1, kernel0):
            mid = mid + x[:, :, k:k + mid_len] * w0[:, k:k + 1]
        if b0 is not None:
            mid = mid + b0[:, None]

        w1 = self.conv_qk[1].weight
        b1 = self.conv_qk[1].bias
        groups = self.num_k_heads + self.num_q_heads
        head_dim = self.head_dim
        batch = x.shape[0]

        mid_g = mid.view(batch, groups, head_dim, mid_len)
        w1_g = w1.view(groups, head_dim, head_dim, w1.shape[2])
        out_g = torch.einsum("godk,sgdk->sgo", w1_g.float(), mid_g.float()).to(
            mid.dtype
        )
        if b1 is not None:
            out_g = out_g + b1.view(1, groups, head_dim)

        return out_g.reshape(batch, groups * head_dim, 1)

    # ================================================================
    # TiDAR spec-decode helpers (ported from jinzhao/tidar v0.15.x)
    # ================================================================
    def _conv_qk_decode(self, x: torch.Tensor) -> torch.Tensor:
        """Manual conv_qk for decode-sized inputs.

        Runs in fp32 (same as _conv_qk_apply) to keep CCA acceptance
        rates matching the eager Python-loop path. The Python-loop
        path uses _conv_qk_apply (fp32), so this method must too —
        bf16 here introduced ~37% mean-acceptance regression vs eager
        on AIME25 thinking-off.

        Input:  [N, C, S]
        Output: [N, C, S_out]
        """
        orig_dtype = x.dtype
        # Stage 1: depthwise conv over sequence.
        w0 = self.conv_qk[0].weight.squeeze(1).to(torch.float32)  # [C, K0]
        b0 = self.conv_qk[0].bias
        b0_fp32 = b0.to(torch.float32) if b0 is not None else None

        x = x.to(torch.float32)
        k0 = w0.shape[1]
        x_windows = x.unfold(-1, k0, 1)
        mid = (x_windows * w0[:, None, :]).sum(dim=-1)
        if b0_fp32 is not None:
            mid = mid + b0_fp32[None, :, None]

        # Stage 2: grouped conv over the depthwise output.
        w1 = self.conv_qk[1].weight.to(torch.float32)
        b1 = self.conv_qk[1].bias
        b1_fp32 = b1.to(torch.float32) if b1 is not None else None
        g = self.num_k_heads + self.num_q_heads
        d = self.head_dim
        k1 = w1.shape[2]
        mid_windows = (mid.view(mid.shape[0], g, d, mid.shape[-1])
                       .unfold(-1, k1, 1))
        w1_grouped = w1.view(g, d, d, k1)
        out = torch.einsum("godk,sgdtk->sgot", w1_grouped, mid_windows)
        if b1_fp32 is not None:
            out = out + b1_fp32.view(1, g, d, 1)
        return out.reshape(x.shape[0], g * d, out.shape[-1]).to(orig_dtype)

    def _spec_decode_prefill_vectorized(
        self,
        hs_p: torch.Tensor,                   # [P*S, 1, hidden_size]
        qk_packed0_p: torch.Tensor,           # [P*S, 1, in_out_ch]
        state_indices_p: torch.Tensor,        # [P] int64 GPU — read slots
        has_initial_states_p: torch.Tensor,   # [P] bool GPU (unused; cached path)
        prev_hs: torch.Tensor,
        conv_states: torch.Tensor,
        P: int, S: int,
        hs2_prefill: torch.Tensor,
        qk_packed3_prefill: torch.Tensor,
        state_indices_p_write: Optional[torch.Tensor] = None,
        skip_writes: bool = False,
    ) -> None:
        """Cudagraph-replay-safe vectorized prefill for uniform (P, S).

        Eliminates Python-state-on-metadata branches from the captured
        path. Performs the default commit unless ``skip_writes`` is set
        (TiDAR drafter scratch path). TiDAR runs additionally stash
        candidates so ``commit_spec_decode_state`` can overwrite the
        default commit with the right post-sampling state.
        """
        H_conv = self.in_out_ch
        H_hs = self.hidden_size
        cca_time0 = self.cca_time0
        total_padding = self.total_padding

        if state_indices_p_write is None:
            state_indices_p_write = state_indices_p

        # Inputs are either [P*S, H] (v0.16 path) or [P*S, 1, H] (v0.15.x).
        # view(P, S, H_*) works for both since total element count matches.
        hs_p_2d = hs_p.reshape(P, S, H_hs)
        qk_packed0_p_2d = qk_packed0_p.reshape(P, S, H_conv)

        # Cached state reads. In steady-state captured replay,
        # has_initial_states_p is always True; cold prefills live on
        # the eager fallback path.
        hs_cached = prev_hs[state_indices_p].to(hs_p_2d.dtype)
        qk_cached = conv_states[state_indices_p].to(
            qk_packed0_p_2d.dtype)
        if hasattr(self, "_patch_probe_qk_cached"):
            probe_reqs = min(P, self._patch_probe_qk_cached.shape[0])
            self._patch_probe_qk_cached[:probe_reqs].copy_(
                qk_cached[:probe_reqs].reshape(probe_reqs, -1))
            self._patch_probe_hs_cached[:probe_reqs].copy_(
                hs_cached[:probe_reqs])

        # hs2 context: [hs_cached, hs_p[:, :-1]] -> [P, S, H_hs].
        hs2_p_2d = torch.cat(
            [hs_cached.unsqueeze(1), hs_p_2d[:, :-1, :]], dim=1)

        # qk_packed2: [qk_cached, qk_packed1_p]
        qk_packed1_p = qk_packed0_p_2d.permute(0, 2, 1)
        qk_packed2_p = torch.cat([qk_cached, qk_packed1_p], dim=-1)

        # Conv: [P, H_conv, total_padding + S] -> [P, H_conv, S]
        # Both paths perform the same FP32 two-stage convolution. Keep cuDNN
        # as the default; the unfold path is useful for ROCm shape tuning.
        if _CCA_AMD_CONV_UNFOLD_ENABLED:
            qk_packed3_p = self._conv_qk_decode(qk_packed2_p)
        else:
            qk_packed3_p = self._conv_qk_apply(qk_packed2_p)
        qk_packed3_prefill.copy_(
            qk_packed3_p.permute(0, 2, 1).contiguous()
            .reshape(qk_packed3_prefill.shape))

        hs2_prefill.copy_(hs2_p_2d.reshape(hs2_prefill.shape))

        if not skip_writes:
            # Default commit. TiDAR runs additionally stash all candidates.
            last_conv = qk_packed2_p[..., -cca_time0:].contiguous()
            last_hs = hs_p_2d[:, -1, :]
            conv_states[state_indices_p_write] = last_conv.to(
                device=conv_states.device, dtype=conv_states.dtype)
            prev_hs[state_indices_p_write] = last_hs.to(
                device=prev_hs.device, dtype=prev_hs.dtype)

            if (self._spec_stash_conv is not None
                    and P <= self._spec_max_P
                    and S <= self._spec_max_S):
                start_off = total_padding - cca_time0 + 1
                windowed = (qk_packed2_p[..., start_off:]
                            .unfold(-1, cca_time0, 1))
                # [P, H_conv, S, cca_time0] -> [P, S, H_conv, cca_time0]
                cand_conv = windowed.permute(0, 2, 1, 3).contiguous().to(
                    dtype=self._spec_stash_conv.dtype)
                cand_hs = hs_p_2d.to(dtype=self._spec_stash_hs.dtype)
                self._spec_stash_conv[:P, :S].copy_(cand_conv)
                self._spec_stash_hs[:P, :S].copy_(cand_hs)
                self._spec_stash_slots[:P].copy_(
                    state_indices_p.to(torch.int64))

    def _spec_decode_proposal_sub_loop(
        self,
        hs_p_props: torch.Tensor,
        qk_packed0_p_props: torch.Tensor,
        proposal_acc_levels: torch.Tensor,
        P: int, P_props: int, K: int,
        hs2_prefill_props: torch.Tensor,
        qk_packed3_prefill_props: torch.Tensor,
    ) -> None:
        """TiDAR single-forward proposal sub-loop.

        For each request i and proposal p_idx, run conv starting from
        the verify-segment stash entry at acc_level = proposal_acc_levels[p_idx].
        No commit/no stash for the proposal segment.
        """
        H_conv = self.in_out_ch
        H_hs = self.hidden_size
        total_padding = self.total_padding

        hs_p_3d = hs_p_props.reshape(P, P_props, K, H_hs)
        qk_packed0_p_3d = qk_packed0_p_props.reshape(P, P_props, K, H_conv)

        # init_conv: [P, P_props, H_conv, cca_time0=total_padding]
        init_conv = (
            self._spec_stash_conv[:P][:, proposal_acc_levels]
            .to(qk_packed0_p_3d.dtype))
        init_hs = (
            self._spec_stash_hs[:P][:, proposal_acc_levels]
            .to(hs_p_3d.dtype))

        # qk_packed2: [P, P_props, H_conv, total_padding + K]
        qk_packed1 = qk_packed0_p_3d.permute(0, 1, 3, 2)
        qk_packed2 = torch.cat([init_conv, qk_packed1], dim=-1)

        qk_packed2_flat = qk_packed2.reshape(
            P * P_props, H_conv, total_padding + K)
        qk_packed3_flat = self._conv_qk_apply(qk_packed2_flat)
        # [P*P_props, H_conv, K] -> [P*P_props*K, 1, H_conv]
        qk_packed3_out = (
            qk_packed3_flat
            .view(P, P_props, H_conv, K)
            .permute(0, 1, 3, 2)
            .contiguous()
            .reshape(qk_packed3_prefill_props.shape))
        qk_packed3_prefill_props.copy_(qk_packed3_out)

        # hs2: [init_hs, mask_hs[0..K-2]] -> [P, P_props, K, H_hs].
        hs2 = torch.cat(
            [init_hs.unsqueeze(2), hs_p_3d[:, :, :-1, :]], dim=2)
        hs2_prefill_props.copy_(
            hs2.reshape(hs2_prefill_props.shape))

    def _stash_spec_candidates_rowwise(
        self,
        hs_p: torch.Tensor,                 # [Tp, H] prefill tokens
        qk_packed0_p: torch.Tensor,         # [Tp, E] prefill conv inputs
        state_indices_p: torch.Tensor,      # [P] read slots
        conv_states: torch.Tensor,
        prev_hs: torch.Tensor,
        num_prefills: int,
        query_start_loc_p: torch.Tensor,
        has_initial_states_p: torch.Tensor,
        query_start_loc_p_cpu: Optional[torch.Tensor] = None,
        has_initial_states_p_cpu: Optional[torch.Tensor] = None,
    ) -> None:
        """Row-wise stash for NON-uniform prefill batches.

        The vectorized prefill path stashes all K+1 post-position candidate
        states per request so ``commit_spec_decode_state`` can roll the CCA
        state back to the accepted prefix after rejection sampling. Batches
        that mix TiDAR verify rows with ordinary prompt-prefill rows (or
        contain a partial final verify block, drafts < K) fail the uniform
        shape gate and take the fused/eager prefill path, which does not
        stash: the verify rows would keep the default post-full-draft-suffix
        state AND the stale stash from an earlier step would be committed
        over live slots.

        Stash row i corresponds to prefill row i (batch order), matching the
        vectorized layout. Rows longer than the stash window (ordinary
        prompt prefills; spec rows always satisfy L = drafts + 1 <= K + 1)
        get slot -1 so commit skips them. Must run BEFORE the prefill path
        updates conv_states, because candidate windows are built from the
        pre-forward cached state. Non-uniform shapes are never cudagraph-
        captured, so the Python control flow here is safe.
        """
        if query_start_loc_p_cpu is not None:
            qsl = [int(x) for x in query_start_loc_p_cpu.tolist()]
        else:
            qsl = [int(x) for x in query_start_loc_p.tolist()]
        if has_initial_states_p_cpu is not None:
            has_init = [bool(x) for x in has_initial_states_p_cpu.tolist()]
        else:
            has_init = [bool(x) for x in has_initial_states_p.tolist()]
        slots_cpu = [
            int(x) for x in
            state_indices_p[:num_prefills].detach().cpu().tolist()
        ]
        max_S = self._spec_max_S
        time0 = self.cca_time0
        total_padding = self.total_padding
        start_off = total_padding - time0 + 1
        n = min(num_prefills, self._spec_max_P, len(qsl) - 1, len(slots_cpu))
        if n <= 0:
            return
        new_slots = torch.full((n, ), -1, dtype=torch.int64,
                               device=self._spec_stash_slots.device)
        for i in range(n):
            start_i, end_i = qsl[i], qsl[i + 1]
            L = end_i - start_i
            if L < 1 or L > max_S or slots_cpu[i] < 0:
                continue
            qk_row = qk_packed0_p[start_i:end_i].T  # [E, L]
            if has_init[i]:
                cached = conv_states[slots_cpu[i]].to(qk_row.dtype)
                qk2 = torch.cat([cached, qk_row], dim=-1)
            else:
                qk2 = F.pad(qk_row, (total_padding, 0))
            # Candidate window n ends at row token n (same layout as the
            # vectorized path): [E, L, time0] -> [L, E, time0].
            windowed = qk2[:, start_off:].unfold(-1, time0, 1)
            self._spec_stash_conv[i, :L].copy_(
                windowed.permute(1, 0, 2).to(self._spec_stash_conv.dtype))
            self._spec_stash_hs[i, :L].copy_(
                hs_p[start_i:end_i].to(self._spec_stash_hs.dtype))
            new_slots[i] = slots_cpu[i]
        self._spec_stash_slots[:n].copy_(new_slots)

    @torch.inference_mode()
    def commit_spec_decode_state(
        self,
        num_accepted_per_batch_idx: list,
        virtual_engine: int = 0,
        idx_gpu: Optional[torch.Tensor] = None,
        arange_gpu: Optional[torch.Tensor] = None,
        mask_gpu: Optional[torch.Tensor] = None,
    ) -> None:
        """Overwrite conv_states / prev_hs with the post-acceptance state.

        Called by the model runner after rejection sampling, OUTSIDE the
        captured cudagraph. ``num_accepted_per_batch_idx[i]`` refers to
        stash row i (= prefill row i of the verify forward, batch order).
        Entries < 0 mark rows that must NOT be committed (non-spec prefill
        rows in a mixed batch); rows whose stashed slot is < 0 (not stashed
        this step) are skipped as well. Skipping is implemented as a no-op
        write-back of the row's current state so no host sync is needed.
        """
        n_used = min(len(num_accepted_per_batch_idx), self._spec_max_P)
        if (n_used == 0 or self._spec_stash_conv is None
                or self._spec_stash_hs is None
                or self._spec_stash_slots is None):
            return
        self_kv_cache = self.kv_cache[virtual_engine]
        conv_states = self_kv_cache[0]
        prev_hs = self_kv_cache[1]

        if idx_gpu is None or arange_gpu is None:
            K_max = self._spec_max_S - 1
            raw = list(num_accepted_per_batch_idx[:n_used])
            if len(raw) < n_used:
                raw.extend([-1] * (n_used - len(raw)))
            accepted = [min(max(0, n), K_max) for n in raw]
            idx = torch.as_tensor(accepted, dtype=torch.long,
                                  device=self._spec_stash_conv.device)
            arange = torch.arange(n_used, device=idx.device)
            mask = torch.as_tensor([n >= 0 for n in raw],
                                   dtype=torch.bool, device=idx.device)
        else:
            idx = idx_gpu
            arange = arange_gpu
            mask = mask_gpu

        selected_conv = self._spec_stash_conv[arange, idx]
        selected_hs = self._spec_stash_hs[arange, idx]

        # Key slot lookup by the same fresh stash rows used to gather
        # candidates. Some mixed prefill/decode steps stash sparse rows, so
        # assuming stash row == positional batch row would write stale state.
        slots = self._spec_stash_slots[arange.to(torch.long)].to(torch.long)
        valid = slots >= 0
        if mask is not None:
            valid = valid & mask
        # Skipped rows write their own current state back (slot 0 is the
        # reserved pad block for rows whose slot is -1) — keeps the scatter
        # shape static and avoids a host sync on the valid mask.
        safe_slots = torch.where(valid, slots, torch.zeros_like(slots))
        cur_conv = conv_states[safe_slots]
        cur_hs = prev_hs[safe_slots]
        out_conv = torch.where(
            valid.view(-1, 1, 1),
            selected_conv.to(device=conv_states.device,
                             dtype=conv_states.dtype),
            cur_conv)
        out_hs = torch.where(
            valid.view(-1, 1),
            selected_hs.to(device=prev_hs.device, dtype=prev_hs.dtype),
            cur_hs)
        conv_states[safe_slots] = out_conv
        prev_hs[safe_slots] = out_hs

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
            hs = hidden_states  # [S, H]
            hs_d = F.pad(hs[:-1], pad=(0, 0, 1, 0))  # [S, H]
            q = self.linear_q(hs)   # [S, latent_q_dim]
            k = self.linear_k(hs)   # [S, latent_k_dim]
            qk_packed0 = torch.cat([q, k], dim=-1)  # [S, latent_q + latent_k]

            S = qk_packed0.shape[0]
            query_pre = qk_packed0[..., :self.latent_q_dim].view(
                S, self.num_q_heads, self.head_dim
            )  # [S, qh, dh]

            key_pre = qk_packed0[..., self.latent_q_dim:].view(
                S, self.num_k_heads, self.head_dim
            )  # [S, kh, dh]
            key_pre = key_pre.unsqueeze(-2).repeat(1, 1, self.gqa_groups, 1) \
                            .view(S, self.num_q_heads, self.head_dim)  # [S, qh, dh]

            qk_mean_q = (query_pre.float() + key_pre.float()) / 2
            qk_mean_k = qk_mean_q.view(S, self.num_k_heads, self.gqa_groups, -1).mean(dim=-2)

            qk_packed1 = qk_packed0.T.unsqueeze(0)  # [1, E, S]
            qk_packed2 = F.pad(qk_packed1, (self.total_padding, 0))
            qk_packed3 = self._conv_qk_apply(qk_packed2).permute(2, 0, 1)  # [S, B, E]

            query = qk_packed3[..., :self.latent_q_dim].view(
                S, self.num_q_heads, self.head_dim
            ).float() + qk_mean_q

            key = qk_packed3[..., self.latent_q_dim:].view(
                S, self.num_k_heads, self.head_dim
            ).float() + qk_mean_k

            v1 = self.val_proj1(hs)   # [S, latent_k_dim/2]
            v2 = self.val_proj2(hs_d) # [S, latent_k_dim/2]
            value = torch.cat([v1, v2], dim=-1).contiguous() \
                        .view(S, self.num_k_heads, self.head_dim)  # [S, kh, dh]

            query, key = self._rms_normalize_qk(query, key, hs.dtype)

            return hs

        num_prefills = attn_metadata.num_prefills  # request count
        num_decodes = attn_metadata.num_decode_tokens  # token count (=request)
        num_prefill_tokens = attn_metadata.num_prefill_tokens  # token count
        has_prefill = num_prefills > 0
        has_decode = num_decodes > 0
        num_actual_tokens = num_decodes + num_prefill_tokens

        num_input_tokens, hidden_size = hidden_states.shape
        hs = hidden_states[:num_actual_tokens]  # [S, H]

        q = self.linear_q(hs)  # [S, latent_q_dim]
        k = self.linear_k(hs)  # [S, latent_k_dim]
        qk_packed0 = torch.cat([q, k], dim=-1)  # [S, latent_q + latent_k]
        if hasattr(self, "_patch_probe_qk_linear"):
            probe_rows = min(qk_packed0.shape[0],
                             self._patch_probe_qk_linear.shape[0])
            self._patch_probe_qk_linear[:probe_rows].copy_(
                qk_packed0[:probe_rows])

        query_pre = qk_packed0[..., :self.latent_q_dim].view(
            num_actual_tokens, self.num_q_heads, self.head_dim
        )  # [S, qh, dh]

        key_pre = qk_packed0[..., self.latent_q_dim:].view(
            num_actual_tokens, self.num_k_heads, self.head_dim
        )  # [S, kh, dh]
        key_pre = key_pre.unsqueeze(-2).repeat(1, 1, self.gqa_groups, 1) \
                          .view(num_actual_tokens, self.num_q_heads, self.head_dim)  # [S, qh, dh]

        qk_mean_q = (query_pre.float() + key_pre.float()) / 2
        qk_mean_k = qk_mean_q.view(num_actual_tokens, self.num_k_heads, self.gqa_groups, -1).mean(dim=-2)

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
            hs2 = torch.zeros((num_prefill_tokens, self.hidden_size), device=hs.device, dtype=hs.dtype)
            qk_packed3_p = torch.zeros((num_prefill_tokens, self.in_out_ch), device=hs.device, dtype=hs.dtype)
            for i in range(len(query_start_loc_p) - 1):
                start_i, end_i = query_start_loc_p[i], query_start_loc_p[i + 1]
                hs2_cur = hs_p[start_i:end_i]  # [S_cur, H]
                qk_packed0_cur = qk_packed0_p[start_i:end_i]  # [S_cur, E]
                qk_packed1_cur = qk_packed0_cur.T.unsqueeze(0)  # [1, E, S_cur]
                
                _use_init_p = bool(has_initial_states_p[i]) and (0 <= int(state_indices_tensor_p[i]) < prev_hs.shape[0])  # ccaoob-prefill-fix(JZ): OOB read slot -> no initial state
                if _use_init_p:
                    hs2_cached = prev_hs[state_indices_tensor_p[i]].to(hs.dtype).unsqueeze(0)  # [1, H]
                    hs2_cur = torch.cat([hs2_cached, hs2_cur[:-1]], dim=0)  # [S_cur, H]
                    qk_packed0_cached = conv_states[state_indices_tensor_p[i]].to(qk_packed0.dtype).unsqueeze(0)  # [1, E, total_padding]
                    qk_packed2_cur = torch.cat([qk_packed0_cached, qk_packed1_cur], dim=-1)  # [1, E, S_cur + total_padding]
                else:
                    hs2_cur = F.pad(hs2_cur[:-1], pad=(0, 0, 1, 0))
                    qk_packed2_cur = F.pad(qk_packed1_cur, (self.total_padding, 0))
                    
                hs2[start_i:end_i] = hs2_cur

                _widx_p = int(state_indices_tensor_p[i])  # ccaoob-prefill-fix(JZ): skip OOB write slot
                if 0 <= _widx_p < conv_states.shape[0]:
                    conv_states_cur = nn.functional.pad(qk_packed2_cur, (self.cca_time0 - qk_packed2_cur.shape[-1], 0))
                    conv_states[_widx_p] = conv_states_cur.squeeze(0).to(
                        device=conv_states.device, dtype=conv_states.dtype)
                
                # Computing conv
                qk_packed3_cur = self._conv_qk_apply(qk_packed2_cur).squeeze(0).T  # [S, E]
                qk_packed3_p[start_i:end_i] = qk_packed3_cur

            qk_packed3_output_list.append(qk_packed3_p)
            hs2_output_list.append(hs2)
            _vals_p = hs_p[query_start_loc_p[1:] - 1].to(device=prev_hs.device, dtype=prev_hs.dtype)  # ccaoob-prefill-fix(JZ)
            _wv_p = (state_indices_tensor_p >= 0) & (state_indices_tensor_p < prev_hs.shape[0])
            if bool(_wv_p.all()):
                prev_hs[state_indices_tensor_p] = _vals_p
            else:
                prev_hs[state_indices_tensor_p[_wv_p]] = _vals_p[_wv_p]

        if has_decode:
            # Generation
            # Handle PAD_SLOT_ID with torch.where for CUDA graph compatibility
            decode_is_pad = ((state_indices_tensor_d == PAD_SLOT_ID) | (state_indices_tensor_d < 0) | (state_indices_tensor_d >= prev_hs.shape[0]))  # ccaoob-fix(JZ): treat <0 / >=num_slots as pad -> avoids forward_triton illegal-mem (Xid31)
            safe_decode_indices = torch.where(
                decode_is_pad,
                torch.zeros_like(state_indices_tensor_d),
                state_indices_tensor_d,
            )
            qk_packed0_d = torch.where(
                decode_is_pad.view(-1, 1),
                qk_packed0_d.new_zeros(()),
                qk_packed0_d,
            )
            hs_d = torch.where(
                decode_is_pad.view(-1, 1),
                hs_d.new_zeros(()),
                hs_d,
            )

            qk_packed0_cached = conv_states[safe_decode_indices]  # [S, E, total_padding]
            qk_packed0_cached = torch.where(
                decode_is_pad.view(-1, 1, 1),
                qk_packed0_cached.new_zeros(()),
                qk_packed0_cached,
            )

            qk_packed0_cat = torch.cat([qk_packed0_cached, qk_packed0_d.unsqueeze(-1)], dim=-1)  # [S, E, total_padding + 1]
            qk_packed3_d = self._decode_conv(qk_packed0_cat).squeeze(-1)  # [S, E]
            # Zero out conv output for padded positions (conv bias adds non-zero values even with zero input)
            qk_packed3_d = torch.where(
                decode_is_pad.view(-1, 1),
                qk_packed3_d.new_zeros(()),
                qk_packed3_d,
            )
            qk_packed3_output_list.insert(0, qk_packed3_d)
            
            new_qk_packed0_cache = qk_packed0_cached.roll(shifts=-1, dims=-1)
            new_qk_packed0_cache[..., -1] = qk_packed0_d
            new_qk_packed0_cache = torch.where(
                decode_is_pad.view(-1, 1, 1),
                new_qk_packed0_cache.new_zeros(()),
                new_qk_packed0_cache,
            )
            conv_states[safe_decode_indices] = new_qk_packed0_cache.to(
                device=conv_states.device, dtype=conv_states.dtype)

            hs2 = prev_hs[safe_decode_indices]  # [S, H]
            hs2 = torch.where(
                decode_is_pad.view(-1, 1),
                hs2.new_zeros(()),
                hs2,
            )
            hs2_output_list.insert(0, hs2)
            new_prev_hs = hs_d.to(prev_hs.dtype)
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
        # Keep the recurrent cache in its configured dtype, but project in the
        # model activation dtype so the GEMM input matches the bf16 weights.
        v2 = self.val_proj2(hs2.to(dtype=hs.dtype))
        value = torch.cat([v1, v2], dim=-1).contiguous()
        value = value.view(num_actual_tokens, self.num_k_heads, self.head_dim)  # [S, kh, dh]

        query = qk_packed3[..., :self.latent_q_dim].view(
            num_actual_tokens, self.num_q_heads, self.head_dim
        ).float() + qk_mean_q

        key = qk_packed3[..., self.latent_q_dim:].view(
            num_actual_tokens, self.num_k_heads, self.head_dim
        ).float() + qk_mean_k

        query, key = self._rms_normalize_qk(query, key, hs.dtype)
        
        query = query.reshape(num_actual_tokens, self.num_q_heads * self.head_dim)
        key   = key.reshape(num_actual_tokens, self.num_k_heads * self.head_dim)
        value = value.reshape(num_actual_tokens, self.num_k_heads * self.head_dim)
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
        # TiDAR fields (set below from CCAAttentionMetadata)
        state_indices_tensor_write = None
        drafter_pass = False
        tidar_sf_verify_len = None
        tidar_sf_acc_levels = None
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
            state_indices_tensor_write = (
                attn_metadata.state_indices_tensor_write)
            drafter_pass = attn_metadata.drafter_pass
            tidar_sf_verify_len = (
                attn_metadata.tidar_single_forward_verify_len)
            tidar_sf_acc_levels = (
                attn_metadata.tidar_single_forward_proposal_acc_levels)
        if attn_metadata is None:
            # V1 profile run
            hs = hidden_states  # [S, H]
            hs_d = F.pad(hs[:-1], pad=(0, 0, 1, 0))  # [S, H]
            q = self.linear_q(hs)   # [S, latent_q_dim]
            k = self.linear_k(hs)   # [S, latent_k_dim]
            qk_packed0 = torch.cat([q, k], dim=-1)  # [S, latent_q + latent_k]

            S = qk_packed0.shape[0]
            query_pre = qk_packed0[..., :self.latent_q_dim].view(
                S, self.num_q_heads, self.head_dim
            )  # [S, qh, dh]

            key_pre = qk_packed0[..., self.latent_q_dim:].view(
                S, self.num_k_heads, self.head_dim
            )  # [S, kh, dh]
            key_pre = key_pre.unsqueeze(-2).repeat(1, 1, self.gqa_groups, 1) \
                            .view(S, self.num_q_heads, self.head_dim)  # [S, qh, dh]

            qk_mean_q = (query_pre.float() + key_pre.float()) / 2
            qk_mean_k = qk_mean_q.view(S, self.num_k_heads, self.gqa_groups, -1).mean(dim=-2)

            qk_packed1 = qk_packed0.T.unsqueeze(0)  # [1, E, S]
            qk_packed2 = F.pad(qk_packed1, (self.total_padding, 0))
            qk_packed3 = self._conv_qk_apply(qk_packed2).permute(2, 0, 1)  # [S, B, E]

            query = qk_packed3[..., :self.latent_q_dim].view(
                S, self.num_q_heads, self.head_dim
            ).float() + qk_mean_q

            key = qk_packed3[..., self.latent_q_dim:].view(
                S, self.num_k_heads, self.head_dim
            ).float() + qk_mean_k

            v1 = self.val_proj1(hs)   # [S, latent_k_dim/2]
            v2 = self.val_proj2(hs_d) # [S, latent_k_dim/2]
            value = torch.cat([v1, v2], dim=-1).contiguous() \
                        .view(S, self.num_k_heads, self.head_dim)  # [S, kh, dh]

            query, key = self._rms_normalize_qk(query, key, hs.dtype)

            return hs

        num_prefills = attn_metadata.num_prefills  # request count
        num_decodes = attn_metadata.num_decode_tokens  # token count (=request)
        num_prefill_tokens = attn_metadata.num_prefill_tokens  # token count
        has_prefill = num_prefills > 0
        has_decode = num_decodes > 0
        num_actual_tokens = num_decodes + num_prefill_tokens

        num_input_tokens, hidden_size = hidden_states.shape
        hs = hidden_states[:num_actual_tokens]  # [S, H]

        q = self.linear_q(hs)  # [S, latent_q_dim]
        k = self.linear_k(hs)  # [S, latent_k_dim]
        qk_packed0 = torch.cat([q, k], dim=-1)  # [S, latent_q + latent_k]

        if _CCA_TRITON_FUSION_ENABLED:
            qk_mean_q, qk_mean_k = fused_qk_mean(
                qk_packed0, self.num_q_heads, self.num_k_heads,
                self.head_dim, self.gqa_groups,
            )
        else:
            query_pre = qk_packed0[..., :self.latent_q_dim].view(
                num_actual_tokens, self.num_q_heads, self.head_dim
            )
            key_pre = qk_packed0[..., self.latent_q_dim:].view(
                num_actual_tokens, self.num_k_heads, self.head_dim
            )
            key_pre = key_pre.unsqueeze(-2).repeat(1, 1, self.gqa_groups, 1) \
                              .view(num_actual_tokens, self.num_q_heads, self.head_dim)
            qk_mean_q = (query_pre.float() + key_pre.float()) / 2
            qk_mean_k = qk_mean_q.view(
                num_actual_tokens, self.num_k_heads, self.gqa_groups, -1
            ).mean(dim=-2)
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

        # TiDAR drafter pass: writes go to scratch (state_indices_tensor_write)
        # NOT the AR slot (state_indices_tensor). Without this split the Python
        # loop below scrambles the AR-side CCA state, garbage propagates to
        # the next verify forward, and the model collapses (mean accept = K+1
        # with 100% acceptance of degenerate tokens).
        state_indices_tensor_write_d = None
        state_indices_tensor_write_p = None
        if state_indices_tensor_write is not None:
            state_indices_tensor_write_d, state_indices_tensor_write_p = (
                torch.split(
                    state_indices_tensor_write[:num_actual_tokens],
                    [num_decodes, num_prefills],
                    dim=0,
                ))

        # TODO: allocate memory for output tensors
        qk_packed3_output_list = []
        hs2_output_list = []
        decode_is_pad: Optional[torch.Tensor] = None

        if has_prefill:
            # === TiDAR vectorized prefill gate ====================
            # Route uniform-S prefill batches (the TiDAR verify steady
            # state) through a vectorized, GPU-tensor-only path that's
            # cudagraph-replay-safe. Uses ONLY batch-shape Python ints
            # for the gate decision — no metadata-flag branching — so
            # capture vs replay never disagree on which branch is
            # recorded.
            P_pref = num_prefills
            S_uniform = (num_prefill_tokens // P_pref) if P_pref > 0 else 0
            single_forward_mode = (
                tidar_sf_verify_len is not None
                and tidar_sf_acc_levels is not None)
            sf_total_S = 0
            sf_P_props = 0
            sf_K = 0
            sf_verify_len = 0
            if single_forward_mode:
                sf_P_props = int(tidar_sf_acc_levels.shape[0])
                sf_verify_len = int(tidar_sf_verify_len)
                if sf_P_props == 0:
                    raise ValueError(
                        "tidar_single_forward_proposal_acc_levels must have "
                        "at least one proposal; got empty tensor.")
                if (S_uniform - sf_verify_len) % sf_P_props != 0:
                    raise ValueError(
                        f"SF layout mismatch: S_uniform={S_uniform}, "
                        f"verify_len={sf_verify_len}, P_props={sf_P_props}")
                sf_K = (S_uniform - sf_verify_len) // sf_P_props
                sf_total_S = sf_verify_len + sf_P_props * sf_K
            capture_vectorized_active = (
                self._use_capture_vectorized
            )
            if capture_vectorized_active:
                from vllm.distributed.parallel_state import (
                    is_in_graph_capture_context,
                )
                capture_vectorized_active = is_in_graph_capture_context()
            # True per-row uniformity, not just divisibility of the total:
            # a [16, 18] pair also satisfies num_prefill_tokens == P * 17
            # but would garble the (P, S) reshape AND desync the runner's
            # commit-context mirror of this gate. Host-side check on the
            # pinned CPU mirror; falls back to the divisibility-only gate
            # when the CPU mirror is unavailable.
            _rows_uniform = True
            _qsl_p_cpu_gate = getattr(
                attn_metadata, "query_start_loc_p_cpu", None)
            if _qsl_p_cpu_gate is not None and P_pref > 0:
                _row_lens_gate = (
                    _qsl_p_cpu_gate[1:] - _qsl_p_cpu_gate[:-1])
                _rows_uniform = bool(
                    (_row_lens_gate == S_uniform).all().item())
            use_vectorized_prefill = (
                (
                    self._use_spec_vectorized
                    and P_pref > 0
                    and num_prefill_tokens == P_pref * S_uniform
                    and S_uniform == self._spec_max_S
                    and P_pref <= self._spec_max_P
                    and _rows_uniform
                    and not single_forward_mode
                )
                or (
                    capture_vectorized_active
                    and P_pref > 0
                    and S_uniform > 0
                    and num_prefill_tokens == P_pref * S_uniform
                    and _rows_uniform
                    and not single_forward_mode
                )
            )
            use_single_forward_prefill = (
                single_forward_mode
                and self._use_spec_vectorized
                and P_pref > 0
                and num_prefill_tokens == P_pref * sf_total_S
                and sf_verify_len <= self._spec_max_S
                and P_pref <= self._spec_max_P
            )

            # Non-uniform prefill batch under TiDAR (mixed verify + prompt
            # prefill rows, or a partial final verify block): the fused /
            # eager paths below don't stash spec candidates, so refresh the
            # stash row-wise here BEFORE conv_states is updated. Without
            # this, commit_spec_decode_state would commit a stale stash and
            # the verify rows would keep the post-full-draft-suffix state.
            if (self._spec_stash_conv is not None
                    and not drafter_pass
                    and not use_vectorized_prefill
                    and not use_single_forward_prefill):
                self._stash_spec_candidates_rowwise(
                    hs_p=hs_p,
                    qk_packed0_p=qk_packed0_p,
                    state_indices_p=state_indices_tensor_p,
                    conv_states=conv_states,
                    prev_hs=prev_hs,
                    num_prefills=P_pref,
                    query_start_loc_p=query_start_loc_p,
                    has_initial_states_p=has_initial_states_p,
                    query_start_loc_p_cpu=getattr(
                        attn_metadata, "query_start_loc_p_cpu", None),
                    has_initial_states_p_cpu=getattr(
                        attn_metadata, "has_initial_states_p_cpu", None),
                )

            if use_vectorized_prefill:
                hs2 = torch.empty(
                    (num_prefill_tokens, self.hidden_size),
                    device=hs.device, dtype=hs.dtype)
                qk_packed3_p = torch.empty(
                    (num_prefill_tokens, self.in_out_ch),
                    device=hs.device, dtype=hs.dtype)
                self._spec_decode_prefill_vectorized(
                    hs_p=hs_p,
                    qk_packed0_p=qk_packed0_p,
                    state_indices_p=state_indices_tensor_p,
                    has_initial_states_p=has_initial_states_p,
                    prev_hs=prev_hs,
                    conv_states=conv_states,
                    P=P_pref, S=S_uniform,
                    hs2_prefill=hs2,
                    qk_packed3_prefill=qk_packed3_p,
                    state_indices_p_write=state_indices_tensor_write,
                    skip_writes=drafter_pass,
                )
                qk_packed3_output_list.append(qk_packed3_p)
                hs2_output_list.append(hs2)
            elif use_single_forward_prefill:
                # TiDAR single-forward: verify segment + P proposal
                # sub-loops in one CCA call.
                H_hs_loc = self.hidden_size
                H_conv_loc = self.in_out_ch
                hs2 = torch.empty(
                    (num_prefill_tokens, H_hs_loc),
                    device=hs.device, dtype=hs.dtype)
                qk_packed3_p = torch.empty(
                    (num_prefill_tokens, H_conv_loc),
                    device=hs.device, dtype=hs.dtype)

                hs_p_4d = hs_p.reshape(P_pref, sf_total_S, H_hs_loc)
                qk_packed0_p_4d = qk_packed0_p.reshape(
                    P_pref, sf_total_S, H_conv_loc)
                hs2_4d = hs2.view(P_pref, sf_total_S, H_hs_loc)
                qk_packed3_p_4d = qk_packed3_p.view(
                    P_pref, sf_total_S, H_conv_loc)

                # Verify slice (first sf_verify_len rows per req).
                hs_p_verify = (
                    hs_p_4d[:, :sf_verify_len].contiguous()
                    .view(P_pref * sf_verify_len, H_hs_loc))
                qk_packed0_p_verify = (
                    qk_packed0_p_4d[:, :sf_verify_len].contiguous()
                    .view(P_pref * sf_verify_len, H_conv_loc))
                verify_hs2_tmp = torch.empty_like(hs_p_verify)
                verify_qk_tmp = torch.empty_like(qk_packed0_p_verify)

                self._spec_decode_prefill_vectorized(
                    hs_p=hs_p_verify,
                    qk_packed0_p=qk_packed0_p_verify,
                    state_indices_p=state_indices_tensor_p,
                    has_initial_states_p=has_initial_states_p,
                    prev_hs=prev_hs,
                    conv_states=conv_states,
                    P=P_pref, S=sf_verify_len,
                    hs2_prefill=verify_hs2_tmp,
                    qk_packed3_prefill=verify_qk_tmp,
                    state_indices_p_write=state_indices_tensor_write,
                    skip_writes=drafter_pass,
                )
                hs2_4d[:, :sf_verify_len].copy_(
                    verify_hs2_tmp.view(
                        P_pref, sf_verify_len, H_hs_loc))
                qk_packed3_p_4d[:, :sf_verify_len].copy_(
                    verify_qk_tmp.view(
                        P_pref, sf_verify_len, H_conv_loc))

                # Proposal slice.
                hs_p_props = (
                    hs_p_4d[:, sf_verify_len:].contiguous()
                    .view(P_pref * sf_P_props * sf_K, H_hs_loc))
                qk_packed0_p_props = (
                    qk_packed0_p_4d[:, sf_verify_len:].contiguous()
                    .view(P_pref * sf_P_props * sf_K, H_conv_loc))
                prop_hs2_tmp = torch.empty_like(hs_p_props)
                prop_qk_tmp = torch.empty_like(qk_packed0_p_props)

                self._spec_decode_proposal_sub_loop(
                    hs_p_props=hs_p_props,
                    qk_packed0_p_props=qk_packed0_p_props,
                    proposal_acc_levels=tidar_sf_acc_levels,
                    P=P_pref, P_props=sf_P_props, K=sf_K,
                    hs2_prefill_props=prop_hs2_tmp,
                    qk_packed3_prefill_props=prop_qk_tmp,
                )
                hs2_4d[:, sf_verify_len:].copy_(
                    prop_hs2_tmp.view(
                        P_pref, sf_P_props * sf_K, H_hs_loc))
                qk_packed3_p_4d[:, sf_verify_len:].copy_(
                    prop_qk_tmp.view(
                        P_pref, sf_P_props * sf_K, H_conv_loc))
                qk_packed3_output_list.append(qk_packed3_p)
                hs2_output_list.append(hs2)
            elif _CCA_FUSED_ENABLED:
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
                use_hip_prefill_fused = (
                    cca_prefill_fused_hip_available()
                    and prev_hs.dtype == hs_p.dtype
                    and conv_states.dtype == qk_packed0_p.dtype
                    and self.dw_weight_flat.dtype == qk_packed0_p.dtype
                    and self.gw_weight_flat.dtype == qk_packed0_p.dtype
                    and (self.conv_qk[0].bias is None
                         or self.conv_qk[0].bias.dtype == qk_packed0_p.dtype)
                    and self.gw_bias_flat.dtype == qk_packed0_p.dtype
                )
                if use_hip_prefill_fused:
                    hs2, qk_packed3_p = cca_prefill_fused_hip(
                        *_prefill_args)
                else:
                    hs2, qk_packed3_p = cca_prefill_fused(*_prefill_args)
                qk_packed3_output_list.append(qk_packed3_p)
                hs2_output_list.append(hs2)
            else:
                hs2 = torch.zeros((num_prefill_tokens, self.hidden_size), device=hs.device, dtype=hs.dtype)
                qk_packed3_p = torch.zeros((num_prefill_tokens, self.in_out_ch), device=hs.device, dtype=hs.dtype)
                # TiDAR: writes go to write-side slot if override is set
                # (drafter pass: AR-read, scratch-write). Otherwise mirror
                # the read slot (verify pass / non-spec).
                _sit_write_p = (state_indices_tensor_write_p
                                if state_indices_tensor_write_p is not None
                                else state_indices_tensor_p)
                # AMD host-overhead fix: pull the per-seq loop bounds and
                # initial-state flags to CPU ONCE (2 D2H) instead of paying a
                # blocking .item()/__bool__ sync on every GPU-scalar slice
                # inside the loop (~5x per seq). On ROCm each such sync is a
                # ~1ms blocking hipMemcpyWithStream; these were the dominant
                # TiDAR host overhead (GPU ~94% idle). Math is identical; only
                # the host<->device sync pattern changes.
                _qsl_cpu = (query_start_loc_p.tolist()
                            if torch.is_tensor(query_start_loc_p)
                            else query_start_loc_p)
                _has_init_cpu = (has_initial_states_p.tolist()
                                 if torch.is_tensor(has_initial_states_p)
                                 else has_initial_states_p)
                # Sparse spec stash for mixed prefill/decode steps: TiDAR
                # verify rows (S_cur <= K+1) still need their K+1 candidate
                # states stashed so commit_spec_decode_state can roll the
                # recurrent state back to the post-acceptance position. The
                # vectorized path only handles uniform-S segments, so this
                # loop stashes eligible rows itself and records which rows
                # it stashed (+ the runner's step seq) for the commit.
                _stash_here = (not drafter_pass
                               and self._spec_stash_conv is not None)
                _stash_j = 0
                _stash_rows: list = []
                for i in range(len(_qsl_cpu) - 1):
                    start_i, end_i = _qsl_cpu[i], _qsl_cpu[i + 1]
                    hs_row = hs_p[start_i:end_i]  # [S_cur, H] raw
                    hs2_cur = hs_row
                    qk_packed0_cur = qk_packed0_p[start_i:end_i]  # [S_cur, E]
                    qk_packed1_cur = qk_packed0_cur.T.unsqueeze(0)  # [1, E, S_cur]

                    _use_init_p = bool(has_initial_states_p[i]) and (0 <= int(state_indices_tensor_p[i]) < prev_hs.shape[0])  # ccaoob-prefill-fix(JZ): OOB read slot -> no initial state
                    if _use_init_p:
                        hs2_cached = prev_hs[state_indices_tensor_p[i]].to(hs.dtype).unsqueeze(0)  # [1, H]
                        hs2_cur = torch.cat([hs2_cached, hs2_cur[:-1]], dim=0)  # [S_cur, H]
                        qk_packed0_cached = conv_states[state_indices_tensor_p[i]].to(qk_packed0.dtype).unsqueeze(0)  # [1, E, total_padding]
                        qk_packed2_cur = torch.cat([qk_packed0_cached, qk_packed1_cur], dim=-1)  # [1, E, S_cur + total_padding]
                    else:
                        hs2_cur = F.pad(hs2_cur[:-1], pad=(0, 0, 1, 0))
                        qk_packed2_cur = F.pad(qk_packed1_cur, (self.total_padding, 0))

                    hs2[start_i:end_i] = hs2_cur

                    if not drafter_pass:
                        _widx_p = int(_sit_write_p[i])  # ccaoob-prefill-fix(JZ): skip OOB write slot
                        if 0 <= _widx_p < conv_states.shape[0]:
                            conv_states_cur = nn.functional.pad(qk_packed2_cur, (self.cca_time0 - qk_packed2_cur.shape[-1], 0))
                            conv_states[_widx_p] = conv_states_cur.squeeze(0).to(
                                device=conv_states.device, dtype=conv_states.dtype)

                    _s_cur = end_i - start_i
                    if (_stash_here and _s_cur <= self._spec_max_S
                            and _stash_j < self._spec_max_P):
                        # Same candidate math as the vectorized path:
                        # window n ends at position n+1 of this row.
                        _start_off = self.total_padding - self.cca_time0 + 1
                        _win = (qk_packed2_cur[..., _start_off:]
                                .unfold(-1, self.cca_time0, 1))
                        # [1, E, S_cur, t0] -> [S_cur, E, t0]
                        self._spec_stash_conv[_stash_j, :_s_cur].copy_(
                            _win.permute(0, 2, 1, 3)[0].to(
                                dtype=self._spec_stash_conv.dtype))
                        self._spec_stash_hs[_stash_j, :_s_cur].copy_(
                            hs_row.to(dtype=self._spec_stash_hs.dtype))
                        self._spec_stash_slots[_stash_j] = (
                            state_indices_tensor_p[i].to(torch.int64))
                        _stash_rows.append(i)
                        _stash_j += 1

                    # Computing conv
                    qk_packed3_cur = self._conv_qk_apply(qk_packed2_cur).squeeze(0).T  # [S, E]
                    qk_packed3_p[start_i:end_i] = qk_packed3_cur
                qk_packed3_output_list.append(qk_packed3_p)
                hs2_output_list.append(hs2)
                if not drafter_pass:
                    _vals_p = hs_p[query_start_loc_p[1:] - 1].to(device=prev_hs.device, dtype=prev_hs.dtype)  # ccaoob-prefill-fix(JZ)
                    _wv_p = (_sit_write_p >= 0) & (_sit_write_p < prev_hs.shape[0])
                    if bool(_wv_p.all()):
                        prev_hs[_sit_write_p] = _vals_p
                    else:
                        prev_hs[_sit_write_p[_wv_p]] = _vals_p[_wv_p]
                if _stash_here:
                    self._spec_stash_eager_rows = _stash_rows
                    self._spec_stash_eager_seq = CCA._tidar_step_seq

        _fused_decode_active = False
        if has_decode:
            use_capture_decode = False
            if self._use_capture_vectorized:
                from vllm.distributed.parallel_state import (
                    is_in_graph_capture_context,
                )
                use_capture_decode = is_in_graph_capture_context()

            # Generation
            if use_capture_decode:
                decode_is_pad = (state_indices_tensor_d == PAD_SLOT_ID)
                safe_decode_indices = torch.where(
                    decode_is_pad,
                    torch.zeros_like(state_indices_tensor_d),
                    state_indices_tensor_d,
                )

                qk_packed0_d = torch.where(
                    decode_is_pad.view(-1, 1),
                    qk_packed0_d.new_zeros(()),
                    qk_packed0_d,
                )
                hs_d = torch.where(
                    decode_is_pad.view(-1, 1),
                    hs_d.new_zeros(()),
                    hs_d,
                )

                qk_cached = conv_states[safe_decode_indices]
                qk_cached = torch.where(
                    decode_is_pad.view(-1, 1, 1),
                    qk_cached.new_zeros(()),
                    qk_cached,
                )
                qk_cached_for_compute = qk_cached
                if qk_cached_for_compute.dtype != qk_packed0_d.dtype:
                    qk_cached_for_compute = qk_cached_for_compute.to(
                        qk_packed0_d.dtype)
                qk_packed2_d = torch.cat(
                    [qk_cached_for_compute, qk_packed0_d.unsqueeze(-1)],
                    dim=-1)
                qk_packed3_d = self._conv_qk_decode(qk_packed2_d).squeeze(-1)
                qk_packed3_d = torch.where(
                    decode_is_pad.view(-1, 1),
                    qk_packed3_d.new_zeros(()),
                    qk_packed3_d,
                )
                qk_packed3_output_list.insert(0, qk_packed3_d)

                new_qk_cache = qk_cached.roll(shifts=-1, dims=-1)
                new_qk_cache[..., -1] = qk_packed0_d.to(new_qk_cache.dtype)
                new_qk_cache = torch.where(
                    decode_is_pad.view(-1, 1, 1),
                    new_qk_cache.new_zeros(()),
                    new_qk_cache,
                )
                conv_states[safe_decode_indices] = new_qk_cache.to(
                    device=conv_states.device, dtype=conv_states.dtype)

                hs2_d = prev_hs[safe_decode_indices].to(hs.dtype)
                hs2_d = torch.where(
                    decode_is_pad.view(-1, 1),
                    hs2_d.new_zeros(()),
                    hs2_d,
                )
                hs2_output_list.insert(0, hs2_d)
                new_prev_hs = hs_d.to(prev_hs.dtype)
                new_prev_hs = torch.where(
                    decode_is_pad.view(-1, 1),
                    new_prev_hs.new_zeros(()),
                    new_prev_hs,
                )
                prev_hs[safe_decode_indices] = new_prev_hs.to(
                    device=prev_hs.device, dtype=prev_hs.dtype)
            elif _CCA_TRITON_FUSION_ENABLED:
                hs2_d, decode_is_pad = fused_pad_gather_scatter(
                    state_indices_tensor_d, hs_d, prev_hs,
                )
                hs2_output_list.insert(0, hs2_d)
            else:
                decode_is_pad = ((state_indices_tensor_d == PAD_SLOT_ID) | (state_indices_tensor_d < 0) | (state_indices_tensor_d >= prev_hs.shape[0]))  # ccaoob-fix(JZ): treat <0 / >=num_slots as pad -> avoids forward_triton illegal-mem (Xid31)
                safe_decode_indices = torch.where(
                    decode_is_pad,
                    torch.zeros_like(state_indices_tensor_d),
                    state_indices_tensor_d,
                )
                hs_d = torch.where(
                    decode_is_pad.view(-1, 1),
                    hs_d.new_zeros(()),
                    hs_d,
                )
                hs2_d = prev_hs[safe_decode_indices]  # [S, H]
                hs2_d = torch.where(
                    decode_is_pad.view(-1, 1),
                    hs2_d.new_zeros(()),
                    hs2_d,
                )
                prev_hs[state_indices_tensor_d] = hs_d.to(
                    device=prev_hs.device, dtype=prev_hs.dtype)
                hs2_output_list.insert(0, hs2_d)

            use_decode_fused = False
            if not use_capture_decode and cca_decode_fused_available():
                # ── Fused decode path ──
                # Single kernel: depthwise conv1d update + grouped conv + qk_mean
                #                + L2 norm + scale + temperature
                b = num_decodes
                dim, _, kernel_width = self.conv_qk[0].weight.shape
                groups = self.num_k_heads + self.num_q_heads
                head_dim = dim // groups

                first_input = qk_packed0_d.unsqueeze(-1).contiguous()  # [B, E, 1]

                dw_weight = self.dw_weight_flat
                if self._gw_weight_T is None:
                    gw = self.gw_weight_flat
                    self._gw_weight_T = gw.permute(0, 2, 3, 1).contiguous().view(
                        gw.shape[0], gw.shape[2] * gw.shape[3], gw.shape[1])
                gw_weight = self._gw_weight_T
                gw_bias = self.gw_bias_flat

                qk_mean_q_d = qk_mean_q[:num_decodes]  # [B, qh, dh]
                qk_mean_k_d = qk_mean_k[:num_decodes]  # [B, kh, dh]
                qk_mean_packed = torch.cat(
                    [qk_mean_q_d, qk_mean_k_d], dim=1).contiguous().to(hs.dtype)

                use_decode_fused = (
                    conv_states.dtype == first_input.dtype
                    and dw_weight.dtype == first_input.dtype
                    and gw_weight.dtype == first_input.dtype
                    and qk_mean_packed.dtype == first_input.dtype
                    and (self.conv_qk[0].bias is None
                         or self.conv_qk[0].bias.dtype == first_input.dtype)
                    and (gw_bias is None or gw_bias.dtype == first_input.dtype)
                )
                if use_decode_fused:
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

            if not use_capture_decode and not use_decode_fused:
                # ── Original unfused decode path ──
                dim, _, kernel_width = self.conv_qk[0].weight.shape
                weights = self.conv_qk[0].weight.reshape(dim, kernel_width).contiguous()

                first_input = qk_packed0_d.unsqueeze(-1).contiguous()
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
                qk_packed3_d = qk_packed3_d.reshape(b, -1).contiguous()  # [S, E]
                qk_packed3_d = torch.where(
                    decode_is_pad.view(-1, 1),
                    qk_packed3_d.new_zeros(()),
                    qk_packed3_d,
                )
                qk_packed3_output_list.insert(0, qk_packed3_d)
        hs2 = torch.vstack(hs2_output_list)[:num_actual_tokens]

        # Values from the two time streams
        v1 = self.val_proj1(hs)   # [S, B, latent_k_dim/2]
        # Keep the recurrent cache in its configured dtype, but project in the
        # model activation dtype so the GEMM input matches the bf16 weights.
        v2 = self.val_proj2(hs2.to(dtype=hs.dtype))
        value = torch.cat([v1, v2], dim=-1).contiguous()
        value = value.view(num_actual_tokens, self.num_k_heads, self.head_dim)  # [S, kh, dh]

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
                qk_packed3_pf = qk_packed3_p.contiguous()
                prefill_query = qk_packed3_pf[..., :self.latent_q_dim].view(
                    num_prefill_tokens, self.num_q_heads, self.head_dim
                ).float() + qk_mean_q[num_decodes:]
                prefill_key = qk_packed3_pf[..., self.latent_q_dim:].view(
                    num_prefill_tokens, self.num_k_heads, self.head_dim
                ).float() + qk_mean_k[num_decodes:]

                prefill_query, prefill_key = self._rms_normalize_qk(
                    prefill_query, prefill_key, hs.dtype)

                query[num_decodes:] = prefill_query.reshape(
                    num_prefill_tokens, self.latent_q_dim)
                key[num_decodes:] = prefill_key.reshape(
                    num_prefill_tokens, self.latent_k_dim)
        else:
            qk_packed3 = torch.vstack(qk_packed3_output_list)[:num_actual_tokens]
            if hasattr(self, "_patch_probe_qk_conv"):
                probe_rows = min(qk_packed3.shape[0],
                                 self._patch_probe_qk_conv.shape[0])
                self._patch_probe_qk_conv[:probe_rows].copy_(
                    qk_packed3[:probe_rows])
            query_h = qk_packed3[..., :self.latent_q_dim].view(
                num_actual_tokens, self.num_q_heads, self.head_dim
            ).float() + qk_mean_q
            key_h = qk_packed3[..., self.latent_q_dim:].view(
                num_actual_tokens, self.num_k_heads, self.head_dim
            ).float() + qk_mean_k

            query_h, key_h = self._rms_normalize_qk(query_h, key_h, hs.dtype)

            query = query_h.reshape(num_actual_tokens, self.latent_q_dim)
            key = key_h.reshape(num_actual_tokens, self.latent_k_dim)

        value = value.reshape(num_actual_tokens, self.num_k_heads * self.head_dim)
        if hasattr(self, "_patch_probe_value"):
            probe_rows = min(value.shape[0],
                             self._patch_probe_value.shape[0])
            self._patch_probe_value[:probe_rows].copy_(value[:probe_rows])

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
