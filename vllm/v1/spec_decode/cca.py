# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionBackend

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from vllm.attention.backends.abstract import AttentionMetadata
from vllm.attention.backends.utils import PAD_SLOT_ID
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
from vllm.utils import direct_register_custom_op
from vllm.v1.attention.backends.cca_attn import (
    CCAAttentionMetadata)

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
            prefix: str = ""
        ):
        super().__init__()
        self.config       = config
        self.model_config = model_config
        self.cache_config = cache_config
        self.layer_number = layer_number
        self.prefix = prefix

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

        # Per-k head temperature (Megatron: shape [num_k_heads])
        self.temp = nn.Parameter(torch.zeros(self.num_k_heads))

        vllm_cfg = get_current_vllm_config()
        compilation_config = vllm_cfg.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        self.kv_cache = (torch.tensor([]), torch.tensor([]))

        # Pre-allocated GPU stash for TiDAR spec-decode candidates. Sized at
        # init from speculative_config.num_speculative_tokens (= K) and
        # scheduler_config.max_num_seqs (= max prefill batch under TiDAR
        # verify). The forward path writes K+1 candidate (conv_state, hs)
        # pairs per req into these buffers via pure GPU ops — no Python
        # list mutation, so cudagraph replay sees fresh values each step.
        # commit_spec_decode_state, called outside the captured graph after
        # rejection sampling, indexes the buffers to write the post-
        # acceptance state to conv_states / prev_hs.
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

    @torch.inference_mode()
    def commit_spec_decode_state(
        self,
        num_accepted_per_batch_idx: list[int],
        virtual_engine: int = 0,
        idx_gpu: Optional[torch.Tensor] = None,
        arange_gpu: Optional[torch.Tensor] = None,
    ) -> None:
        """Overwrite conv_states / prev_hs with the post-acceptance state.

        Called by the model runner after rejection sampling, OUTSIDE the
        captured cudagraph. ``num_accepted_per_batch_idx[i]`` is the number
        of drafts accepted for input-batch position i. We gather the i-th
        candidate at offset (num_accepted+1) clamped to [0, K] from the
        GPU stash and write it back to the slot the forward stashed.

        The forward already wrote the last-position state to
        conv_states / prev_hs (the "default commit"). For TiDAR spec verify
        steps this call OVERWRITES that with the right candidate. For
        non-spec batches commit isn't called and the last-position write
        stays.

        ``n_used`` is derived from ``len(num_accepted_per_batch_idx)``
        rather than a Python state attribute set in the forward — Python
        assignments inside the captured graph only run at capture time
        and would not refresh on replay.

        ``idx_gpu`` / ``arange_gpu`` are pre-built per-step indices that
        the runner can hoist out of the per-layer loop. When None, this
        layer builds them itself (~one CPU→GPU sync per layer × 22
        layers = ~22 syncs/step). Passing them in cuts those syncs to
        one per step.
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
            # Standalone path (used in eager fallback or when caller didn't
            # hoist). Build a per-req acceptance index on-device.
            K_max = self._spec_max_S - 1
            accepted = [
                min(max(0, n), K_max)
                for n in num_accepted_per_batch_idx[:n_used]
            ]
            if len(accepted) < n_used:
                accepted.extend([0] * (n_used - len(accepted)))
            idx = torch.as_tensor(accepted, dtype=torch.long,
                                  device=self._spec_stash_conv.device)
            arange = torch.arange(n_used, device=idx.device)
        else:
            idx = idx_gpu
            arange = arange_gpu

        # Gather post-acceptance candidate per req. Stash layout:
        #   _spec_stash_conv[i, n] = conv_state at end of position n+1
        #   _spec_stash_hs[i, n]   = hs at position n
        # Take candidate at offset accepted[i] for each req.
        selected_conv = self._spec_stash_conv[arange, idx]
        selected_hs = self._spec_stash_hs[arange, idx]

        slots = self._spec_stash_slots[:n_used].to(torch.long)
        conv_states[slots] = selected_conv.to(
            device=conv_states.device, dtype=conv_states.dtype)
        prev_hs[slots] = selected_hs.to(
            device=prev_hs.device, dtype=prev_hs.dtype)

        # DEBUG (TiDAR FULL cudagraph drift): dump per-step CCA state for
        # req 0 to localize where eager and FULL diverge. Gated on
        # VLLM_TIDAR_DEBUG_STASH=1. Runs OUTSIDE the captured graph, so
        # tensor reads + torch.save are safe.
        if os.environ.get("VLLM_TIDAR_DEBUG_STASH", "0") == "1":
            step = getattr(self, "_debug_stash_step", 0)
            self._debug_stash_step = step + 1
            max_steps = int(
                os.environ.get("VLLM_TIDAR_DEBUG_STASH_MAX_STEPS", "16"))
            if step < max_steps:
                tag = os.environ.get(
                    "VLLM_TIDAR_DEBUG_STASH_TAG", "unknown")
                out_dir = os.environ.get(
                    "VLLM_TIDAR_DEBUG_STASH_DIR", "/tmp/tidar_dump")
                os.makedirs(out_dir, exist_ok=True)
                slot0 = int(slots[0].item())
                dump = {
                    "step": step,
                    "layer_prefix": getattr(self, "prefix", "unknown"),
                    "n_used": n_used,
                    "num_accepted": list(num_accepted_per_batch_idx),
                    "idx": idx.detach().cpu().clone(),
                    "slot0": slot0,
                    "stash_conv_0": (
                        self._spec_stash_conv[0].detach().cpu().clone()),
                    "stash_hs_0": (
                        self._spec_stash_hs[0].detach().cpu().clone()),
                    "selected_conv_0": (
                        selected_conv[0].detach().cpu().clone()),
                    "selected_hs_0": selected_hs[0].detach().cpu().clone(),
                    "committed_conv_slot0": (
                        conv_states[slot0].detach().cpu().clone()),
                    "committed_hs_slot0": (
                        prev_hs[slot0].detach().cpu().clone()),
                }
                safe_prefix = (
                    str(getattr(self, "prefix", "unknown"))
                    .replace("/", "_").replace(".", "_"))
                torch.save(
                    dump,
                    f"{out_dir}/{tag}_step{step:03d}_layer_{safe_prefix}.pt")

    def _rms_normalize_qk(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Equivalent to RMSNorm with unit weights and eps=1e-12/head_dim.
        # Normalize one tensor at a time in fp32 to reduce peak memory versus
        # the custom rms_norm op, which materializes an additional fp32 output.
        eps = 1e-12
        sqrt_head_dim = float(self.sqrt_head_dim)

        query_fp32 = query.to(torch.float32)
        q_norm = torch.linalg.vector_norm(
            query_fp32, ord=2, dim=-1, keepdim=True)
        query_fp32.mul_(torch.rsqrt(q_norm * q_norm + eps))
        query_fp32.mul_(sqrt_head_dim)
        query.copy_(query_fp32)

        key_fp32 = key.to(torch.float32)
        k_norm = torch.linalg.vector_norm(
            key_fp32, ord=2, dim=-1, keepdim=True)
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
        w0 = self.conv_qk[0].weight.squeeze(1)  # [C, K0]
        b0 = self.conv_qk[0].bias               # [C] or None

        x = x.to(w0.dtype)
        k0 = w0.shape[1]
        x_windows = x.unfold(-1, k0, 1)  # [N, C, L_mid, K0]
        mid = (x_windows * w0[:, None, :]).sum(dim=-1)  # [N, C, L_mid]
        if b0 is not None:
            mid = mid + b0[None, :, None]

        # Stage 2: grouped conv over the depthwise output.
        w1 = self.conv_qk[1].weight  # [C, D, K1]
        b1 = self.conv_qk[1].bias    # [C] or None
        g = self.num_k_heads + self.num_q_heads
        d = self.head_dim
        k1 = w1.shape[2]
        mid_windows = mid.view(mid.shape[0], g, d, mid.shape[-1]).unfold(-1, k1, 1)
        w1_grouped = w1.view(g, d, d, k1)
        out = torch.einsum("godk,sgdtk->sgot", w1_grouped, mid_windows)
        if b1 is not None:
            out = out + b1.view(1, g, d, 1)
        return out.reshape(x.shape[0], g * d, out.shape[-1])

    def _spec_decode_prefill_vectorized(
        self,
        hs_p: torch.Tensor,                   # [P*S, 1, hidden_size]
        qk_packed0_p: torch.Tensor,           # [P*S, 1, in_out_ch]
        state_indices_p: torch.Tensor,        # [P] int64 GPU — read slots
        has_initial_states_p: torch.Tensor,   # [P] bool GPU
        prev_hs: torch.Tensor,                # kv_cache[1] [n_blocks, hidden_size]
        conv_states: torch.Tensor,            # kv_cache[0] [n_blocks, in_out_ch, total_padding]
        P: int, S: int,
        hs2_prefill: torch.Tensor,            # [P*S, 1, hidden_size] — filled in place
        qk_packed3_prefill: torch.Tensor,     # [P*S, 1, in_out_ch] — filled in place
        state_indices_p_write: Optional[torch.Tensor] = None,  # [P] int64 GPU — write slots; defaults to state_indices_p
        skip_writes: bool = False,            # TiDAR drafter: skip default-commit + stash (pure scratch)
    ) -> None:
        """Cudagraph-replay-safe vectorized prefill for uniform shape (P, S).

        Eliminates every Python-state-on-metadata branch from the captured
        path: ``has_initial_states_p`` is a GPU bool tensor consumed via
        torch.where (so both init and cold paths are recorded), and
        ``state_indices_p`` is a GPU index tensor (so prev_hs / conv_states
        reads/writes are dynamic gather/scatter ops).

        ALWAYS performs both:
          (1) "default commit": writes the last-position state to
              conv_states[slot] / prev_hs[slot]. Correct for non-spec
              prefills; harmless for spec verify because (2) below stashes
              all K+1 candidates and ``commit_spec_decode_state``
              overwrites with the right one after rejection sampling.
          (2) "stash": writes K+1 candidate (conv_state, hs) pairs into
              the pre-allocated GPU stash buffers, indexed by req.

        Bounds ``commit_spec_decode_state``'s gather by P implicitly:
        commit derives n_used from the Python list it receives from the
        runner, which is the same length as the input batch's prefill
        section. No Python state is mutated here — that would not refresh
        on cudagraph replay.
        """
        H_conv = self.in_out_ch
        H_hs = self.hidden_size
        cca_time0 = self.cca_time0
        total_padding = self.total_padding

        # Read slot ordering (state_indices_p) and write slot ordering
        # (state_indices_p_write) differ for TiDAR drafter forwards under
        # the single-state design: drafter reads from the AR block (post-
        # acceptance state) but scatters its default-commit writes into the
        # draft block (treated as scratch — overwritten next step). Verify
        # forward and non-spec prefill keep them equal (read=write=AR).
        if state_indices_p_write is None:
            state_indices_p_write = state_indices_p

        # [P*S, 1, H] -> [P, S, H]
        hs_p_2d = hs_p.view(P, S, H_hs)
        qk_packed0_p_2d = qk_packed0_p.view(P, S, H_conv)

        # Cached state reads. In steady-state TiDAR captured replay,
        # has_initial_states_p is always True: every captured forward
        # follows the cold prompt prefill that lives off the captured
        # path entirely. Cold-prefill batches of length K+1 are rare
        # enough we keep them on the eager fallback (the gate above
        # only requires shape, but cold prefill of exactly K+1 tokens
        # is unusual). So skip the torch.where + cold-path zeros
        # branches and use the cached state directly. Saves ~4 GPU ops
        # per CCA layer per forward = ~88 ops/step at K=16, b=1.
        hs_cached = prev_hs[state_indices_p].to(hs_p_2d.dtype)  # [P, H_hs]
        qk_cached = conv_states[state_indices_p].to(
            qk_packed0_p_2d.dtype)  # [P, H_conv, total_padding]

        # Build hs2 context: [hs_cached, hs_p[:, :-1]] -> [P, S, H_hs].
        hs2_p_2d = torch.cat(
            [hs_cached.unsqueeze(1), hs_p_2d[:, :-1, :]], dim=1)

        # Build qk_packed2: [qk_cached, qk_packed1_p] -> [P, H_conv, total_padding + S].
        qk_packed1_p = qk_packed0_p_2d.permute(0, 2, 1)
        qk_packed2_p = torch.cat([qk_cached, qk_packed1_p], dim=-1)

        # Conv: [P, H_conv, total_padding + S] -> [P, H_conv, S]
        qk_packed3_p = self._conv_qk_decode(qk_packed2_p)
        # [P, H_conv, S] -> [P, S, H_conv] -> [P*S, 1, H_conv]
        qk_packed3_prefill.copy_(
            qk_packed3_p.permute(0, 2, 1).contiguous().view(
                P * S, 1, H_conv))

        # hs2 [P, S, H_hs] -> [P*S, 1, H_hs]
        hs2_prefill.copy_(hs2_p_2d.reshape(P * S, 1, H_hs))

        # Default commit + stash. Both are pure scratch under the TiDAR
        # drafter forward (path 2): the draft slot is never read after this
        # forward, and the stash is consumed only by commit_spec_decode_state
        # which fires post-verify. Skipping them entirely (compute + writes)
        # under skip_writes=True saves ~4 ms/step at K=16 b=1 across 22 CCA
        # layers — the savings come from BOTH the skipped writes and from
        # inductor's kernel-selection / fusion heuristic seeing a smaller
        # graph for the drafter (Tier 3 captures verify and drafter as
        # separate cudagraphs, dynamo specializes the trace on the bool).
        # The smaller-graph effect introduces LSB-level float drift in the
        # upstream conv on a minority of prompts, so the drafter produces
        # slightly different drafts than verify would have predicted at the
        # same prefix. The model still emits valid argmax tokens; the byte
        # stream just doesn't match AR-eager exactly. (For Minerva b=1 K=16
        # n=20, 4/20 prompts diverge by ~10-160 tokens at the stop sequence
        # boundary — comparable in scale to AR-cap's drift from AR-eager.)
        # The kept tradeoff: +13% throughput vs Tier 3 alone in exchange
        # for losing byte-identical-to-AR-eager output. Use the v2 pattern
        # below (move compute outside the gate) to recover byte-identicality
        # at the cost of most of the perf win.
        if not skip_writes:
            # Default commit: last-position state, written to the WRITE slot.
            # For verify (read=write=AR): writes the last-K-position state to
            # AR; commit_spec_decode_state then overwrites with the right
            # post-acceptance candidate.
            last_conv = qk_packed2_p[..., -cca_time0:].contiguous()  # [P, H_conv, cca_time0]
            last_hs = hs_p_2d[:, -1, :]  # [P, H_hs]
            conv_states[state_indices_p_write] = last_conv.to(
                device=conv_states.device, dtype=conv_states.dtype)
            prev_hs[state_indices_p_write] = last_hs.to(
                device=prev_hs.device, dtype=prev_hs.dtype)

            # Stash all K+1 = S candidates per req.
            # Candidate window n for req i: qk_packed2_p[i, :, start_off+n :
            # start_off+n+cca_time0] where start_off = total_padding-cca_time0+1.
            # Vectorized via unfold over the time axis.
            start_off = total_padding - cca_time0 + 1
            windowed = qk_packed2_p[..., start_off:].unfold(-1, cca_time0, 1)
            # windowed: [P, H_conv, S, cca_time0] -> [P, S, H_conv, cca_time0]
            cand_conv = windowed.permute(0, 2, 1, 3).contiguous().to(
                dtype=self._spec_stash_conv.dtype)
            cand_hs = hs_p_2d.to(dtype=self._spec_stash_hs.dtype)  # [P, S, H_hs]
            # Write into stash buffer prefix [:P, :S].
            self._spec_stash_conv[:P, :S].copy_(cand_conv)
            self._spec_stash_hs[:P, :S].copy_(cand_hs)
            self._spec_stash_slots[:P].copy_(state_indices_p.to(torch.int64))

    def _spec_decode_proposal_sub_loop(
        self,
        hs_p_props: torch.Tensor,            # [P * P_props * K, 1, H_hs]
        qk_packed0_p_props: torch.Tensor,    # [P * P_props * K, 1, H_conv]
        proposal_acc_levels: torch.Tensor,   # [P_props] int64 GPU
        P: int,
        P_props: int,
        K: int,
        hs2_prefill_props: torch.Tensor,         # [P*P_props*K, 1, H_hs] - filled
        qk_packed3_prefill_props: torch.Tensor,  # [P*P_props*K, 1, H_conv] - filled
    ) -> None:
        """TiDAR single-forward proposal sub-loop.

        Pre-condition: ``_spec_decode_prefill_vectorized`` has already
        been called on the verify segment for this step, populating
        ``self._spec_stash_conv[:P, :verify_len]`` and
        ``self._spec_stash_hs[:P, :verify_len]`` with one candidate
        (conv_state, hs) per verify position.

        For each request ``i`` and proposal ``p_idx ∈ [0, P_props)``:
          - Initial conv state = ``_spec_stash_conv[i, acc_lvl]``  where
            ``acc_lvl = proposal_acc_levels[p_idx]``. Per the live
            convention (see cca.py:243), this is the conv window for
            "state after consuming verify[0..acc_lvl]" (= anchor + p_j
            drafts). Matches the post-rejection commit semantics.
          - Initial hs   = ``_spec_stash_hs[i, acc_lvl]``.
          - Run conv on ``[init_conv, mask_qk_inputs]`` (length
            ``total_padding + K``) → K outputs.
          - Build hs2 = ``[init_hs, mask_hs_inputs[:K-1]]`` (K positions).
          - Write outputs into the proposal slice of the prefill buffers.

        No commit / no stash for the proposal segment -- the post-mask
        state is discarded. The verify segment's stash drives the
        post-step ``commit_spec_decode_state`` call (unchanged).

        See docs/tidar_single_forward_design_2026-05-13.md §6 and
        scripts/_tidar_cca_subloops_smoke.py for the standalone
        algorithm validation.
        """
        H_conv = self.in_out_ch
        H_hs = self.hidden_size
        total_padding = self.total_padding

        # Reshape inputs to [P, P_props, K, H_*].
        hs_p_3d = hs_p_props.view(P, P_props, K, H_hs)
        qk_packed0_p_3d = qk_packed0_p_props.view(P, P_props, K, H_conv)

        # Gather initial states from the verify-segment stash.
        # Index along the S dim of ``_spec_stash_conv``:
        #   _spec_stash_conv: [max_P, max_S, H_conv, cca_time0]
        # Use stash[acc_level + 1] so CCA's state has consumed through
        # verify[p_j+1] (which approximates the bonus token under the
        # bonus-context-injection design). This matches the attention's
        # `kv_local <= p_j + 1` rule: both let the proposal see one more
        # verify slot than the "p_j drafts accepted" semantics would
        # otherwise suggest. Result shape: [P, P_props, H_conv, cca_time0].
        proposal_init_idx = (proposal_acc_levels + 1).clamp(
            max=self._spec_stash_conv.shape[1] - 1)
        init_conv = (
            self._spec_stash_conv[:P][:, proposal_init_idx]
            .to(qk_packed0_p_3d.dtype))
        init_hs = (
            self._spec_stash_hs[:P][:, proposal_init_idx]
            .to(hs_p_3d.dtype))  # [P, P_props, H_hs]

        # Build qk_packed2 for proposals: concat(init_conv, mask_qk).
        # qk_packed0_p_3d: [P, P_props, K, H_conv]
        #   permute -> [P, P_props, H_conv, K]
        qk_packed1 = qk_packed0_p_3d.permute(0, 1, 3, 2)
        # qk_packed2: [P, P_props, H_conv, total_padding + K]
        qk_packed2 = torch.cat([init_conv, qk_packed1], dim=-1)

        # Run conv. Flatten the (P, P_props) batch dims for _conv_qk_decode.
        qk_packed2_flat = qk_packed2.reshape(
            P * P_props, H_conv, total_padding + K)
        qk_packed3_flat = self._conv_qk_decode(qk_packed2_flat)
        # qk_packed3_flat: [P*P_props, H_conv, K]

        # Reshape to the prefill output format: [P*P_props*K, 1, H_conv].
        qk_packed3_out = (
            qk_packed3_flat
            .view(P, P_props, H_conv, K)
            .permute(0, 1, 3, 2)
            .reshape(P * P_props * K, 1, H_conv)
            .contiguous())
        qk_packed3_prefill_props.copy_(qk_packed3_out)

        # Build hs2: [init_hs, mask_hs[0..K-2]]  -> [P, P_props, K, H_hs].
        hs2 = torch.cat(
            [init_hs.unsqueeze(2), hs_p_3d[:, :, :-1, :]], dim=2)
        hs2_prefill_props.copy_(
            hs2.reshape(P * P_props * K, 1, H_hs))

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
            # Optional separate write-side ordering (TiDAR drafter under
            # the single-state design uses AR slots for read, draft slots
            # for scratch write). When None, write side mirrors read side.
            state_indices_tensor_write = (
                attn_metadata.state_indices_tensor_write)
            has_initial_states_p = attn_metadata.has_initial_states_p
            query_start_loc_p = attn_metadata.query_start_loc_p
            # Host-side mirrors used by the prefill loop below to avoid a
            # cudaStreamSynchronize per Python slice index. Both fields are
            # populated by CCAAttentionMetadataBuilder.build when num_prefills
            # > 0; they are None for decode-only batches and during the V1
            # profile run.
            query_start_loc_p_cpu = attn_metadata.query_start_loc_p_cpu
            has_initial_states_p_cpu = attn_metadata.has_initial_states_p_cpu
            # Host list (Python ints) of state-cache slot indices in batch
            # order. Indexing conv_states / prev_hs with 0-dim GPU tensors
            # forced an implicit sync per access; using Python ints from
            # this list keeps the indexed reads/writes purely on-device.
            state_indices_list = attn_metadata.state_indices_list
            # TiDAR drafter forward (path 2): when True, the CCA forward
            # skips the default-commit + stash writes since those land in
            # scratch slots nobody reads. See _spec_decode_prefill_vectorized.
            drafter_pass = attn_metadata.drafter_pass
            # TiDAR single-forward (sparse-proposal): when set, per-req
            # input layout is [verify (verify_len), proposals (P_props*K)].
            # CCA runs verify with stash, then proposal sub-loops from
            # stash[acc_lvl]. See _spec_decode_proposal_sub_loop +
            # docs/tidar_single_forward_design_2026-05-13.md §6.
            tidar_sf_verify_len = (
                attn_metadata.tidar_single_forward_verify_len)
            tidar_sf_acc_levels = (
                attn_metadata.tidar_single_forward_proposal_acc_levels)

        if attn_metadata is None:
            # V1 profile run
            hs = hidden_states.unsqueeze(0).transpose(0, 1).contiguous()
            hs_d = F.pad(hs[:-1], pad=(0, 0, 0, 0, 1, 0))    # [S, B, H]
            q = self.linear_q(hs)  # [S, B, latent_q_dim]
            k = self.linear_k(hs)  # [S, B, latent_k_dim]
            qk_packed0 = torch.cat([q, k], dim=-1)  # [S, B, latent_q + latent_k]
            del q
            del k

            # Pre-mean tensors in head form (for "qk_mean_{q,k}" calc)
            query_pre = qk_packed0[..., :self.latent_q_dim].view(
                *qk_packed0.shape[:2], self.num_q_heads, self.head_dim
            )  # [S, B, qh, dh]

            key_base = qk_packed0[..., self.latent_q_dim:].view(
                *qk_packed0.shape[:2], self.num_k_heads, self.head_dim
            )  # [S, B, kh, dh]

            qk_packed1 = qk_packed0.permute(1, 2, 0)             # [B, E, S]
            qk_packed2 = F.pad(qk_packed1, (self.total_padding, 0))
            qk_packed3 = self.conv_qk(qk_packed2).permute(2, 0, 1)  # [S, B, E]

            # Build queries/keys from conv output + means
            query = qk_packed3[..., :self.latent_q_dim].view(
                *qk_packed3.shape[:2], self.num_q_heads, self.head_dim
            ).float()

            key = qk_packed3[..., self.latent_q_dim:].view(
                *qk_packed3.shape[:2], self.num_k_heads, self.head_dim
            ).float()
            query, key = self._add_grouped_qk_means_inplace(query, key, query_pre, key_base)
            del query_pre
            del key_base
            del qk_packed0
            del qk_packed3

            # Values from the two time streams
            v1 = self.val_proj1(hs)   # [S, B, latent_k_dim/2]
            v2 = self.val_proj2(hs_d) # [S, B, latent_k_dim/2]
            value = torch.cat([v1, v2], dim=-1).contiguous() \
                        .view(*hs.shape[:2], self.num_k_heads, self.head_dim)  # [S, B, kh, dh]

            query, key = self._rms_normalize_qk(query.contiguous(), key.contiguous())
            
            return hs

        num_prefills = attn_metadata.num_prefills          # request count
        num_decode_reqs = attn_metadata.num_decodes        # request count
        num_decodes = attn_metadata.num_decode_tokens      # token count
        num_prefill_tokens = attn_metadata.num_prefill_tokens  # token count
        # When reorder_batch_threshold == 1 every decode req emits exactly
        # one token, so num_decodes == num_decode_reqs by coincidence.
        # At threshold > 1 (e.g., spec-decode with K+1 query length per
        # decode req) they diverge — use num_decode_reqs for any
        # request-dim index into state_indices_tensor / state_indices_list,
        # and num_decodes for token-dim slices into qk_packed0 / hs.
        num_reqs = num_decode_reqs + num_prefills
        has_prefill = num_prefills > 0
        has_decode = num_decodes > 0
        num_actual_tokens = num_decodes + num_prefill_tokens

        num_input_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states[:num_actual_tokens]

        # Batch size is effectively 1 in this path, so insert the singleton
        # dimension directly instead of transposing and materializing a copy.
        hs = hidden_states.unsqueeze(1)  # [S, 1, H]
        batch_size = hs.shape[1]

        q = self.linear_q(hs)  # [S, B, latent_q_dim]
        k = self.linear_k(hs)  # [S, B, latent_k_dim]
        qk_packed0 = torch.cat([q, k], dim=-1)  # [S, B, latent_q + latent_k]
        del q
        del k

        # Pre-mean tensors in head form (for "qk_mean_{q,k}" calc)
        query_pre = qk_packed0[..., :self.latent_q_dim].view(
            *qk_packed0.shape[:2], self.num_q_heads, self.head_dim
        )  # [S, B, qh, dh]

        key_base = qk_packed0[..., self.latent_q_dim:].view(
            *qk_packed0.shape[:2], self.num_k_heads, self.head_dim
        )  # [S, B, kh, dh]

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

        # Split along batch (request) dimension. state_indices_tensor is
        # per-request, so slice to num_reqs and split with the per-request
        # decode count, not the token count.
        state_indices_tensor_d, state_indices_tensor_p = torch.split(
            state_indices_tensor[:num_reqs],
            [num_decode_reqs, num_prefills],
            dim=0,
        )
        # Same split for the optional write-side override.
        if state_indices_tensor_write is not None:
            state_indices_tensor_write_d, state_indices_tensor_write_p = (
                torch.split(state_indices_tensor_write[:num_reqs],
                            [num_decode_reqs, num_prefills], dim=0))
        else:
            state_indices_tensor_write_d = None
            state_indices_tensor_write_p = None

        qk_packed3 = torch.empty(
            (num_actual_tokens, batch_size, self.in_out_ch),
            device=hs.device,
            dtype=hs.dtype,
        )
        hs2 = torch.empty(
            (num_actual_tokens, batch_size, self.hidden_size),
            device=hs.device,
            dtype=hs.dtype,
        )
        decode_is_pad: Optional[torch.Tensor] = None
        if has_prefill:
            # Prefill
            prefill_slice = slice(num_decodes, num_decodes + num_prefill_tokens)
            hs2_prefill = hs2[prefill_slice]
            qk_packed3_prefill = qk_packed3[prefill_slice]

            # Shape-only gate: route uniform-S prefill batches (the TiDAR
            # verify steady state) through a vectorized, GPU-tensor-only
            # path that's cudagraph-replay-safe. The decision uses ONLY
            # batch-shape Python ints — no metadata flag conditionals —
            # so capture vs replay never disagree on which branch is
            # recorded.
            #
            # Non-uniform / cold-prefill batches stay on the eager loopy
            # path below (which does Python branching on per-req metadata).
            # Cold prefills are not captured under FULL_DECODE_ONLY anyway.
            P_pref = num_prefills
            S_uniform = (num_prefill_tokens // P_pref) if P_pref > 0 else 0
            # TiDAR single-forward (sparse proposal) gate: per-req layout
            # is [verify (sf_verify_len), proposals (P_props*K)].
            single_forward_mode = (
                tidar_sf_verify_len is not None
                and tidar_sf_acc_levels is not None)
            sf_total_S = 0
            sf_P_props = 0
            sf_K = 0
            if single_forward_mode:
                sf_P_props = int(tidar_sf_acc_levels.shape[0])
                sf_verify_len = int(tidar_sf_verify_len)
                if sf_P_props == 0:
                    raise ValueError(
                        "tidar_single_forward_proposal_acc_levels must have "
                        "at least one proposal; got empty tensor.")
                if (S_uniform - sf_verify_len) % sf_P_props != 0:
                    raise ValueError(
                        f"Single-forward layout mismatch: S_uniform="
                        f"{S_uniform}, verify_len={sf_verify_len}, "
                        f"P_props={sf_P_props}; "
                        f"(S - verify_len) must be divisible by P_props.")
                sf_K = (S_uniform - sf_verify_len) // sf_P_props
                sf_total_S = sf_verify_len + sf_P_props * sf_K
            use_vectorized_prefill = (
                self._spec_stash_conv is not None
                and P_pref > 0
                and num_prefill_tokens == P_pref * S_uniform
                and S_uniform == self._spec_max_S
                and P_pref <= self._spec_max_P
                and hs_p.shape[1] == 1
                and not single_forward_mode
            )
            use_single_forward_prefill = (
                single_forward_mode
                and self._spec_stash_conv is not None
                and P_pref > 0
                and num_prefill_tokens == P_pref * sf_total_S
                and sf_verify_len <= self._spec_max_S
                and P_pref <= self._spec_max_P
                and hs_p.shape[1] == 1
            )

            if use_vectorized_prefill:
                self._spec_decode_prefill_vectorized(
                    hs_p=hs_p,
                    qk_packed0_p=qk_packed0_p,
                    state_indices_p=state_indices_tensor_p,
                    has_initial_states_p=has_initial_states_p,
                    prev_hs=prev_hs,
                    conv_states=conv_states,
                    P=P_pref,
                    S=S_uniform,
                    hs2_prefill=hs2_prefill,
                    qk_packed3_prefill=qk_packed3_prefill,
                    state_indices_p_write=state_indices_tensor_write_p,
                    skip_writes=drafter_pass,
                )
            elif use_single_forward_prefill:
                # TiDAR single-forward: verify segment + P proposal sub-loops
                # in one CCA call. Per-req layout = [verify (sf_verify_len),
                # prop_1 (sf_K), prop_2 (sf_K), ..., prop_P (sf_K)]. We
                # extract verify and proposal slices, run the existing
                # vectorized prefill on the verify slice (with stash),
                # then the proposal sub-loop. Outputs are scattered back
                # into hs2_prefill / qk_packed3_prefill at the right rows.
                H_hs_loc = self.hidden_size
                H_conv_loc = self.in_out_ch

                hs_p_4d = hs_p.view(P_pref, sf_total_S, 1, H_hs_loc)
                qk_packed0_p_4d = qk_packed0_p.view(
                    P_pref, sf_total_S, 1, H_conv_loc)
                hs2_prefill_4d = hs2_prefill.view(
                    P_pref, sf_total_S, 1, H_hs_loc)
                qk_packed3_prefill_4d = qk_packed3_prefill.view(
                    P_pref, sf_total_S, 1, H_conv_loc)

                # Verify slice (first sf_verify_len rows per req). Use
                # .contiguous() on the input slice to ensure the existing
                # _spec_decode_prefill_vectorized's reshape ops work; the
                # output slice is fed via a separate temp buffer then
                # scattered back (the non-contiguous output view can't
                # accept a flat .copy_).
                hs_p_verify = (
                    hs_p_4d[:, :sf_verify_len].contiguous()
                    .view(P_pref * sf_verify_len, 1, H_hs_loc))
                qk_packed0_p_verify = (
                    qk_packed0_p_4d[:, :sf_verify_len].contiguous()
                    .view(P_pref * sf_verify_len, 1, H_conv_loc))
                verify_hs2_tmp = torch.empty_like(hs_p_verify)
                verify_qk_tmp = torch.empty_like(qk_packed0_p_verify)

                self._spec_decode_prefill_vectorized(
                    hs_p=hs_p_verify,
                    qk_packed0_p=qk_packed0_p_verify,
                    state_indices_p=state_indices_tensor_p,
                    has_initial_states_p=has_initial_states_p,
                    prev_hs=prev_hs,
                    conv_states=conv_states,
                    P=P_pref,
                    S=sf_verify_len,
                    hs2_prefill=verify_hs2_tmp,
                    qk_packed3_prefill=verify_qk_tmp,
                    state_indices_p_write=state_indices_tensor_write_p,
                    skip_writes=drafter_pass,
                )
                # Scatter verify outputs back into the prefill buffer.
                hs2_prefill_4d[:, :sf_verify_len].copy_(
                    verify_hs2_tmp.view(
                        P_pref, sf_verify_len, 1, H_hs_loc))
                qk_packed3_prefill_4d[:, :sf_verify_len].copy_(
                    verify_qk_tmp.view(
                        P_pref, sf_verify_len, 1, H_conv_loc))

                # Proposal slice (next sf_P_props * sf_K rows per req).
                hs_p_props = (
                    hs_p_4d[:, sf_verify_len:].contiguous()
                    .view(P_pref * sf_P_props * sf_K, 1, H_hs_loc))
                qk_packed0_p_props = (
                    qk_packed0_p_4d[:, sf_verify_len:].contiguous()
                    .view(P_pref * sf_P_props * sf_K, 1, H_conv_loc))
                prop_hs2_tmp = torch.empty_like(hs_p_props)
                prop_qk_tmp = torch.empty_like(qk_packed0_p_props)

                self._spec_decode_proposal_sub_loop(
                    hs_p_props=hs_p_props,
                    qk_packed0_p_props=qk_packed0_p_props,
                    proposal_acc_levels=tidar_sf_acc_levels,
                    P=P_pref,
                    P_props=sf_P_props,
                    K=sf_K,
                    hs2_prefill_props=prop_hs2_tmp,
                    qk_packed3_prefill_props=prop_qk_tmp,
                )
                hs2_prefill_4d[:, sf_verify_len:].copy_(
                    prop_hs2_tmp.view(
                        P_pref, sf_P_props * sf_K, 1, H_hs_loc))
                qk_packed3_prefill_4d[:, sf_verify_len:].copy_(
                    prop_qk_tmp.view(
                        P_pref, sf_P_props * sf_K, 1, H_conv_loc))
            else:
                # Eager loopy path (cold prefill, mixed varlen, or
                # non-spec). NOT captured under FULL_DECODE_ONLY — the
                # cudagraph dispatch only fires for uniform shapes, which
                # the gate above covers.
                assert query_start_loc_p_cpu is not None
                assert has_initial_states_p_cpu is not None
                assert state_indices_list is not None
                qsl_p_list = query_start_loc_p_cpu.tolist()
                has_init_p_list = has_initial_states_p_cpu.tolist()
                for i in range(len(qsl_p_list) - 1):
                    start_i = qsl_p_list[i]
                    end_i = qsl_p_list[i + 1]
                    has_init_i = has_init_p_list[i]
                    slot_i = state_indices_list[num_decode_reqs + i]
                    hs2_cur = hs_p[start_i:end_i, :, :]
                    qk_packed0_cur = qk_packed0_p[start_i:end_i, :, :]
                    qk_packed1_cur = qk_packed0_cur.permute(1, 2, 0)

                    if has_init_i:
                        hs2_cached = prev_hs[slot_i].unsqueeze(0).unsqueeze(0)
                        if hs2_cached.dtype != hs2_cur.dtype:
                            hs2_cached = hs2_cached.to(hs2_cur.dtype)
                        hs2_cur = torch.cat([hs2_cached, hs2_cur[:-1]], dim=0)
                        qk_packed0_cached = conv_states[slot_i].unsqueeze(0)
                        if qk_packed0_cached.dtype != qk_packed1_cur.dtype:
                            qk_packed0_cached = qk_packed0_cached.to(
                                qk_packed1_cur.dtype)
                        qk_packed2_cur = torch.cat(
                            [qk_packed0_cached, qk_packed1_cur], dim=-1)
                    else:
                        hs2_cur = F.pad(hs2_cur[:-1], pad=(0, 0, 0, 0, 1, 0))
                        qk_packed2_cur = F.pad(
                            qk_packed1_cur, (self.total_padding, 0))

                    hs2_prefill[start_i:end_i] = hs2_cur

                    conv_states_cur = nn.functional.pad(
                        qk_packed2_cur,
                        (self.cca_time0 - qk_packed2_cur.shape[-1], 0))
                    conv_states[slot_i] = conv_states_cur.to(
                        device=conv_states.device, dtype=conv_states.dtype)
                    prev_hs[slot_i] = hs_p[
                        end_i - 1, 0, :].to(device=prev_hs.device,
                                            dtype=prev_hs.dtype)

                    qk_packed3_cur = self._conv_qk_decode(
                        qk_packed2_cur).permute(2, 0, 1)
                    qk_packed3_prefill[start_i:end_i] = qk_packed3_cur

        if has_decode:
            # Generation
            # In generation B and S are actually the same in meaning
            # That's why we don't need to transpose qk_packed0
            # qk_packed0_d [S, 1, H]
            decode_is_pad = (state_indices_tensor_d == PAD_SLOT_ID)
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
            qk_packed0_cached_for_compute = qk_packed0_cached
            decode_qk_dtype = qk_packed0_d.dtype
            if qk_packed0_cached_for_compute.dtype != decode_qk_dtype:
                qk_packed0_cached_for_compute = qk_packed0_cached_for_compute.to(decode_qk_dtype)
            qk_packed0_cat = torch.cat([qk_packed0_cached_for_compute, qk_packed0_d.transpose(1, 2)], dim=-1) # [S, H, total_padding + 1]
            qk_packed3_d = self._conv_qk_decode(qk_packed0_cat).transpose(1, 2)  # [S, 1, E]
            qk_packed3[:num_decodes] = qk_packed3_d
            
            new_qk_packed0_cache = qk_packed0_cached.roll(shifts=-1, dims=-1)
            new_qk_packed0_cache[..., -1] = qk_packed0_d[:, 0, :].to(new_qk_packed0_cache.dtype)
            new_qk_packed0_cache = torch.where(
                decode_is_pad.view(-1, 1, 1),
                new_qk_packed0_cache.new_zeros(()),
                new_qk_packed0_cache,
            )
            conv_states[safe_decode_indices] = new_qk_packed0_cache.to(
                device=conv_states.device, dtype=conv_states.dtype)

            hs2_decode = prev_hs[safe_decode_indices].unsqueeze(1) # [S, 1, H]
            hs2_decode = torch.where(
                decode_is_pad.view(-1, 1, 1),
                hs2_decode.new_zeros(()),
                hs2_decode,
            )
            if hs2_decode.dtype != hs.dtype:
                hs2_decode = hs2_decode.to(hs.dtype)
            hs2[:num_decodes] = hs2_decode
            new_prev_hs = hs_d[:, 0, :].to(prev_hs.dtype)
            new_prev_hs = torch.where(
                decode_is_pad.view(-1, 1),
                new_prev_hs.new_zeros(()),
                new_prev_hs,
            )
            prev_hs[safe_decode_indices] = new_prev_hs.to(
                device=prev_hs.device, dtype=prev_hs.dtype)

        del qk_packed0_d
        del qk_packed0_p
        del hs_d
        del hs_p

        # Values from the two time streams
        v1 = self.val_proj1(hs)   # [S, B, latent_k_dim/2]
        v2 = self.val_proj2(hs2)
        value = torch.cat([v1, v2], dim=-1).contiguous()
        value = value.view(num_actual_tokens, batch_size, self.num_k_heads, self.head_dim)  # [S, B, kh, dh]
        del hs2

        # Build queries/keys from conv output + means
        query = qk_packed3[..., :self.latent_q_dim].view(
            num_actual_tokens, batch_size, self.num_q_heads, self.head_dim
        ).float()

        key = qk_packed3[..., self.latent_q_dim:].view(
            num_actual_tokens, batch_size, self.num_k_heads, self.head_dim
        ).float()
        query, key = self._add_grouped_qk_means_inplace(query, key, query_pre, key_base)
        del query_pre
        del key_base
        del qk_packed0
        del qk_packed3

        query, key = self._rms_normalize_qk(query.contiguous(), key.contiguous())
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
