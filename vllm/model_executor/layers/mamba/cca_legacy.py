"""Unmodified CCA decode path used for the fused-route isolation experiment.

This module intentionally mirrors the clean Zaya CCA implementation.  It is
debug-only: the mixed-state extension must not change the compiled fallback
graph for shapes that do not select it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from vllm.forward_context import get_forward_context
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.attention.backends.cca_attn import CCAAttentionMetadata
from vllm.v1.attention.backends.utils import PAD_SLOT_ID


def forward_cuda_legacy(self, hidden_states: torch.Tensor,
                        output: torch.Tensor):
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
        self._forward_no_cache(hidden_states, output)
        return

    num_prefills = attn_metadata.num_prefills
    num_decodes = attn_metadata.num_decode_tokens
    num_prefill_tokens = attn_metadata.num_prefill_tokens
    has_prefill = num_prefills > 0
    has_decode = num_decodes > 0
    num_actual_tokens = num_decodes + num_prefill_tokens

    hidden_states = hidden_states[:num_actual_tokens]
    hs = hidden_states.unsqueeze(1)
    batch_size = hs.shape[1]

    qk_packed0, v1, delayed_v_state = self._project_all(hs)

    query_pre = qk_packed0[..., :self.latent_q_dim].view(
        *qk_packed0.shape[:2], self.num_q_heads, self.head_dim)
    key_base = qk_packed0[..., self.latent_q_dim:].view(
        *qk_packed0.shape[:2], self.num_k_heads, self.head_dim)

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
    delayed_v_state_d, delayed_v_state_p = torch.split(
        delayed_v_state[:num_actual_tokens],
        [num_decodes, num_prefill_tokens],
        dim=0,
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
        prefill_slice = slice(num_decodes, num_decodes + num_prefill_tokens)
        value_delayed_prefill = value_delayed[prefill_slice]
        qk_packed3_prefill = qk_packed3[prefill_slice]
        for i in range(len(query_start_loc_p) - 1):
            start_i, end_i = query_start_loc_p[i], query_start_loc_p[i + 1]
            qk_packed0_cur = qk_packed0_p[start_i:end_i, :, :]
            delayed_v_state_cur = delayed_v_state_p[start_i:end_i]
            qk_packed1_cur = qk_packed0_cur.permute(1, 2, 0)

            if has_initial_states_p[i]:
                value_delayed_cached = recurrent_states[
                    state_indices_tensor_p[i]].unsqueeze(0).unsqueeze(0)
                if value_delayed_cached.dtype != value_delayed.dtype:
                    value_delayed_cached = value_delayed_cached.to(
                        value_delayed.dtype)
                qk_packed0_cached = conv_states[
                    state_indices_tensor_p[i]].unsqueeze(0)
                if qk_packed0_cached.dtype != qk_packed1_cur.dtype:
                    qk_packed0_cached = qk_packed0_cached.to(
                        qk_packed1_cur.dtype)
                qk_packed2_cur = torch.cat(
                    [qk_packed0_cached, qk_packed1_cur], dim=-1)
            else:
                if self.config.attention_bias:
                    _, _, value_delayed_cached = self._project_all(
                        hs_p.new_zeros(1, 1, self.hidden_size))
                else:
                    value_delayed_cached = delayed_v_state_p.new_zeros(
                        1, 1, self.recurrent_v_dim)
                qk_packed2_cur = F.pad(
                    qk_packed1_cur, (self.total_padding, 0))

            value_delayed_prefill[start_i:end_i] = torch.cat(
                [value_delayed_cached, delayed_v_state_cur[:-1]], dim=0)

            conv_states_cur = F.pad(
                qk_packed2_cur,
                (self.total_padding - qk_packed2_cur.shape[-1], 0),
            )
            conv_states[state_indices_tensor_p[i]] = conv_states_cur.to(
                device=conv_states.device, dtype=conv_states.dtype)
            qk_packed3_cur = self.conv_qk_grouped(
                self.conv_qk_depthwise(qk_packed2_cur)).permute(2, 0, 1)
            qk_packed3_prefill[start_i:end_i] = qk_packed3_cur

        recurrent_states[state_indices_tensor_p] = delayed_v_state_p[
            query_start_loc_p[1:] - 1, 0, :].to(
                device=recurrent_states.device, dtype=recurrent_states.dtype)

    if has_decode:
        assert state_indices_tensor_d is not None
        decode_is_pad = state_indices_tensor_d == PAD_SLOT_ID
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

        qk_packed0_cached = conv_states[safe_decode_indices]
        qk_packed0_cached = torch.where(
            decode_is_pad.view(-1, 1, 1),
            qk_packed0_cached.new_zeros(()),
            qk_packed0_cached,
        )
        qk_packed0_cached_for_compute = qk_packed0_cached
        decode_qk_dtype = qk_packed0_d.dtype
        if qk_packed0_cached_for_compute.dtype != decode_qk_dtype:
            qk_packed0_cached_for_compute = qk_packed0_cached_for_compute.to(
                decode_qk_dtype)
        qk_packed0_cat = torch.cat(
            [qk_packed0_cached_for_compute,
             qk_packed0_d.transpose(1, 2)], dim=-1)
        qk_packed3_d = self._conv_qk_decode(
            qk_packed0_cat).transpose(1, 2)
        qk_packed3[:num_decodes] = qk_packed3_d

        new_qk_packed0_cache = qk_packed0_cached.roll(shifts=-1, dims=-1)
        new_qk_packed0_cache[..., -1] = qk_packed0_d[:, 0, :].to(
            new_qk_packed0_cache.dtype)
        new_qk_packed0_cache = torch.where(
            decode_is_pad.view(-1, 1, 1),
            new_qk_packed0_cache.new_zeros(()),
            new_qk_packed0_cache,
        )
        conv_states[safe_decode_indices] = new_qk_packed0_cache.to(
            device=conv_states.device, dtype=conv_states.dtype)

        value_delayed_decode = recurrent_states[
            safe_decode_indices].unsqueeze(1)
        value_delayed_decode = torch.where(
            decode_is_pad.view(-1, 1, 1),
            value_delayed_decode.new_zeros(()),
            value_delayed_decode,
        )
        if value_delayed_decode.dtype != value_delayed.dtype:
            value_delayed_decode = value_delayed_decode.to(
                value_delayed.dtype)
        value_delayed[:num_decodes] = value_delayed_decode
        new_recurrent_state = delayed_v_state_d[:, 0, :].to(
            recurrent_states.dtype)
        new_recurrent_state = torch.where(
            decode_is_pad.view(-1, 1),
            new_recurrent_state.new_zeros(()),
            new_recurrent_state,
        )
        recurrent_states[safe_decode_indices] = new_recurrent_state.to(
            device=recurrent_states.device, dtype=recurrent_states.dtype)

    del qk_packed0_d
    del qk_packed0_p
    del hs_d
    del hs_p
    del delayed_v_state_d
    del delayed_v_state_p

    value = torch.cat([v1, value_delayed], dim=-1).contiguous()
    value = value.view(
        num_actual_tokens, batch_size, self.num_k_heads, self.head_dim)
    del value_delayed

    query = qk_packed3[..., :self.latent_q_dim].view(
        num_actual_tokens, batch_size, self.num_q_heads,
        self.head_dim).float()
    key = qk_packed3[..., self.latent_q_dim:].view(
        num_actual_tokens, batch_size, self.num_k_heads,
        self.head_dim).float()
    query, key = self._add_grouped_qk_means_inplace(
        query, key, query_pre, key_base)
    del query_pre
    del key_base
    del qk_packed0
    del qk_packed3

    query, key = self._rms_normalize_qk(query.contiguous(),
                                        key.contiguous())
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
