import os
import logging

import triton
import triton.language as tl

import torch
import torch.nn as nn 
import torch.nn.functional as F
from triton import Config

logger = logging.getLogger(__name__)

# Environment variable to select grouped conv1d decode backend:
#   "triton" (default) | "cpp"
_GCONV_BACKEND = os.environ.get("VLLM_GCONV1D_BACKEND", "triton").lower()

_cpp_grouped_conv1d_fn = None

if _GCONV_BACKEND == "cpp":
    try:
        from torch.utils.cpp_extension import load_inline as _load_inline
        _kernel_path = os.environ.get("VLLM_GCONV1D_BACKEND_KERNEL_PATH", None)
        if _kernel_path is None:
            raise ValueError("VLLM_GCONV1D_BACKEND_KERNEL_PATH is not set")
        if os.path.exists(_kernel_path):
            with open(_kernel_path, "r") as _f:
                _cuda_src = _f.read()
            _cpp_wrapper = """
#include <torch/extension.h>
torch::Tensor grouped_conv1d_w2_decode(
    const torch::Tensor& x, const torch::Tensor& weight,
    const std::optional<torch::Tensor>& bias_opt);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("grouped_conv1d_w2_decode", &grouped_conv1d_w2_decode);
}
"""
            _extra = ["-O3"]
            if torch.version.hip is not None:
                _extra.append("-DUSE_ROCM")
            _mod = _load_inline(
                name="gconv1d_w2_decode_cca",
                cpp_sources=[_cpp_wrapper],
                cuda_sources=[_cuda_src],
                extra_cflags=_extra,
                extra_cuda_cflags=_extra,
                verbose=False,
            )
            _cpp_grouped_conv1d_fn = _mod.grouped_conv1d_w2_decode
            logger.info("Loaded C++ grouped_conv1d_w2_decode kernel")
        else:
            logger.warning("C++ kernel source not found at %s, "
                           "falling back to triton", _kernel_path)
            _GCONV_BACKEND = "triton"
    except Exception as e:
        logger.warning("Failed to compile C++ grouped_conv1d_w2_decode: %s, "
                       "falling back to triton", e)
        _GCONV_BACKEND = "triton"

# ---------------------------------------------------------------------------
# CCA decode fused kernel (depthwise conv + grouped conv + qk_mean + L2 norm
# + scale + temperature — all in a single kernel launch).
#
# Controlled by two environment variables:
#   VLLM_CCA_FUSED_ENABLED       "1" / "true" / "yes" to enable (default: off)
#   VLLM_CCA_FUSED_KERNEL_PATH   absolute path to cca_decode_fused.cu
# ---------------------------------------------------------------------------
_CCA_FUSED_ENABLED = os.environ.get(
    "VLLM_CCA_FUSED_ENABLED", "0").lower() in ("1", "true", "yes")
_CCA_FUSED_SHAPE_DEBUG = os.environ.get(
    "VLLM_CCA_FUSED_SHAPE_DEBUG", "0").lower() in ("1", "true", "yes")
_cpp_cca_fused_fn = None

if _CCA_FUSED_ENABLED:
    try:
        from torch.utils.cpp_extension import load_inline as _load_inline_fused
        _fused_kernel_path = os.environ.get(
            "VLLM_CCA_FUSED_KERNEL_PATH", None)
        if _fused_kernel_path is None:
            raise ValueError("VLLM_CCA_FUSED_KERNEL_PATH is not set")
        if os.path.exists(_fused_kernel_path):
            with open(_fused_kernel_path, "r") as _f:
                _fused_cuda_src = _f.read()
            _fused_cpp_wrapper = """
#include <torch/extension.h>
torch::Tensor cca_decode_fused(
    const torch::Tensor& new_token,
    const torch::Tensor& dw_weight,
    const std::optional<torch::Tensor>& dw_bias,
    torch::Tensor& conv_state,
    const torch::Tensor& state_indices,
    const torch::Tensor& gw_weight,
    const std::optional<torch::Tensor>& gw_bias,
    const torch::Tensor& qk_mean,
    const torch::Tensor& temp,
    int64_t num_q_heads,
    double sqrt_head_dim,
    bool clamp_temp);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cca_decode_fused", &cca_decode_fused);
}
"""
            _fused_extra = ["-O3"]
            if torch.version.hip is not None:
                _fused_extra.append("-DUSE_ROCM")
            _fused_mod = _load_inline_fused(
                name="cca_decode_fused_cca",
                cpp_sources=[_fused_cpp_wrapper],
                cuda_sources=[_fused_cuda_src],
                extra_cflags=_fused_extra,
                extra_cuda_cflags=_fused_extra,
                verbose=False,
            )
            _cpp_cca_fused_fn = _fused_mod.cca_decode_fused
            logger.info("Loaded C++ cca_decode_fused kernel from %s",
                        _fused_kernel_path)
        else:
            logger.warning("CCA fused kernel source not found at %s, "
                           "disabling fusion", _fused_kernel_path)
            _CCA_FUSED_ENABLED = False
    except Exception as e:
        logger.warning("Failed to compile CCA fused kernel: %s, "
                       "disabling fusion", e)
        _CCA_FUSED_ENABLED = False

@triton.jit()
def _causal_conv1d_update_kernel(
    # Pointers to matrices
    x_ptr,  # (batch, dim, seqlen)
    w_ptr,  # (dim, width)
    bias_ptr,
    conv_state_ptr,
    conv_state_indices_ptr,
    query_start_loc_ptr,  # (batch + 1)
    o_ptr,  # (batch, dim, seqlen)
    # Matrix dimensions
    batch: int,
    dim: tl.constexpr,
    seqlen: tl.constexpr,
    state_len: tl.constexpr,
    num_cache_lines: tl.constexpr,  # added to support vLLM larger cache lines
    # is_valid,
    # Strides
    stride_x_seq: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_conv_state_seq: tl.constexpr,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_state_indices: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    # others
    pad_slot_id: tl.constexpr,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    idx_seq = tl.program_id(0)
    if idx_seq >= batch:
        return 
    
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N) 
    mask_feat = idx_feats < dim 

    # Always use state_indices_tensor_d
    cache_row = tl.load(
        conv_state_indices_ptr + idx_seq,
        mask = True,
        other = pad_slot_id,
    ).to(tl.int64)

    valid_row = (cache_row >= 0) & (cache_row < num_cache_lines)

    if USE_PAD_SLOT:
        valid_row = valid_row & (cache_row != pad_slot_id)
        
    mask_c = mask_feat & valid_row
    
    if IS_VARLEN:
        query_start_index = tl.load(query_start_loc_ptr + idx_seq).to(tl.int64)
        query_end_index = tl.load(query_start_loc_ptr + (idx_seq + 1)).to(
            tl.int64)
        # revise state_len and seqlen
        state_len = state_len - (seqlen -
                                 (query_end_index - query_start_index))
        seqlen = query_end_index - query_start_index
        x_offset = query_start_index * stride_x_token
        o_offset = query_start_index * stride_o_token
    else:
        query_start_index = idx_seq * seqlen
        query_end_index = query_start_index + seqlen
        x_offset = idx_seq * stride_x_seq
        o_offset = idx_seq * stride_o_seq

    if query_start_index == query_end_index:
        return
    
    mask_c = mask_feat & valid_row
    cache_row_safe = tl.where(valid_row, cache_row, 0).to(tl.int64)
    # Base pointers
    x_base = x_ptr + idx_seq * stride_x_seq + idx_feats * stride_x_dim
    conv_states_base = conv_state_ptr + cache_row_safe * stride_conv_state_seq + idx_feats * stride_conv_state_dim
    w_base = w_ptr + idx_feats * stride_w_dim

    # Load cached tokens (length 2)
    x0 = tl.load(conv_states_base + 0 * stride_conv_state_tok, mask=mask_c, other=0.0).to(tl.float32)
    x1 = tl.load(conv_states_base + 1 * stride_conv_state_tok, mask=mask_c, other=0.0).to(tl.float32)
    # Load new token (decode => seqlen == 1 => token offset 0)
    x2 = tl.load(x_base + 0 * stride_x_token, mask=mask_c, other=0.0).to(tl.float32)
    # Load weights
    w0 = tl.load(w_base + 0 * stride_w_width, mask=mask_c, other=0.0).to(tl.float32)
    w1 = tl.load(w_base + 1 * stride_w_width, mask=mask_c, other=0.0).to(tl.float32)

    # Bias
    if HAS_BIAS:
        b = tl.load(bias_ptr + idx_feats, mask=mask_c, other=0.0).to(tl.float32)
    else:
        b = tl.zeros((BLOCK_N,), dtype=tl.float32)
    tl.debug_barrier()
    # Compute last two outputs of length-3 causal conv (y1, y2)
    y1 = b + w0 * x0 + w1 * x1
    y2 = b + w0 * x1 + w1 * x2

    # Store outputs to (batch, dim, 2)
    o_base = o_ptr + idx_seq * stride_o_seq + idx_feats * stride_o_dim
    tl.store(o_base + 0 * stride_o_token, y1, mask=mask_c)
    tl.store(o_base + 1 * stride_o_token, y2, mask=mask_c)

    # Update state: roll(-1) then append new token
    # new_state[0] = old_state[1] (= x1), new_state[1] = new_token (= x2)
    # (store as original dtype)
    tl.store(conv_states_base + 0 * stride_conv_state_tok, x1.to(tl.bfloat16), mask=mask_c)
    tl.store(conv_states_base + 1 * stride_conv_state_tok, x2.to(tl.bfloat16), mask=mask_c)


@triton.jit
def _grouped_conv1d_w2_decode_kernel(
    x, y, weight, bias,
    G: tl.constexpr, D: tl.constexpr, W: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):

    i_n = tl.program_id(0)   # Batch
    i_g = tl.program_id(1)   # Group

    c = tl.arange(0, D)      # Full D=128
    o_w = tl.arange(0, W)    # W=2
    
    stride_ng = G * D
    base_offset = i_n * stride_ng * W + i_g * D * W
    
    # --- Load Inputs ---
    x_mask = (c[:,None] < D) & (o_w[None,:] < W)
    x_full = tl.load(x + base_offset + c[:,None]*W + o_w[None,:], mask=x_mask, other=0.0).to(tl.float32)  # [128, 2]
    
    b_x_hist, b_x_curr = tl.split(x_full) # Each [128]

    weight_base = weight + i_g * D * D * W
    w_mask = (c[:,None,None] < D) & (c[None,:,None] < D) &  (o_w[None, None, :]<W)
    w_off = c[:, None, None] * D * W + c[None, :, None] * W + o_w[None, None, :]  # FIX: add o_w
    b_w = tl.load(weight_base + w_off, mask=w_mask, other=0.0).to(tl.float32)   # [128, 128, 2]
    
    b_w_hist, b_w_curr = tl.split(b_w)  # Each [128, 128]

    # --- Compute: acc = w_hist @ x_hist + w_curr @ x_curr ---
    acc = tl.zeros((D,), dtype=tl.float32)
    acc += tl.sum(b_w_hist * b_x_hist, axis=1) + tl.sum(b_w_curr * b_x_curr, axis=1)  # [128]

    if HAS_BIAS:
        acc += tl.load(bias + i_g * D + c, mask=None).to(tl.float32)
    

    tl.store(y + i_n * stride_ng + i_g * D + c, 
             acc, 
             mask=None)


def _grouped_conv1d_decode_triton(x, weight, bias=None):
    B, G, D, W = x.shape
    G, D, _, W = weight.shape
    y = torch.empty(B, G, D, dtype=x.dtype, device=x.device)

    x = x.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    grid = (B, G)
    _grouped_conv1d_w2_decode_kernel[grid](
        x=x,
        y=y,
        weight=weight,
        bias=bias,
        G=G,
        D=D,
        W=W,
        HAS_BIAS=(bias is not None),
    )
    return y


def grouped_conv1d_decode(x, weight, bias=None):
    """Grouped 1D conv (W=2) decode.

    Backend controlled by env var VLLM_GCONV1D_BACKEND:
        "triton" (default) | "cpp"
    """
    if _GCONV_BACKEND == "cpp" and _cpp_grouped_conv1d_fn is not None:
        return _cpp_grouped_conv1d_fn(x.contiguous(),
                                      weight.contiguous(),
                                      bias.contiguous() if bias is not None else None)
    return _grouped_conv1d_decode_triton(x, weight, bias)


def cca_decode_fused_available():
    """Return True when the fused CCA decode kernel is compiled and ready."""
    return _CCA_FUSED_ENABLED and _cpp_cca_fused_fn is not None


def cca_decode_fused(
    new_token,          # [B, E, 1]
    dw_weight,          # [E, 2]
    dw_bias,            # [E] or None
    conv_state,         # [num_cache, E, state_len] — mutated in-place
    state_indices,      # [B] int64
    gw_weight,          # [G, D, D, 2]
    gw_bias,            # [G, D] or None
    qk_mean,            # [B, G, D]
    temp,               # [num_k_heads] float
    num_q_heads,        # int
    sqrt_head_dim,      # float
    clamp_temp,         # bool
):
    """CCA decode fused kernel: depthwise conv + grouped conv + post-processing.

    Returns [B, G*D] with query/key heads already L2-normalised, scaled, and
    temperature-applied.

    Requires VLLM_CCA_FUSED_ENABLED=1 and VLLM_CCA_FUSED_KERNEL_PATH set.
    """
    assert _cpp_cca_fused_fn is not None, (
        "cca_decode_fused called but C++ kernel not loaded. "
        "Set VLLM_CCA_FUSED_ENABLED=1 and VLLM_CCA_FUSED_KERNEL_PATH."
    )
    if _CCA_FUSED_SHAPE_DEBUG:
        logger.info(
            "[cca_decode_fused] input shapes: "
            "new_token=%s, dw_weight=%s, dw_bias=%s, conv_state=%s, "
            "state_indices=%s, gw_weight=%s, gw_bias=%s, qk_mean=%s, temp=%s",
            tuple(new_token.shape),
            tuple(dw_weight.shape),
            None if dw_bias is None else tuple(dw_bias.shape),
            tuple(conv_state.shape),
            tuple(state_indices.shape),
            tuple(gw_weight.shape),
            None if gw_bias is None else tuple(gw_bias.shape),
            tuple(qk_mean.shape),
            tuple(temp.shape),
        )

    out = _cpp_cca_fused_fn(
        new_token.contiguous(),
        dw_weight.contiguous(),
        dw_bias.contiguous() if dw_bias is not None else None,
        conv_state,
        state_indices.to(torch.int64).contiguous(),
        gw_weight.contiguous(),
        gw_bias.contiguous() if gw_bias is not None else None,
        qk_mean.contiguous(),
        temp.float().contiguous(),
        int(num_q_heads),
        float(sqrt_head_dim),
        bool(clamp_temp),
    )
    if _CCA_FUSED_SHAPE_DEBUG:
        logger.info("[cca_decode_fused] output shape: out=%s", tuple(out.shape))
    return out


def run_causal_conv1d_update(
    x,              # Input: (batch, dim, seqlen)
    conv_state,     # State: (batch, dim, state_len)
    weight,         # Weights: (dim, kernel_width)
    bias,           # Bias: (dim,)
    conv_state_indices,
    query_start_loc=None,
    seqlen=1,       # The snippet provided is optimized/unrolled for seqlen 2
):
    """
    Python wrapper to launch the Triton kernel.
    """
    batch, dim, _ = x.shape
    num_cache_lines, _, state_len = conv_state.shape
    kernel_width = weight.shape[1]
    # num_decodes = conv_state_indices.shape[0]
    # Output tensor initialization
    out = torch.empty(batch, dim, seqlen+1, device=x.device, dtype=x.dtype)
    stride_w_dim, stride_w_width = weight.stride()

    stride_conv_state_seq, stride_conv_state_dim, stride_conv_state_token = conv_state.stride(
    )

    BLOCK_N = 32  # Block size for the dimension (channel) axis
    
    grid = lambda META: (
        batch, 
        triton.cdiv(dim, META['BLOCK_N'])
    )
   
    # conv_state_indices = torch.arange(batch, device=x.device, dtype=torch.int64)
    if query_start_loc is None:
        # X (batch, dim, seqlen)
        stride_x_seq, stride_x_dim, stride_x_token = x.stride()
        stride_o_seq, stride_o_dim, stride_o_token = out.stride()
    else:
        # X (dim, cu_seqlen)
        stride_x_token, stride_x_dim = x.stride()
        stride_x_seq = 0
        stride_o_token, stride_o_dim = out.stride()
        stride_o_seq = 0

    stride_state_indices = conv_state_indices.stride(
        0) if conv_state_indices is not None else 0
    
    # Define Next Power of 2 for State Length (optimization often used in Triton)
    np2_statelen = triton.next_power_of_2(state_len)

    _causal_conv1d_update_kernel[grid](
        # Pointers
        x,
        weight,
        bias,
        conv_state,
        conv_state_indices,
        query_start_loc,
        out,
        # Matrix Dimensions
        batch,
        dim,
        seqlen,
        state_len,
        num_cache_lines, # For simple case, cache lines = batch size
        # Strides (The kernel expects specific layouts)
        stride_x_seq,
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_conv_state_seq,
        stride_conv_state_dim,
        stride_conv_state_token,
        stride_state_indices,
        stride_o_seq,
        stride_o_dim,
        stride_o_token,
        # Constants / Others
        pad_slot_id=-1,
        # Meta-parameters (Compile-time constants)
        HAS_BIAS=True if bias is not None else False,
        KERNEL_WIDTH=kernel_width,
        IS_VARLEN=False,            # Set to True if using query_start_loc 
        NP2_STATELEN=np2_statelen,
        USE_PAD_SLOT=True,
        BLOCK_N=BLOCK_N,
    )
    
    return out





#########    TO BE DELETED AT DISCRETION ###############

# @triton.autotune(
#     configs=configs,
#     key=["batch_seq", "dim_q", "dim_k", "num_q_heads", "num_k_heads"]
# )
# @triton.jit
# def _qk_norm_fwd_kernel(
#     q_pre_ptr,
#     k_pre_ptr,
#     q_ptr,
#     k_ptr,
#     out_k_ptr,
#     out_q_ptr,
#     out_qn_ptr,
#     out_kn_ptr,
#     temp_k_ptr,
#     qn_b_stride,
#     qn_h_stride,
#     kn_b_stride,
#     kn_h_stride,
#     q_b_strides,
#     q_h_strides,
#     q_c_strides,
#     k_b_strides,
#     k_h_strides,
#     k_c_strides,
#     batch_seq:tl.constexpr,
#     dim_q:tl.constexpr,
#     dim_k:tl.constexpr,
#     num_q_heads:tl.constexpr,
#     num_k_heads:tl.constexpr,
#     n_gqa:tl.constexpr,
#     BLOCK_BT:tl.constexpr,
#     BLOCK_DIM: tl.constexpr,
#     F16: tl.constexpr,

# ):  
#     tl.static_assert(
#     num_q_heads == num_k_heads * n_gqa,
#     "num_q_heads must equal num_k_heads * n_gqa"
#     )

#     pid0 = tl.program_id(0)
#     pid1= tl.program_id(1)

  
#     q_pre_block_ptr = tl.make_block_ptr(
#         base=q_pre_ptr,
#         shape=(batch_seq, num_q_heads, dim_q),
#         strides=(q_b_strides, q_h_strides, q_c_strides),
#         offsets=(pid0 * BLOCK_BT, 0, pid1 * BLOCK_DIM),
#         block_shape=(BLOCK_BT, num_q_heads, BLOCK_DIM),
#         order=(0, 1, 2)
#     )

#     k_pre_block_ptr = tl.make_block_ptr(
#         base=k_pre_ptr,
#         shape=(batch_seq, num_q_heads, dim_q),  # Same heads/channels as q_pre
#         strides=(q_b_strides, q_h_strides, q_c_strides),
#         offsets=(pid0 * BLOCK_BT, 0, pid1 * BLOCK_DIM),
#         block_shape=(BLOCK_BT, num_q_heads, BLOCK_DIM),
#         order=(0, 1, 2)
#     )
    
    
#     q_pre = tl.load(q_pre_block_ptr, boundary_check=(0,), padding_option="zero")
#     k_pre = tl.load(k_pre_block_ptr, boundary_check=(0,), padding_option="zero")


#     ptrs_q = tl.make_block_ptr(
#         base = q_ptr,
#         shape=(batch_seq, num_q_heads, dim_q),
#         strides=(q_b_strides, q_h_strides, q_c_strides),
#         offsets=(pid0*BLOCK_BT, 0, pid1 * BLOCK_DIM),
#         block_shape=(BLOCK_BT, num_q_heads, BLOCK_DIM),
#         order = (0, 1, 2)
#     )

#     ptrs_k = tl.make_block_ptr(
#         base=k_ptr,
#         shape=(batch_seq, num_k_heads, dim_k),
#         strides=(k_b_strides, k_h_strides, k_c_strides),
#         offsets=(pid0*BLOCK_BT, 0,  pid1 * BLOCK_DIM),
#         block_shape=(BLOCK_BT, num_k_heads, BLOCK_DIM),
#         order = (0,1,2),
#     )

#     temp_ptr = temp_k_ptr + tl.arange(0,num_k_heads)
#     temp_k = tl.load(temp_ptr)

#     temp_k = temp_k[None,:,None]
    
#     q = tl.load(ptrs_q,  boundary_check=(0,), padding_option="zero")
#     k = tl.load(ptrs_k,  boundary_check=(0,), padding_option="zero")

#     qk_sum = (q_pre + k_pre)/2
#     q += qk_sum
#     qk_sum_k = tl.sum(tl.reshape(qk_sum, (BLOCK_BT, num_k_heads, n_gqa, dim_k)), axis=2)/n_gqa
#     k += qk_sum_k
#     q_sq = q*q
#     q_sum_sq = tl.sum(q_sq, axis=2, keep_dims=True) 
#     q_norms = tl.sqrt(q_sum_sq + 1e-10)  


#     k_sq = k*k
#     k_sum_sq = tl.sum(k_sq, axis=2, keep_dims=True) 
#     k_norms = tl.sqrt(k_sum_sq + 1e-10)

#     q /= q_norms 
#     k /= k_norms

#     if F16:
#         q = q.to(tl.float32)
#         k = k.to(tl.float32)
#         temp_k = temp_k.to(tl.float32)

#     q *= tl.sqrt(tl.cast(dim_q, tl.float32))
#     k *= tl.sqrt(tl.cast(dim_k, tl.float32))*tl.exp(temp_k)

#     if F16:
#         q = q.to(tl.bfloat16)
#         k = k.to(tl.bfloat16)
#         temp_k = temp_k.to(tl.bfloat16)


    
#     store_k_ptrs = tl.make_block_ptr(
#         base=out_k_ptr,
#         shape = (batch_seq, num_k_heads, dim_k),
#         block_shape=(BLOCK_BT, num_k_heads, BLOCK_DIM),
#         strides = (k_b_strides, k_h_strides, k_c_strides),
#         offsets = (pid0*BLOCK_BT, 0, pid1*BLOCK_DIM),
#         order = (0,1,2),

#     )
    
#     store_kn_ptrs = tl.make_block_ptr(
#         base=out_kn_ptr,
#         shape = (batch_seq, num_k_heads),
#         block_shape=(BLOCK_BT, num_k_heads),
#         strides = (kn_b_stride, kn_h_stride),
#         offsets = (pid0*BLOCK_BT, 0),
#         order = (0,1),

#     )

#     store_qn_ptrs = tl.make_block_ptr(
#         base=out_qn_ptr,
#         shape = (batch_seq, num_q_heads),
#         block_shape=(BLOCK_BT, num_q_heads),
#         strides = (qn_b_stride, qn_h_stride),
#         offsets = (pid0*BLOCK_BT, 0),
#         order = (0,1),

#     )


#     store_q_ptrs = tl.make_block_ptr(
#         base=out_q_ptr,
#         shape = (batch_seq, num_q_heads, dim_q),
#         block_shape=(BLOCK_BT, num_q_heads,  BLOCK_DIM),
#         strides = (q_b_strides, q_h_strides, q_c_strides),
#         offsets = (pid0*BLOCK_BT, 0, pid1*BLOCK_DIM),
#         order = (0,1,2),

#     )

#     tl.store(store_k_ptrs, k)
#     tl.store(store_q_ptrs, q)
#     tl.store(store_qn_ptrs, tl.reshape(q_norms,(BLOCK_BT, num_q_heads)))
#     tl.store(store_kn_ptrs, tl.reshape(k_norms,(BLOCK_BT,num_k_heads)))


# def qk_norm_fwd(q_pre, k_pre, query, key, temp):#, BLOCK_BT = 2):

#     bt, hq, cq = query.shape
#     bt, hk, ck = key.shape

#     F16 = query.dtype==torch.bfloat16

#     assert q_pre.shape == query.shape == k_pre.shape
#     n_gqa = hq//hk 
   
#     assert temp.shape == (hk,)

#     grid = lambda meta: (triton.cdiv(bt, meta['BLOCK_BT']),triton.cdiv(cq, meta['BLOCK_DIM']))

#     out_q = torch.empty_like(query)
#     out_k = torch.empty_like(key)

#     out_qn = torch.zeros((bt, hq), device=query.device)
#     out_kn = torch.zeros((bt, hk), device=query.device)
 

#     _qk_norm_fwd_kernel[grid](q_pre.contiguous(), k_pre.contiguous(), 
#                               query.contiguous(), key.contiguous(), 
#                               out_k, out_q, temp_k_ptr=temp.contiguous(),
#                               out_qn_ptr = out_qn.contiguous(), out_kn_ptr = out_kn.contiguous(),
#                               qn_b_stride = out_qn.stride(0), qn_h_stride = out_qn.stride(1), 
#                               kn_b_stride= out_kn.stride(0),
#                               kn_h_stride= out_kn.stride(1),
#                               batch_seq=bt, dim_q = cq, dim_k = ck, 
#                               num_q_heads=hq, num_k_heads=hk,
#                               n_gqa = n_gqa,
#                               q_b_strides=query.stride(0),
#                               q_h_strides=query.stride(1),
#                               q_c_strides=query.stride(2),
#                               k_b_strides=key.stride(0),
#                               k_h_strides=key.stride(1),
#                               k_c_strides=key.stride(2),
#                               BLOCK_DIM=triton.next_power_of_2(cq),
#                               F16=F16,
#                               #BLOCK_BT=BLOCK_BT,
#     )

#     return out_q, out_k, out_qn, out_kn

# # Autotune configurations - experiment with different block sizes
# # Autotune configurations - experiment with different block sizes
# configs = [
#     Config({"BLOCK_BT":1}, num_warps =4),
#     Config({"BLOCK_BT":2}, num_warps =8),
#     Config({"BLOCK_BT":4}, num_warps =8),
#     Config({"BLOCK_BT":8}, num_warps =8),
#     Config({"BLOCK_BT":16}, num_warps =8),
#     Config({"BLOCK_BT":32}, num_warps =8),

#     # Config({"BLOCK_BT":64}),
#     # Config({"BLOCK_BT":128}),ß

  
# ]

# @triton.autotune(
#     configs=configs,
#     key=["batch_seq", "dim_q", "dim_k", "num_q_heads", "num_k_heads"]
# )

# @triton.jit
# def _qk_norm_dq_dk_bwd_kernel(
#     dq_ptr, #to write
#     dk_ptr, #to write
#     dqo_ptr, #input to bwd d_out_q
#     dko_ptr, #input to bwd d_out_k
#     dtemp_ptr,
#     dqpre_ptr,
#     qo_ptr, 
#     ko_ptr,
#     qn_ptr,
#     kn_ptr,
#     temp_k_ptr,
#     q_b_strides,
#     qn_b_strides,
#     qn_h_strides,
#     kn_b_strides,
#     q_h_strides,
#     q_c_strides,
#     k_b_strides,
#     k_h_strides,
#     kn_h_strides,
#     k_c_strides,
#     batch_seq: tl.constexpr,
#     dim_q: tl.constexpr,
#     dim_k: tl.constexpr,
#     num_q_heads: tl.constexpr,
#     num_k_heads: tl.constexpr,
#     n_gqa: tl.constexpr,
#     BLOCK_BT: tl.constexpr,
#     BLOCK_DIM:tl.constexpr,
#     F16:tl.constexpr,
# ):
#     tl.static_assert(
#         num_q_heads == num_k_heads * n_gqa,
#         "num_q_heads must equal num_k_heads * n_gqa"
#     )

#     pid0 = tl.program_id(0)
#     pid1 = tl.program_id(1)


#     ptrs_dqo = tl.make_block_ptr(
#         base = dqo_ptr,
#         shape=(batch_seq, num_q_heads, dim_q),
#         strides=(q_b_strides, q_h_strides, q_c_strides),
#         offsets=(pid0*BLOCK_BT, 0, pid1*BLOCK_DIM),
#         block_shape=(BLOCK_BT, num_q_heads, BLOCK_DIM),
#         order = (0, 1, 2)
#     )

#     ptrs_dko = tl.make_block_ptr(
#         base=dko_ptr,
#         shape=(batch_seq, num_k_heads, dim_k),
#         strides=(k_b_strides, k_h_strides, k_c_strides),
#         offsets=(pid0*BLOCK_BT, 0, pid1*BLOCK_DIM),
#         block_shape=(BLOCK_BT, num_k_heads, BLOCK_DIM),
#         order = (0,1,2),
#     )

#     ptrs_dq = tl.make_block_ptr(
#         base = dq_ptr,
#         shape=(batch_seq, num_q_heads, dim_q),
#         strides=(q_b_strides, q_h_strides, q_c_strides),
#         offsets=(pid0*BLOCK_BT, 0, pid1*BLOCK_DIM),
#         block_shape=(BLOCK_BT, num_q_heads, BLOCK_DIM),
#         order = (0, 1, 2)
#     )

#     ptrs_dk = tl.make_block_ptr(
#         base=dk_ptr,
#         shape=(batch_seq, num_k_heads, dim_k),
#         strides=(k_b_strides, k_h_strides, k_c_strides),
#         offsets=(pid0*BLOCK_BT, 0, pid1*BLOCK_DIM),
#         block_shape=(BLOCK_BT, num_k_heads, BLOCK_DIM),
#         order = (0,1,2),
#     )

#     qn_ptr = tl.make_block_ptr(
#         base = qn_ptr,
#         shape=(batch_seq, num_q_heads),
#         strides= (qn_b_strides, qn_h_strides),
#         offsets = (pid0*BLOCK_BT, 0),
#         block_shape = (BLOCK_BT, num_q_heads),
#         order= (0,1),
#     )

#     kn_ptr = tl.make_block_ptr(
#         base = kn_ptr,
#         shape=(batch_seq, num_k_heads),
#         strides= (kn_b_strides, kn_h_strides),
#         offsets = (pid0*BLOCK_BT, 0),
#         block_shape = (BLOCK_BT, num_k_heads),
#         order= (0,1),
#     )

#     ptrs_qo = tl.make_block_ptr(
#         base = qo_ptr,
#         shape=(batch_seq, num_q_heads, dim_q),
#         strides=(q_b_strides, q_h_strides, q_c_strides),
#         offsets=(pid0*BLOCK_BT, 0, pid1*BLOCK_DIM),
#         block_shape=(BLOCK_BT, num_q_heads, BLOCK_DIM),
#         order = (0, 1, 2)
#     )


#     ptrs_ko = tl.make_block_ptr(
#         base=ko_ptr,
#         shape=(batch_seq, num_k_heads, dim_k),
#         strides=(k_b_strides, k_h_strides, k_c_strides),
#         offsets=(pid0*BLOCK_BT, 0, pid1*BLOCK_DIM),
#         block_shape=(BLOCK_BT, num_k_heads, BLOCK_DIM),
#         order = (0,1,2),
#     )
    

#     q_o = tl.load(ptrs_qo,  boundary_check=(0,), padding_option="zero")
#     k_o = tl.load(ptrs_ko,  boundary_check=(0,), padding_option="zero")
#     qn = tl.load(qn_ptr, boundary_check=(0,), padding_option="zero")
#     kn = tl.load(kn_ptr, boundary_check=(0,), padding_option="zero")
    
#     k_dim = tl.sqrt(tl.cast(dim_k, tl.float32))
#     q_dim = tl.sqrt(tl.cast(dim_q, tl.float32))

    
#     temp_ptr = temp_k_ptr + tl.arange(0,num_k_heads)
#     temp_k = tl.load(temp_ptr)
#     temp_k = temp_k[None,:,None]

#     dqo = tl.load(ptrs_dqo,  boundary_check=(0,), padding_option="zero")
#     dko = tl.load(ptrs_dko,  boundary_check=(0,), padding_option="zero")

    
    
#     cq = tl.sum(dqo * q_o, axis=-1)
#     dq_ = (q_dim*dqo -  (1/q_dim)*q_o * cq[:,:,None])/qn[:,:,None]


#     ck = tl.sum(dko * k_o, axis=-1, keep_dims=True)
    
#     if F16:
#         dko = dko.to(tl.float32)
#         kn = kn.to(tl.float32)
#         temp_k = temp_k.to(tl.float32)
        
        

#     dk_ = (tl.cast(k_dim, tl.float32)*dko -  k_o * tl.cast(ck, tl.float32) * tl.exp(-temp_k))/kn[:,:,None]


#     dq_pre = 0.5*(dq_ + tl.reshape(tl.broadcast_to(tl.reshape(dk_, (BLOCK_BT, num_k_heads, 1, BLOCK_DIM)),
#                                               (BLOCK_BT, num_k_heads, n_gqa, BLOCK_DIM)),
#                                               BLOCK_BT, num_q_heads, BLOCK_DIM)/n_gqa)
#     if F16:
#         dk_ = dk_.to(tl.bfloat16)
#         dq_ = dq_.to(tl.bfloat16)
#         dq_pre = dq_pre.to(tl.bfloat16)

#     dq_pre_block_ptr = tl.make_block_ptr(
#         base=dqpre_ptr,
#         shape=(batch_seq, num_q_heads, dim_q),
#         strides=(q_b_strides, q_h_strides, q_c_strides),
#         offsets=(pid0 * BLOCK_BT, 0, pid1*BLOCK_DIM),
#         block_shape=(BLOCK_BT, num_q_heads, BLOCK_DIM),
#         order=(0, 1, 2)
#     )

#     dtemp_loc = tl.sum(tl.sum(k_o*dko*tl.exp(temp_k)*tl.cast(k_dim, tl.float32), axis=0, keep_dims=True), axis=2, keep_dims=True)
#     dtemp_loc = dtemp_loc.reshape(num_k_heads,)
#     if F16:
#         dtemp_loc = dtemp_loc.to(tl.float32)

#     dtemp_ptr = dtemp_ptr + tl.arange(0,num_k_heads)

#     tl.store(ptrs_dq, dq_)
#     tl.store(ptrs_dk, dk_)
#     tl.store(dq_pre_block_ptr, dq_pre)
#     tl.atomic_add(dtemp_ptr, dtemp_loc)


# class QKNormFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, q_pre, k_pre, query, key, temp):
        
#         # Forward pass
#         out_q, out_k, out_qn,out_kn = qk_norm_fwd(q_pre, k_pre, query, key, temp)#, BLOCK_BT)

#         # Save tensors for backward pass
#         ctx.save_for_backward(out_q, out_k, out_qn, out_kn, temp, q_pre, k_pre, query, key)
#         return out_q, out_k
    
#     @staticmethod
#     def backward(ctx, grad_output_q, grad_output_k):
#         out_q, out_k, out_qn, out_kn, temp, q_pre, k_pre, query, key = ctx.saved_tensors
#         #BLOCK_BT = ctx.BLOCK_BT
        
#         bt, hq, cq = out_q.shape
#         bt, hk, ck = out_k.shape
#         n_gqa = hq // hk
        
#         # Initialize gradients
#         grad_q_pre = torch.zeros_like(q_pre)
#         grad_query = torch.empty_like(query)
#         grad_key = torch.empty_like(key)
#         grad_temp = torch.zeros_like(temp).to(torch.float32)
#         F16 = q_pre.dtype==torch.bfloat16

        
#         grid = lambda meta: ((triton.cdiv(bt, meta['BLOCK_BT']),1))
        
#         # Backward pass
#         _qk_norm_dq_dk_bwd_kernel[grid](
#             dq_ptr=grad_query, dk_ptr=grad_key,
#             dqo_ptr= grad_output_q.contiguous(), dko_ptr=grad_output_k.contiguous(),
#             dtemp_ptr=grad_temp,
#             dqpre_ptr=grad_q_pre.contiguous(),
#             qo_ptr=out_q.contiguous(), ko_ptr=out_k.contiguous(), qn_ptr=out_qn.contiguous(), kn_ptr=out_kn.contiguous(),
#             temp_k_ptr=temp.contiguous(),
#             batch_seq=bt, dim_q=cq, dim_k=ck,
#             num_q_heads=hq, num_k_heads=hk,
#             n_gqa=n_gqa,
#             q_b_strides=query.stride(0),
#             qn_b_strides= out_qn.stride(0),
#             qn_h_strides = out_qn.stride(1),
#             kn_b_strides=out_kn.stride(0),
#             kn_h_strides=out_kn.stride(1),
#             q_h_strides=query.stride(1),
#             q_c_strides=query.stride(2),
#             k_b_strides=key.stride(0),
#             k_h_strides=key.stride(1),
#             k_c_strides=key.stride(2),
#             BLOCK_DIM = triton.next_power_of_2(cq),
#             F16=F16,
#             #BLOCK_BT=BLOCK_BT,

#         )

        
#         return grad_q_pre, grad_q_pre, grad_query, grad_key, grad_temp, None


# # Convenience function to use the custom autograd function

# def qk_norm(q_pre, k_pre, query, key, temp):
#     return QKNormFunction.apply(q_pre, k_pre, query, key, temp)#, BLOCK_BT)




# @torch.compile()
# def qk_norm_ref(q_pre, k_pre, query, key, temp):

#     bt, hq, cq = q_pre.shape 

#     assert q_pre.shape==query.shape==k_pre.shape

#     bt, hk, ck = key.shape

#     gqa = hq//hk

#     qk_m = (q_pre+k_pre)/2

#     query = query+ qk_m

#     key = key+ (qk_m.view(bt, hk, gqa, -1).sum(-2))/gqa

#     qn = query.norm(p=2, dim=-1, keepdim=True)

#     kn = key.norm(p=2, dim=-1, keepdim=True)

#     query = query/qn
#     key =key/kn

#     query = query*torch.sqrt(torch.tensor(cq))
#     key= key*torch.sqrt(torch.tensor(ck))*torch.exp(temp[None,:,None])

#     return query, key

# def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
#     return cu_seqlens[1:] - cu_seqlens[:-1]

# def prepare_chunk_indices(
#     cu_seqlens: torch.LongTensor,
#     chunk_size: int,
# ) -> torch.LongTensor:
#     indices = torch.cat([torch.arange(n) for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()])
#     return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens.device)

        
# @triton.jit
# def _grouped_conv_varlen_fwd_triton(
#     x_ptr, w_ptr, b_ptr, o_ptr, 
#     cu_seqlens, chunk_indices,
#     # Dense tensor strides
#     x_g_stride, x_c_stride, x_t_stride,
#     w_g_stride, w_oc_stride, w_ic_stride, w_k_stride,
#     o_g_stride, o_c_stride, o_t_stride, b_g_stride, b_c_stride,  # ADDED: Explicit bias strides
#     HAS_BIAS: tl.constexpr, G: tl.constexpr,
#     h_dim: tl.constexpr, BT: tl.constexpr, width: tl.constexpr,
#     BD: tl.constexpr,
# ):
#     #3D Grid: (channel_blocks, time_chunks, groups)
#     pid_c = tl.program_id(0)  # Channel block
#     pid_t = tl.program_id(1)  # Time chunk (flattened across sequences)
#     pid_g = tl.program_id(2)  # Group (parallelized!)

#     # === Varlen Logic: Resolve chunk to (sequence, offset) - logic from FLA ===
#     seq_idx = tl.load(chunk_indices + pid_t * 2).to(tl.int32)
#     chunk_idx = tl.load(chunk_indices + pid_t * 2 + 1).to(tl.int32)
    
#     bos = tl.load(cu_seqlens + seq_idx).to(tl.int64)
#     eos = tl.load(cu_seqlens + seq_idx + 1).to(tl.int64)
#     T = eos - bos


#     c_out = pid_c * BD + tl.arange(0, BD)
#     msk_c_out = c_out < h_dim
    
#     c_in = tl.arange(0, h_dim)  # Full reduction dimension
#     msk_c_in = c_in < h_dim
    
#     t_out = chunk_idx * BT + tl.arange(0, BT)
#     msk_t_out = t_out < T

#     # Initialize Accumulator [BD, BT]
#     acc = tl.zeros((BD, BT), dtype=tl.float32)

#     # Load Bias (clean block pointer)
#     if HAS_BIAS:
#         b_offs = b_ptr + pid_g * b_g_stride + c_out * b_c_stride
#         b_slc = tl.load(b_offs, mask=msk_c_out, other=0.0)  # [BD]
#         acc += b_slc[:, None]

#     # Convolution Loop over Kernel Width 
#     for k in range(-width+1, 1):
#     #for k in range(width):
#         # --- Load Weight (block pointer for clean channel access) ---
#         # w: [G, h_dim, h_dim, width] -> w[pid_g, c_out, c_in, k]
#         w_offs = w_ptr + pid_g * w_g_stride + (k+width-1)* w_k_stride#k * w_k_stride
#         w_ptr_block = tl.make_block_ptr(
#             base=w_offs,
#             shape=(h_dim, h_dim),  # [out_ch, in_ch]
#             block_shape=(BD, h_dim),  # [BD, VD]
#             offsets=(pid_c * BD, 0),
#             strides=(w_oc_stride, w_ic_stride),
#             order=(0, 1)
#         ) 
#         w_slc = tl.load(w_ptr_block, boundary_check=(0, 1))  # [BD, h_dim]

#         # x: [G, h_dim, T_total + width - 1] -> x[pid_g, c_in, bos + t_out + k]
#         x_offs = (x_ptr + 
#                   pid_g * x_g_stride + 
#                   c_in[None, :, None] * x_c_stride + 
#                   (bos + t_out[None, None, :] + k) * x_t_stride)
#         msk_t_in = (t_out + k)>=0
#         x_slc = tl.load(x_offs, 
#                        mask=msk_c_in[None, :, None] & msk_t_in[None, None, :], 
#                        other=0.0)  # [1, h_dim, BT]
#         x_slc = tl.reshape(x_slc, (h_dim, BT))  # [h_dim, BT]

#         # --- Accumulate with efficient dot ---
#         acc += tl.dot(w_slc, x_slc)  # [BD, h_dim] @ [h_dim, BT] -> [BD, BT]

#     # === Store Output (MANUAL offset for varlen time) ===
#     # o: [G, h_dim, T_total] -> o[pid_g, c_out, bos + t_out]
#     o_offs = (o_ptr + 
#               pid_g * o_g_stride + 
#               c_out[:, None] * o_c_stride + 
#               (bos + t_out[None, :]) * o_t_stride)
#     tl.store(o_offs, acc, mask=msk_c_out[:, None] & msk_t_out[None, :])

# def g_conv1d_varlen_fwd(x, w, b, width, num_groups, cu_seqlens, chunk_indices):
#     D, T_total = x.shape
#     G = num_groups
#     h_dim = D // G
    
#     # Reshape to logical dimensions

#     x = x.reshape(G, h_dim, T_total).contiguous()
#     w = w.reshape(G, h_dim, h_dim, width).contiguous()
#     b = b.reshape(G, h_dim).contiguous()
#     o = torch.empty((G, h_dim, T_total), device=x.device, dtype=x.dtype)
    
#     # Prepare chunking (if not provided)
#     if cu_seqlens is not None:
#         max_len = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
#         BT = min(64, triton.next_power_of_2(max(max_len // 16, 16)))
#         chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
#     else:
#         cu_seqlens = torch.tensor([0, T_total], device=x.device, dtype=torch.int32)
#         BT =min(64, triton.next_power_of_2(max(T_total // 16, 16)))
#         chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    
#     NT = len(chunk_indices)
    
#     # Launch with 3D grid
#     grid = lambda meta: (triton.cdiv(h_dim, meta['BD']), NT, G)
    
#     # Extract strides for the logical shapes
#     x_strides = x.stride(0), x.stride(1), x.stride(2)
#     w_strides = w.stride(0), w.stride(1), w.stride(2), w.stride(3)
#     o_strides = o.stride(0), o.stride(1), o.stride(2)
#     b_strides = (b.stride(0), b.stride(1)) if b is not None else (0, 0)
    
#     _grouped_conv_varlen_fwd_triton[grid](
#         x_ptr=x, w_ptr=w, b_ptr=b, o_ptr=o,
#         cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
#         x_g_stride=x_strides[0], x_c_stride=x_strides[1], x_t_stride=x_strides[2],
#         w_g_stride=w_strides[0], w_oc_stride=w_strides[1], w_ic_stride=w_strides[2], w_k_stride=w_strides[3],
#         o_g_stride=o_strides[0], o_c_stride=o_strides[1], o_t_stride=o_strides[2],
#         b_g_stride=b_strides[0], b_c_stride=b_strides[1],  
#         HAS_BIAS=b is not None, G=G,
#         h_dim=h_dim, BT=BT, width=width,
#         BD=128,
#     )
    
#     return o.reshape(D, T_total)

# # import torch
# # import triton
# # import triton.language as tl 
# # from typing import Optional
# # import torch.nn as nn 
# # import torch.nn.functional as F

# @triton.jit
# def varlen_cache_kernel(
#     x,
#     final_state,
#     cu_seqlens,
#     T,
#     D,
#     W,
#     BD: tl.constexpr,
#     BW: tl.constexpr,

# ):
#     i_d, i_n = tl.program_id(0), tl.program_id(1)

#     bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
#     T = eos - bos
   

#     o_t = eos - BW + tl.arange(0, BW)
#     o_d = i_d * BD + tl.arange(0, BD)
#     o_w = W - BW + tl.arange(0, BW)
#     m_t = (o_t >= tl.maximum(bos, eos - W))
#     m_d = o_d < D
#     m_w = (o_w >= 0) & (o_w < W)

#     b_x = tl.load(x + o_t * D + o_d[:, None], mask=(m_t & m_d[:, None]), other=0)
#     # if USE_INITIAL_STATE:
#     #     if T < BW:
#     #         o_c = W - (BW - T) + tl.arange(0, BW)
#     #         m_c = (o_c >= 0) & (o_c < W)
#     #         b_cache = tl.load(initial_state + i_n * D*W + o_d[:, None] * W + o_c, mask=m_d[:, None] & m_c, other=0)
#     #         b_x += b_cache

#     tl.store(final_state + i_n * D*W + o_d[:, None] * W + o_w, b_x, mask=m_d[:, None] & m_w)


# def cache_states(
#     x: torch.Tensor,
#     state_len: int,
#     # initial_state: torch.Tensor | None = None,
#     cu_seqlens: torch.Tensor | None = None,
# ) -> torch.Tensor:
#     B, T, D, W = *x.shape, state_len
#     N = len(cu_seqlens) - 1 if cu_seqlens is not None else B

#     final_state = torch.empty(N, D, W, dtype=x.dtype, device=x.device)
#     BD = min(triton.next_power_of_2(D), 256)
#     BW = triton.next_power_of_2(W)
#     grid = (triton.cdiv(D, BD), N)
#     varlen_cache_kernel[grid](
#         x=x,
#         final_state=final_state,
#         cu_seqlens=cu_seqlens,
#         T=T,
#         D=D,
#         W=W,
#         BW=BW,
#         BD=BD,
#     )
#     return final_state


# @triton.jit()
# def _causal_conv1d_update_kernel(
#     # Pointers to matrices
#     x_ptr,  # (batch, dim, seqlen)
#     w_ptr,  # (dim, width)
#     bias_ptr,
#     conv_state_ptr,
#     conv_state_indices_ptr,
#     query_start_loc_ptr,  # (batch + 1)
#     o_ptr,  # (batch, dim, seqlen)
#     # Matrix dimensions
#     batch: int,
#     dim: tl.constexpr,
#     seqlen: tl.constexpr,
#     state_len: tl.constexpr,
#     num_cache_lines: tl.constexpr,  # added to support vLLM larger cache lines
#     # Strides
#     stride_x_seq: tl.constexpr,
#     stride_x_dim: tl.constexpr,
#     stride_x_token: tl.constexpr,
#     stride_w_dim: tl.constexpr,
#     stride_w_width: tl.constexpr,
#     stride_conv_state_seq: tl.constexpr,
#     stride_conv_state_dim: tl.constexpr,
#     stride_conv_state_tok: tl.constexpr,
#     stride_state_indices: tl.constexpr,
#     stride_o_seq: tl.constexpr,
#     stride_o_dim: tl.constexpr,
#     stride_o_token: tl.constexpr,
#     # others
#     pad_slot_id: tl.constexpr,
#     # Meta-parameters
#     HAS_BIAS: tl.constexpr,
#     KERNEL_WIDTH: tl.constexpr,
#     IS_VARLEN: tl.constexpr,
#     IS_CONTINUOUS_BATCHING: tl.constexpr,
#     NP2_STATELEN: tl.constexpr,
#     USE_PAD_SLOT: tl.constexpr,
#     BLOCK_N: tl.constexpr,
# ):
#     # ruff: noqa: E501
#     idx_seq = tl.program_id(0)
#     if idx_seq >= batch:
#         return

#     # [BLOCK_N,] elements along the feature-dimension (channel)
#     idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

#     if IS_CONTINUOUS_BATCHING:
#         # mask = idx_seq < batch
#         conv_state_batch_coord = tl.load(conv_state_indices_ptr +
#                                          idx_seq * stride_state_indices).to(
#                                              tl.int64)
#     else:
#         conv_state_batch_coord = idx_seq
#     if USE_PAD_SLOT:  # noqa
#         if conv_state_batch_coord == pad_slot_id:
#             # not processing as this is not the actual sequence
#             return

#     if IS_VARLEN:
#         query_start_index = tl.load(query_start_loc_ptr + idx_seq).to(tl.int64)
#         query_end_index = tl.load(query_start_loc_ptr + (idx_seq + 1)).to(
#             tl.int64)
#         # revise state_len and seqlen
#         state_len = state_len - (seqlen -
#                                  (query_end_index - query_start_index))
#         seqlen = query_end_index - query_start_index
#         x_offset = query_start_index * stride_x_token
#         o_offset = query_start_index * stride_o_token
#     else:
#         query_start_index = idx_seq * seqlen
#         query_end_index = query_start_index + seqlen
#         x_offset = idx_seq * stride_x_seq
#         o_offset = idx_seq * stride_o_seq

#     if query_start_index == query_end_index:
#         return

    
#     # conv_state_token_offset = 0

#     # STEP 1: READ init_state data
#     conv_states_base = (conv_state_ptr +
#                         (conv_state_batch_coord * stride_conv_state_seq) +
#                         (idx_feats * stride_conv_state_dim))
#     mask_w = idx_feats < dim

#     prior_tokens = conv_states_base #+ conv_state_token_offset * stride_conv_state_tok
#     if KERNEL_WIDTH >= 2:
#         conv_states_ptrs = prior_tokens  # [BLOCK_N]
#         col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
#         conv_states_ptrs = prior_tokens + 1 * stride_conv_state_tok  # [BLOCK_N]
#         col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
   

#     # STEP 2: assume state_len > seqlen
#     idx_tokens = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

#     # With speculative decoding, the conv_state updates works in a sliding
#     # window manner, at each forward pass, the tokens are shift by 1, so we
#     # load since idx_tokens + 1.
#     conv_state_ptrs_source = (
#         conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) +
#         # conv_state_token_offset * stride_conv_state_tok +
#         (idx_feats * stride_conv_state_dim)[None, :] +
#         ((idx_tokens + seqlen) * stride_conv_state_tok)[:, None])  # [BLOCK_M, BLOCK_N]
#         #((idx_tokens + (1 if IS_SPEC_DECODING else seqlen)) *
         
#     mask = ((conv_state_batch_coord < num_cache_lines)
#             & ((idx_tokens + seqlen) < state_len)[:, None]
#             & (idx_feats < dim)[None, :])
#     conv_state = tl.load(conv_state_ptrs_source, mask, other=0.0)

#     VAL = state_len - seqlen
#     x_base = x_ptr + x_offset + (idx_feats * stride_x_dim)  # [BLOCK_N]

#     x_ptrs = x_base[None, :] + (
#         (idx_tokens - VAL) * stride_x_token)[:, None]  # [BLOCK_M, BLOCK_N]

#     mask_x = ((idx_tokens - VAL >= 0)[:, None] &
#               (idx_tokens - VAL < seqlen)[:, None] & (idx_feats < dim)[None, :]
#               )  # token-index  # token-index  # feature-index
#     loaded_x = tl.load(x_ptrs, mask_x, 0.0)
#     tl.debug_barrier()

#     new_conv_state = tl.where(mask, conv_state, loaded_x)

#     conv_state_base = (conv_state_ptr +
#                        (conv_state_batch_coord * stride_conv_state_seq) +
#                        (idx_feats * stride_conv_state_dim))  # [BLOCK_N,]
#     conv_state_ptrs_target = conv_state_base + (
#         idx_tokens * stride_conv_state_tok)[:, None]  # [BLOCK_M, BLOCK_N]
#     mask = (idx_tokens < state_len)[:, None] & (idx_feats < dim)[None, :]
#     tl.store(conv_state_ptrs_target, new_conv_state, mask)

#     # STEP 3: init accumulator
#     if HAS_BIAS:
#         bias = bias_ptr + idx_feats
#         mask_bias = idx_feats < dim
#         acc_preload = tl.load(bias, mask=mask_bias,
#                               other=0.0).to(tl.float32)  # [BLOCK_N]
#     else:
#         acc_preload = tl.zeros((BLOCK_N, ), dtype=tl.float32)

#     # STEP 4:
#     # PRE-LOAD WEIGHTS
#     # first kernel column, configured for weights to handle BLOCK_N features in range
#     w_base = w_ptr + (idx_feats * stride_w_dim)  # [BLOCK_N,]
#     mask_w = idx_feats < dim
#     if KERNEL_WIDTH >= 2:
#         w_ptrs = w_base + (0 * stride_w_width)  # [BLOCK_N] tensor
#         w_col0 = tl.load(w_ptrs, mask_w, other=0.0)
#         w_ptrs = w_base + (1 * stride_w_width)  # [BLOCK_N] tensor
#         w_col1 = tl.load(w_ptrs, mask_w, other=0.0)
    

#     acc = acc_preload
#     acc += w_col0*col0 

#     #col1 = tl.load(x_base_1d + 0 * stride_x_token, mask=mask_x_1d)
#     acc += w_col1*col1 

#     o_ptrs = o_ptr + o_offset + 0 * stride_o_token + (idx_feats *
#                                                               stride_o_dim)
#     tl.store(o_ptrs, acc, (0 < seqlen+1) & (idx_feats < dim))

#     acc = acc_preload

#     acc += w_col0*col1 

#     x_base_1d = x_base  # starting of chunk [BLOCK_N]
#     mask_x_1d = idx_feats < dim
#     col2 = tl.load(x_base_1d + 0 * stride_x_token, mask=mask_x_1d)
#     acc += w_col1*col2 

#     o_ptrs = o_ptr + o_offset + 1 * stride_o_token + (idx_feats *
#                                                               stride_o_dim)
#     tl.store(o_ptrs, acc, (1 < seqlen+1) & (idx_feats < dim))
# # Autotune configurations - experiment with different block sizes
# configs = [
#     Config({"BLOCK_BT":1}, num_warps=8),
#     Config({"BLOCK_BT":2}, num_warps=8),
#     Config({"BLOCK_BT":4}, num_warps=8),
#     Config({"BLOCK_BT":8}, num_warps=8),
#     # Config({"BLOCK_BT":16}, num_warps=8),
#     # Config({"BLOCK_BT":32}),
#     # Config({"BLOCK_BT":64}),
#     # Config({"BLOCK_BT":128}),

  
# ]

# @triton.jit
# def _corrected_grouped_conv1d_w2_decode_kernel(
#     x, y, weight, bias,
#     G: tl.constexpr, D: tl.constexpr, W: tl.constexpr,
#     HAS_BIAS: tl.constexpr,
# ):

#     pid_b = tl.program_id(0)   # Batch
#     pid_g = tl.program_id(1)   # Group

#     c = tl.arange(0, D)      # Full D=128
#     mask_c = c < D
#     o_w = tl.arange(0, W)    # W=2
    
#     stride_ng = G * D
#     base_offset = pid_b * stride_ng * W + pid_g * D * W
    
#     # --- Load Inputs ---
#     # x_mask = (c[:,None] < D) & (o_w[None,:] < W)
#     # x_full = tl.load(x + base_offset + c[:,None]*W + o_w[None,:], mask=x_mask, other=0.0).to(tl.float32)  # [128, 2]
    
#     # b_x_hist, b_x_curr = tl.split(x_full) # Each [128]
#     x_hist = tl.load(x + base_offset + c * 2 + 0, mask=mask_c, other=0.0).to(tl.float32)  # [D]
#     x_curr = tl.load(x + base_offset + c * 2 + 1, mask=mask_c, other=0.0).to(tl.float32)  # [D]

#     ci = tl.arange(0, D)  # [D]
#     mask_ci = ci < D

#     base_w = pid_g * (D * D * 2)

#     w_hist_ptrs = base_w + c[:, None] * (D * 2) + ci[None, :] * 2 + 0  # [D, D]
#     w_curr_ptrs = base_w + c[:, None] * (D * 2) + ci[None, :] * 2 + 1  # [D, D]
    
#     w_hist = tl.load(weight + w_hist_ptrs, mask=mask_c[:, None] & mask_ci[None, :], other=0.0).to(tl.float32)
#     w_curr = tl.load(weight + w_curr_ptrs, mask=mask_c[:, None] & mask_ci[None, :], other=0.0).to(tl.float32)

#     acc = tl.sum(w_hist * x_hist[None, :], axis=1) + tl.sum(w_curr * x_curr[None, :], axis=1)  # [D]

#     if HAS_BIAS:
#         acc += tl.load(bias + pid_g * D + c, mask=None).to(tl.float32)
    

#     tl.store(y + pid_b * stride_ng + pid_g * D + c, 
#              acc, 
#              mask=mask_c)
# # -------------------------------------------------------------------------
# # 3. Test Execution
# # -------------------------------------------------------------------------
# if __name__ == "__main__":
#     # Ensure this is running on GPU
#     device = "cuda"
    
#     # Hyperparameters
#     BATCH = 4
#     DIM = 2048        # Model dimension (e.g., 4096 in real LLMs)
#     SEQLEN = 1       # Processing 2 tokens (as per the kernel logic)
#     STATE_LEN = 2    # Buffer size (usually strictly KERNEL_WIDTH)
#     KERNEL_WIDTH = 2
#     G = 16
    
#     # Initialize Tensors
#     # Shape: (Batch, Dim, Seq)
#     x = torch.randn((BATCH, DIM, SEQLEN), device=device, dtype=torch.float32)
    
#     # State: (Batch, Dim, State_Len) - stores history
#     # We initialize it with some data to verify the "Shift" works
#     conv_state = torch.randn((BATCH, DIM, STATE_LEN), device=device, dtype=torch.float32)
#     conv_state_copy = conv_state.clone() # Keep a copy to verify update
    
#     # Weights: (Dim, Kernel_Width) - simplified 1D Depthwise Conv
#     weight = torch.randn((DIM, KERNEL_WIDTH), device=device, dtype=torch.float32)

#     weight2 = torch.randn((G,DIM//G,DIM//G, KERNEL_WIDTH), device=device, dtype=torch.float32)
#     bias2 = torch.randn((G,DIM//G), device=device, dtype=torch.float32)




    

    
    
#     # Bias: (Dim)
#     bias = torch.randn((DIM,), device=device, dtype=torch.float32)

#     print(f"Input shape: {x.shape}")
#     print(f"State shape: {conv_state.shape}")

#     # Run the Kernel
#     # NOTE: You must include the @triton.jit decorated function code 
#     # in the script before running this.
#     try:

#         output = run_causal_conv1d_update(x, conv_state, weight, bias, seqlen=SEQLEN)

#         w0 = weight[:,0]
#         w1 = weight[:,1]

#         print(f"output1 at position 0 matches: {torch.allclose(conv_state_copy[...,0]*w0+conv_state_copy[...,1]*w1 + bias,output[...,0], atol=1e-6)}")
#         print(f"output1 at position 1 matches: {torch.allclose(conv_state_copy[...,1]*w0+x[...,0]*w1 + bias,output[...,1], atol=1e-6)}")

        
        
#         print("\n--- Kernel Executed Successfully ---")
#         print(f"Output shape: {output.shape}")
        
#         # Verification Logic (Simplified)
#         # 1. Check if state updated (shifted left + appended new X)
#         # The last `SEQLEN` elements of the new state should match `x`
#         state_tail = conv_state[:, :, -SEQLEN:]
#         diff = torch.abs(state_tail - x).max()
#         print(f"State Update Correctness (Max Diff): {diff.item()}")
        
#     except Exception as e:
#         print(f"Execution failed. Did you paste the kernel code? Error: {e}")
    
#     o = output.reshape(BATCH, DIM//G, G, -1).transpose(1,2).contiguous()

#     #out_m = torch.einsum('goiw,bgiw->bgo', weight2, o)+bias2
#     out2 = grouped_conv1d_decode(x=o, weight=weight2, bias=bias2)
#     # Reshape to standard conv1d layout
#     x_conv = o.reshape(BATCH, DIM, 2)                    # [B, C_in, L]
#     weight_conv = weight2.reshape(DIM, DIM//G, 2)          # [C_out, C_in/groups, K]
    
#     # Convolve: L_out = L - K + 1 = 1
#     y = F.conv1d(x_conv, weight_conv, groups=G, bias=bias2.reshape(DIM,))    # [B, C_out, 1]
#     out_m = y.squeeze(-1).reshape(BATCH, G, DIM//G)             # [B, G, D]

#     print(f"{nn.MSELoss()(out2,out_m)/torch.norm(out_m)=}") # still doesn't work...
    
#     print("-"*30)
#     print("TESTING CACHE KERNEL")
#     print("-"*30)

#     device = 'cuda'
#     C, width = 2048, 2  
#     seq_lengths = torch.tensor([
#         512, 1024, 768, 128, 1024, 2048, 512, 1024, 768, 128,
#         1024, 1024, 2048, 512, 1024, 768, 512, 1024, 768, 128
#     ], device=device)

#     T_total = seq_lengths.sum().item()
#     N = len(seq_lengths)  # 20 sequences

#     # Create cu_seqlens: [0, 512, 1536, 2304, ..., 16768]
#     cu_seqlens = torch.cat([
#         torch.zeros(1, dtype=torch.int32, device=device),
#         seq_lengths.cumsum(0, dtype=torch.int32)
#     ])

#     # CORRECTED: Create data in (T_total, D) format, not (1, D, T_total)
#     x = (1 / C ** 0.5) * torch.randn(1,
#         T_total, C,# Flattened: all sequences concatenated
#         device=device,
#         dtype=torch.bfloat16
#     )

#     # # No initial state (first iteration)
#     # initial_state = None

#     # Call the function
#     final_state = cache_states(
#         x=x,
#         state_len=width,
#         cu_seqlens=cu_seqlens
#     )

#     print(f"Input shape: {x.shape}")        # torch.Size([16768, 2048])
#     outs= []
#     for s in cu_seqlens[1:]:
#         outs.append(x[:,s-2:s,...])
#     loop_outs = torch.cat(outs, dim=0)
#     print(f"Output shape: {final_state.shape}")  # torch.Size(
#     print(f"Loop output shape: {loop_outs.shape}")
#     print(f"Loop extracted states = kernel extracted states: {(loop_outs.transpose(1,2)==final_state).all().item()}")