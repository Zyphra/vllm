/*
 * CCA Decode Fused Kernel — "Ultimate" fusion for AMD MI-series GPUs.
 *
 * Fuses in a SINGLE kernel launch per (batch, group):
 *
 *   Phase 1 — Depthwise causal conv1d update  (K=2):
 *       For each channel d in group g:
 *         y0 = dw_bias + dw_w0*state0 + dw_w1*state1
 *         y1 = dw_bias + dw_w0*state1 + dw_w1*new_token
 *         Update conv state: state0←state1, state1←new_token
 *       Produces [D, 2] intermediate in LDS.
 *
 *   Phase 2 — Grouped conv1d  (W=2):
 *       acc[d] = GEMV(gw[g,d,:,:], phase1_out[:,:]) + gw_bias[g,d]
 *
 *   Phase 3 — Add qk_mean, L2-normalize, scale, temperature:
 *       acc += qk_mean[batch, group, d]
 *       norm  = sqrt(sum(acc^2))         (block-level reduction)
 *       acc  *= sqrt_head_dim / norm
 *       if key_head: acc *= temp[kh_idx]
 *
 *   Phase 4 — Write directly to output q/k buffer.
 *
 * Tensor shapes (contiguous):
 *   new_token:   [B, E, 1]         E = G * D  (total channels)
 *   dw_weight:   [E, K=2]          depthwise conv weight
 *   dw_bias:     [E]               depthwise conv bias  (optional)
 *   conv_state:  [num_cache, E, state_len]    KV-cache conv states
 *   state_idx:   [B]               int64 index into conv_state
 *   gw_weight:   [G, D, D, 2]      grouped conv weight
 *   gw_bias:     [G, D]            grouped conv bias    (optional)
 *   qk_mean:     [B, G, D]         packed q/k mean residual
 *   temp:        [num_k_heads]      per-key-head temperature
 *   output:      [B, q_dim + k_dim] pre-allocated output buffer
 */

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#ifdef USE_ROCM
  #include <hip/hip_runtime.h>
#else
  #include <cuda_runtime.h>
#endif

#ifndef USE_ROCM
  #define VLLM_SHFL_XOR_SYNC(var, mask) __shfl_xor_sync(0xFFFFFFFF, var, mask)
#else
  #define VLLM_SHFL_XOR_SYNC(var, mask) __shfl_xor(var, mask)
#endif

namespace {

// Block-level sum reduction via LDS.
// BLOCK_D threads each contribute one float; returns the sum on ALL threads.
template <int BLOCK_D>
__device__ __forceinline__ float block_reduce_sum(float val, float* smem) {
    const int tid = threadIdx.x;
    smem[tid] = val;
    __syncthreads();

    // Tree reduction
    #pragma unroll
    for (int s = BLOCK_D / 2; s > 0; s >>= 1) {
        if (tid < s)
            smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float result = smem[0];
    __syncthreads();   // so smem can be reused
    return result;
}

// =======================================================================
// The fused kernel
//
// Grid  (B, G)      Block (BLOCK_D)
// LDS   D*2 floats  (phase-1 output)  +  BLOCK_D floats (reduction scratch)
// =======================================================================
template <typename scalar_t, int BLOCK_D,
          bool HAS_DW_BIAS, bool HAS_GW_BIAS, bool CLAMP_TEMP>
__global__ void cca_decode_fused_kernel(
    // Phase 1 inputs
    const scalar_t* __restrict__ new_token,   // [B, E, 1]
    const scalar_t* __restrict__ dw_weight,   // [E, 2]
    const scalar_t* __restrict__ dw_bias,     // [E]
    scalar_t*       __restrict__ conv_state,  // [num_cache, E, state_len]
    const int64_t*  __restrict__ state_idx,   // [B]
    // Phase 2 inputs
    const scalar_t* __restrict__ gw_weight,   // [G, D, D, 2]
    const scalar_t* __restrict__ gw_bias,     // [G, D]
    // Phase 3 inputs
    const scalar_t* __restrict__ qk_mean,     // [B, G, D]
    const float*    __restrict__ temp,        // [num_k_heads]
    // Phase 4 output
    scalar_t*       __restrict__ output,      // [B, out_dim]  out_dim = (qh+kh)*D
    // Dimensions
    const int G, const int D,
    const int E,                               // = G * D
    const int state_len,                       // conv_state last-dim size
    const int num_q_heads,
    const float sqrt_head_dim,
    const int num_cache_lines)
{
    const int i_n   = blockIdx.x;   // batch
    const int i_g   = blockIdx.y;   // group  (0..qh-1 = Q, qh..G-1 = K)
    const int d_out = threadIdx.x;  // output channel within group

    const int DW = D * 2;
    const int GD = G * D;

    // LDS layout: [D*2] phase1 output  |  [BLOCK_D] reduction scratch
    extern __shared__ char raw_smem[];
    float* s_phase1 = reinterpret_cast<float*>(raw_smem);
    float* s_reduce = s_phase1 + DW;

    // ================================================================
    // Phase 1: Depthwise causal conv1d update (per-channel, K=2)
    // ================================================================
    const int64_t cache_row = state_idx[i_n];
    const bool valid_row = (cache_row >= 0) & (cache_row < num_cache_lines);

    if (d_out < D) {
        const int ch = i_g * D + d_out;   // flat channel index in [0, E)

        if (valid_row) {
            // conv_state layout: [num_cache, E, state_len] (contiguous)
            const int64_t cs_base = cache_row * E * state_len + ch * state_len;
            float s0 = static_cast<float>(conv_state[cs_base + 0]);
            float s1 = static_cast<float>(conv_state[cs_base + 1]);

            float xt = static_cast<float>(new_token[i_n * E + ch]);

            float dw0 = static_cast<float>(dw_weight[ch * 2 + 0]);
            float dw1 = static_cast<float>(dw_weight[ch * 2 + 1]);

            float db = 0.0f;
            if constexpr (HAS_DW_BIAS)
                db = static_cast<float>(dw_bias[ch]);

            float y0 = fmaf(dw0, s0, fmaf(dw1, s1, db));
            float y1 = fmaf(dw0, s1, fmaf(dw1, xt, db));

            s_phase1[d_out * 2 + 0] = y0;
            s_phase1[d_out * 2 + 1] = y1;

            // Update conv state in-place
            conv_state[cs_base + 0] = static_cast<scalar_t>(s1);
            conv_state[cs_base + 1] = static_cast<scalar_t>(xt);
        } else {
            // Invalid/padded row — zero the LDS so subsequent phases
            // produce deterministic (zero) output.
            s_phase1[d_out * 2 + 0] = 0.0f;
            s_phase1[d_out * 2 + 1] = 0.0f;
        }
    }
    __syncthreads();

    if (d_out >= D) return;

    // ================================================================
    // Phase 2: Grouped conv GEMV
    //   acc = Σ_k Σ_w  gw_weight[g, d_out, k, w] * s_phase1[k, w]
    // ================================================================
    const scalar_t* gw_row = gw_weight
        + static_cast<int64_t>(i_g) * D * DW + d_out * DW;

    float acc = 0.0f;

    if constexpr (sizeof(scalar_t) <= 2) {
        constexpr int PACK = 8;
        const int n_full   = DW / PACK;
        const int4* w_vec  = reinterpret_cast<const int4*>(gw_row);

        for (int p = 0; p < n_full; ++p) {
            int4 pk = w_vec[p];
            const scalar_t* e = reinterpret_cast<const scalar_t*>(&pk);
            const int base = p * PACK;
            #pragma unroll
            for (int j = 0; j < PACK; ++j)
                acc = fmaf(static_cast<float>(e[j]), s_phase1[base + j], acc);
        }
        for (int k = n_full * PACK; k < DW; ++k)
            acc = fmaf(static_cast<float>(gw_row[k]), s_phase1[k], acc);
    } else {
        constexpr int PACK = 4;
        const int n_full      = DW / PACK;
        const float4* w_vec   = reinterpret_cast<const float4*>(gw_row);

        for (int p = 0; p < n_full; ++p) {
            float4 v = w_vec[p];
            const int base = p * PACK;
            acc = fmaf(v.x, s_phase1[base + 0], acc);
            acc = fmaf(v.y, s_phase1[base + 1], acc);
            acc = fmaf(v.z, s_phase1[base + 2], acc);
            acc = fmaf(v.w, s_phase1[base + 3], acc);
        }
        for (int k = n_full * PACK; k < DW; ++k)
            acc = fmaf(gw_row[k], s_phase1[k], acc);
    }

    if constexpr (HAS_GW_BIAS)
        acc += static_cast<float>(gw_bias[i_g * D + d_out]);

    // ================================================================
    // Phase 3: Add qk_mean + L2 norm + scale + temperature
    // ================================================================
    acc += static_cast<float>(qk_mean[i_n * GD + i_g * D + d_out]);

    // L2 norm via block-level reduction
    float norm_sq = block_reduce_sum<BLOCK_D>(acc * acc, s_reduce);
    float inv_norm = rsqrtf(norm_sq + 1e-12f);

    acc *= sqrt_head_dim * inv_norm;

    // Temperature for key heads
    if (i_g >= num_q_heads) {
        int k_idx = i_g - num_q_heads;
        float t = temp[k_idx];
        if constexpr (CLAMP_TEMP)
            t = expf(fminf(fmaxf(t, 1e-7f), 2.0f));
        acc *= t;
    }

    // ================================================================
    // Phase 4: Write directly to output[batch, ...]
    //   output layout: [B, (qh+kh)*D]
    //   group i_g contributes D elements at offset i_g*D
    // ================================================================
    output[i_n * GD + i_g * D + d_out] = static_cast<scalar_t>(acc);
}

// =======================================================================
// Typed launcher
// =======================================================================
template <typename scalar_t, int BLOCK_D,
          bool HAS_DW_BIAS, bool HAS_GW_BIAS, bool CLAMP_TEMP>
void launch_fused(
    const scalar_t* new_token, const scalar_t* dw_weight,
    const scalar_t* dw_bias, scalar_t* conv_state,
    const int64_t* state_idx,
    const scalar_t* gw_weight, const scalar_t* gw_bias,
    const scalar_t* qk_mean, const float* temp,
    scalar_t* output,
    int B, int G, int D, int E, int state_len,
    int num_q_heads, float sqrt_head_dim,
    int num_cache_lines, cudaStream_t stream)
{
    dim3 grid(B, G);
    int smem = (D * 2 + BLOCK_D) * static_cast<int>(sizeof(float));
    cca_decode_fused_kernel<scalar_t, BLOCK_D,
                            HAS_DW_BIAS, HAS_GW_BIAS, CLAMP_TEMP>
        <<<grid, BLOCK_D, smem, stream>>>(
            new_token, dw_weight, dw_bias, conv_state, state_idx,
            gw_weight, gw_bias, qk_mean, temp, output,
            G, D, E, state_len, num_q_heads, sqrt_head_dim, num_cache_lines);
}

}  // anonymous namespace

// =======================================================================
// Public entry point
// =======================================================================
torch::Tensor cca_decode_fused(
    const torch::Tensor& new_token,       // [B, E, 1]
    const torch::Tensor& dw_weight,       // [E, 2]
    const std::optional<torch::Tensor>& dw_bias_opt, // [E]
    torch::Tensor& conv_state,            // [num_cache, E, state_len] — mutated
    const torch::Tensor& state_indices,   // [B] int64
    const torch::Tensor& gw_weight,       // [G, D, D, 2]
    const std::optional<torch::Tensor>& gw_bias_opt, // [G, D]
    const torch::Tensor& qk_mean,         // [B, G, D]
    const torch::Tensor& temp,            // [num_k_heads] float
    int64_t num_q_heads,
    double sqrt_head_dim,
    bool clamp_temp)
{
    TORCH_CHECK(new_token.is_cuda(), "new_token must be on CUDA/HIP");
    TORCH_CHECK(new_token.dim() == 3 && new_token.size(2) == 1,
                "new_token must be [B, E, 1]");

    const int B = new_token.size(0);
    const int E = new_token.size(1);  // total channels = G * D
    const int G = gw_weight.size(0);
    const int D = gw_weight.size(1);
    TORCH_CHECK(G * D == E, "G*D must equal E");
    TORCH_CHECK(gw_weight.size(2) == D && gw_weight.size(3) == 2,
                "gw_weight must be [G,D,D,2]");

    const bool has_dw_bias = dw_bias_opt.has_value() && dw_bias_opt->defined();
    const bool has_gw_bias = gw_bias_opt.has_value() && gw_bias_opt->defined();
    const int num_cache_lines = conv_state.size(0);
    const int state_len = conv_state.size(2);

    auto nt_c = new_token.contiguous();
    auto dww  = dw_weight.contiguous();
    auto gww  = gw_weight.contiguous();
    auto qkm  = qk_mean.contiguous();
    auto tmp   = temp.to(torch::kFloat32).contiguous();
    auto sidx  = state_indices.contiguous();

    torch::Tensor dwb, gwb;
    if (has_dw_bias) dwb = dw_bias_opt->contiguous();
    if (has_gw_bias) gwb = gw_bias_opt->contiguous();

    // Output: [B, G*D]  (q heads then k heads, each D-wide)
    auto output = torch::empty({B, G * D}, nt_c.options());

    const at::cuda::OptionalCUDAGuard guard(new_token.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Dispatch macro
#define _DISPATCH(...)                                                       \
    AT_DISPATCH_SWITCH(nt_c.scalar_type(), "cca_decode_fused",               \
        AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] { __VA_ARGS__; })    \
        AT_DISPATCH_CASE(at::ScalarType::Half,     [&] { __VA_ARGS__; })    \
        AT_DISPATCH_CASE(at::ScalarType::Float,    [&] { __VA_ARGS__; })    \
    )

#define _PTR(t) (t).data_ptr<scalar_t>()
#define _OPTR(t, cond) ((cond) ? (t).data_ptr<scalar_t>() : static_cast<scalar_t*>(nullptr))

    auto run = [&](auto dw_tag, auto gw_tag, auto ct_tag) {
        constexpr bool DWB = decltype(dw_tag)::value;
        constexpr bool GWB = decltype(gw_tag)::value;
        constexpr bool CT  = decltype(ct_tag)::value;

        if (D <= 64) {
            _DISPATCH(launch_fused<scalar_t, 64, DWB, GWB, CT>(
                _PTR(nt_c), _PTR(dww), _OPTR(dwb, has_dw_bias),
                _PTR(conv_state), sidx.data_ptr<int64_t>(),
                _PTR(gww), _OPTR(gwb, has_gw_bias),
                _PTR(qkm), tmp.data_ptr<float>(),
                _PTR(output),
                B, G, D, E, state_len,
                static_cast<int>(num_q_heads),
                static_cast<float>(sqrt_head_dim),
                num_cache_lines, stream));
        } else if (D <= 128) {
            _DISPATCH(launch_fused<scalar_t, 128, DWB, GWB, CT>(
                _PTR(nt_c), _PTR(dww), _OPTR(dwb, has_dw_bias),
                _PTR(conv_state), sidx.data_ptr<int64_t>(),
                _PTR(gww), _OPTR(gwb, has_gw_bias),
                _PTR(qkm), tmp.data_ptr<float>(),
                _PTR(output),
                B, G, D, E, state_len,
                static_cast<int>(num_q_heads),
                static_cast<float>(sqrt_head_dim),
                num_cache_lines, stream));
        } else {
            _DISPATCH(launch_fused<scalar_t, 256, DWB, GWB, CT>(
                _PTR(nt_c), _PTR(dww), _OPTR(dwb, has_dw_bias),
                _PTR(conv_state), sidx.data_ptr<int64_t>(),
                _PTR(gww), _OPTR(gwb, has_gw_bias),
                _PTR(qkm), tmp.data_ptr<float>(),
                _PTR(output),
                B, G, D, E, state_len,
                static_cast<int>(num_q_heads),
                static_cast<float>(sqrt_head_dim),
                num_cache_lines, stream));
        }
    };

    // 8 specialisations: (has_dw_bias, has_gw_bias, clamp_temp)
    auto pick = [&](auto dw_tag, auto gw_tag) {
        if (clamp_temp) run(dw_tag, gw_tag, std::true_type{});
        else            run(dw_tag, gw_tag, std::false_type{});
    };
    auto pick2 = [&](auto dw_tag) {
        if (has_gw_bias) pick(dw_tag, std::true_type{});
        else             pick(dw_tag, std::false_type{});
    };
    if (has_dw_bias) pick2(std::true_type{});
    else             pick2(std::false_type{});

#undef _DISPATCH
#undef _PTR
#undef _OPTR

    return output;
}
