/*
 * CCA Decode Fused Kernel v2 — optimised for MI300X decode serving.
 *
 * Runtime shapes:
 *   B=128  E=1280  G=10  D=128  state_len=2  qh=8  kh=2
 *   → grid(B,G)  block(128)
 *
 * Fuses in a SINGLE kernel launch per (batch, group):
 *
 *   Phase 1 — Depthwise causal conv1d update  (K=2)
 *   Phase 2 — Grouped conv1d  (W=2)   coalesced GEMV
 *   Phase 3 — qk_mean + L2-norm + scale + temperature
 *   Phase 4 — Write to output q/k buffer
 *
 * v2 key change: gw_weight is now TRANSPOSED to [G, D*2, D] so that
 * consecutive threads (d_out) read consecutive bf16 elements → fully
 * coalesced global/L2 reads.  Previous layout [G, D, D, 2] caused each
 * warp to issue 64 separate cache-line requests per iteration (threads
 * 512 bytes apart).  With the transposed layout this drops to 2 cache
 * lines per warp per iteration — a ~32x reduction in L1 pressure.
 *
 * Other v2 changes:
 *   - __launch_bounds__(BLOCK_D, 8) for higher occupancy / latency hiding
 *   - Host-side: skip redundant contiguous() and temp conversion
 */

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#ifdef USE_ROCM
  #include <hip/hip_runtime.h>
  #define VLLM_SHFL_XOR_SYNC(var, mask) __shfl_xor(var, mask)
  #define WARP_SZ 64
#else
  #include <cuda_runtime.h>
  #define VLLM_SHFL_XOR_SYNC(var, mask) __shfl_xor_sync(0xFFFFFFFF, var, mask)
  #define WARP_SZ 32
#endif

namespace {

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SZ / 2; offset > 0; offset >>= 1)
        val += VLLM_SHFL_XOR_SYNC(val, offset);
    return val;
}

template <int BLOCK_D>
__device__ __forceinline__ float block_reduce_sum(float val, float* smem) {
    constexpr int NUM_WARPS = (BLOCK_D + WARP_SZ - 1) / WARP_SZ;

    val = warp_reduce_sum(val);

    if constexpr (NUM_WARPS == 1)
        return val;

    const int warp_id = threadIdx.x / WARP_SZ;
    const int lane    = threadIdx.x & (WARP_SZ - 1);

    if (lane == 0)
        smem[warp_id] = val;
    __syncthreads();

    float total = 0.0f;
    #pragma unroll
    for (int w = 0; w < NUM_WARPS; ++w)
        total += smem[w];
    __syncthreads();
    return total;
}

// =======================================================================
// Fused kernel v2
//
// Grid  (B, G)      Block (BLOCK_D)
// LDS   D*2 floats (phase-1 output) + NUM_WARPS floats (reduction scratch)
//
// gw_weight MUST be in transposed layout [G, D*2, D]  (not [G, D, D, 2])
// =======================================================================
template <typename scalar_t, int BLOCK_D,
          bool HAS_DW_BIAS, bool HAS_GW_BIAS, bool CLAMP_TEMP>
__global__ void
__launch_bounds__(BLOCK_D, 8)
cca_decode_fused_kernel(
    const scalar_t* __restrict__ new_token,   // [B, E, 1]
    const scalar_t* __restrict__ dw_weight,   // [E, 2]
    const scalar_t* __restrict__ dw_bias,     // [E]
    scalar_t*       __restrict__ conv_state,  // [num_cache, E, state_len]
    const int64_t*  __restrict__ state_idx,   // [B]
    const scalar_t* __restrict__ gw_weight_T, // [G, D*2, D]  ← TRANSPOSED
    const scalar_t* __restrict__ gw_bias,     // [G, D]
    const scalar_t* __restrict__ qk_mean,     // [B, G, D]
    const float*    __restrict__ temp,        // [num_k_heads]
    scalar_t*       __restrict__ output,      // [B, G*D]
    const int G, const int D,
    const int E,
    const int state_len,
    const int num_q_heads,
    const float sqrt_head_dim,
    const int num_cache_lines,
    const int64_t pad_slot_id)
{
    const int i_n   = blockIdx.x;
    const int i_g   = blockIdx.y;
    const int d_out = threadIdx.x;

    const int DW = D * 2;
    const int GD = G * D;

    extern __shared__ char raw_smem[];
    float* s_phase1 = reinterpret_cast<float*>(raw_smem);
    constexpr int NUM_WARPS = (BLOCK_D + WARP_SZ - 1) / WARP_SZ;
    float* s_reduce = s_phase1 + DW;

    // ================================================================
    // Phase 1: Depthwise causal conv1d update (per-channel, K=2)
    // ================================================================
    const int64_t cache_row = state_idx[i_n];
    const bool valid_row = (cache_row >= 0)
                         & (cache_row < num_cache_lines)
                         & (cache_row != pad_slot_id);

    if (d_out < D) {
        const int ch = i_g * D + d_out;

        if (valid_row) {
            const int64_t cs_base = cache_row * E * state_len + ch * state_len;
            float s0 = static_cast<float>(conv_state[cs_base + 0]);
            float s1 = static_cast<float>(conv_state[cs_base + 1]);

            float xt = static_cast<float>(new_token[i_n * E + ch]);

            float dw0 = static_cast<float>(dw_weight[ch * 2 + 0]);
            float dw1 = static_cast<float>(dw_weight[ch * 2 + 1]);

            float db = 0.0f;
            if constexpr (HAS_DW_BIAS)
                db = static_cast<float>(dw_bias[ch]);

            s_phase1[d_out * 2 + 0] = fmaf(dw0, s0, fmaf(dw1, s1, db));
            s_phase1[d_out * 2 + 1] = fmaf(dw0, s1, fmaf(dw1, xt, db));

            conv_state[cs_base + 0] = static_cast<scalar_t>(s1);
            conv_state[cs_base + 1] = static_cast<scalar_t>(xt);
        } else {
            s_phase1[d_out * 2 + 0] = 0.0f;
            s_phase1[d_out * 2 + 1] = 0.0f;
        }
    }
    __syncthreads();

    if (d_out >= D) return;

    // ================================================================
    // Phase 2: Grouped conv GEMV — COALESCED column-oriented access
    //
    //   weight_T layout: [G, DW, D]
    //   weight_T[g, k, d_out] at offset  g*DW*D + k*D + d_out
    //
    //   Consecutive threads (d_out, d_out+1) read addresses that differ
    //   by 1 element (2 bytes) → perfectly coalesced.  A full warp of
    //   64 threads touches only 2 cache lines (128 bytes) per iteration.
    // ================================================================
    const scalar_t* gw_base = gw_weight_T
        + static_cast<int64_t>(i_g) * DW * D;

    float acc = 0.0f;

    #pragma unroll 8
    for (int k = 0; k < DW; ++k) {
        float w = static_cast<float>(gw_base[k * D + d_out]);
        acc = fmaf(w, s_phase1[k], acc);
    }

    if constexpr (HAS_GW_BIAS)
        acc += static_cast<float>(gw_bias[i_g * D + d_out]);

    // ================================================================
    // Phase 3: Add qk_mean + L2 norm + scale + temperature
    // ================================================================
    acc += static_cast<float>(qk_mean[i_n * GD + i_g * D + d_out]);

    float norm_sq = block_reduce_sum<BLOCK_D>(acc * acc, s_reduce);
    float inv_norm = rsqrtf(norm_sq + 1e-12f);

    acc *= sqrt_head_dim * inv_norm;

    if (i_g >= num_q_heads) {
        int k_idx = i_g - num_q_heads;
        float t = temp[k_idx];
        if constexpr (CLAMP_TEMP)
            t = expf(fminf(fmaxf(t, 1e-7f), 2.0f));
        acc *= t;
    }

    // ================================================================
    // Phase 4: Write output
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
    const scalar_t* gw_weight_T, const scalar_t* gw_bias,
    const scalar_t* qk_mean, const float* temp,
    scalar_t* output,
    int B, int G, int D, int E, int state_len,
    int num_q_heads, float sqrt_head_dim,
    int num_cache_lines, int64_t pad_slot_id,
    cudaStream_t stream)
{
    dim3 grid(B, G);
    constexpr int NUM_WARPS = (BLOCK_D + WARP_SZ - 1) / WARP_SZ;
    int smem = (D * 2 + NUM_WARPS) * static_cast<int>(sizeof(float));
    cca_decode_fused_kernel<scalar_t, BLOCK_D,
                            HAS_DW_BIAS, HAS_GW_BIAS, CLAMP_TEMP>
        <<<grid, BLOCK_D, smem, stream>>>(
            new_token, dw_weight, dw_bias, conv_state, state_idx,
            gw_weight_T, gw_bias, qk_mean, temp, output,
            G, D, E, state_len, num_q_heads, sqrt_head_dim,
            num_cache_lines, pad_slot_id);
}

}  // anonymous namespace

// =======================================================================
// Public entry point
//
// gw_weight must be PRE-TRANSPOSED to [G, D*2, D] by the caller.
// =======================================================================
torch::Tensor cca_decode_fused(
    const torch::Tensor& new_token,       // [B, E, 1]
    const torch::Tensor& dw_weight,       // [E, 2]
    const std::optional<torch::Tensor>& dw_bias_opt, // [E]
    torch::Tensor& conv_state,            // [num_cache, E, state_len]
    const torch::Tensor& state_indices,   // [B] int64
    const torch::Tensor& gw_weight,       // [G, D*2, D]  ← transposed
    const std::optional<torch::Tensor>& gw_bias_opt, // [G, D]
    const torch::Tensor& qk_mean,         // [B, G, D]
    const torch::Tensor& temp,            // [num_k_heads] float
    int64_t num_q_heads,
    double sqrt_head_dim,
    bool clamp_temp,
    int64_t pad_slot_id)
{
    TORCH_CHECK(new_token.is_cuda(), "new_token must be on CUDA/HIP");
    TORCH_CHECK(new_token.dim() == 3 && new_token.size(2) == 1,
                "new_token must be [B, E, 1]");

    const int B = new_token.size(0);
    const int E = new_token.size(1);
    const int G = gw_weight.size(0);
    const int D = gw_weight.size(2);    // transposed: [G, DW, D]
    const int DW = gw_weight.size(1);
    TORCH_CHECK(G * D == E, "G*D must equal E");
    TORCH_CHECK(DW == D * 2,
                "gw_weight must be transposed to [G, D*2, D]");

    const bool has_dw_bias = dw_bias_opt.has_value() && dw_bias_opt->defined();
    const bool has_gw_bias = gw_bias_opt.has_value() && gw_bias_opt->defined();
    const int num_cache_lines = conv_state.size(0);
    const int state_len = conv_state.size(2);

    auto output = torch::empty({B, G * D}, new_token.options());

    const at::cuda::OptionalCUDAGuard guard(new_token.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor dwb, gwb;
    if (has_dw_bias) dwb = dw_bias_opt->contiguous();
    if (has_gw_bias) gwb = gw_bias_opt->contiguous();

    auto tmp = (temp.scalar_type() == at::kFloat)
        ? temp : temp.to(torch::kFloat32);

#define _DISPATCH(...)                                                       \
    AT_DISPATCH_SWITCH(new_token.scalar_type(), "cca_decode_fused",          \
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
                _PTR(new_token), _PTR(dw_weight), _OPTR(dwb, has_dw_bias),
                _PTR(conv_state), state_indices.data_ptr<int64_t>(),
                _PTR(gw_weight), _OPTR(gwb, has_gw_bias),
                _PTR(qk_mean), tmp.data_ptr<float>(),
                _PTR(output),
                B, G, D, E, state_len,
                static_cast<int>(num_q_heads),
                static_cast<float>(sqrt_head_dim),
                num_cache_lines, pad_slot_id, stream));
        } else if (D <= 128) {
            _DISPATCH(launch_fused<scalar_t, 128, DWB, GWB, CT>(
                _PTR(new_token), _PTR(dw_weight), _OPTR(dwb, has_dw_bias),
                _PTR(conv_state), state_indices.data_ptr<int64_t>(),
                _PTR(gw_weight), _OPTR(gwb, has_gw_bias),
                _PTR(qk_mean), tmp.data_ptr<float>(),
                _PTR(output),
                B, G, D, E, state_len,
                static_cast<int>(num_q_heads),
                static_cast<float>(sqrt_head_dim),
                num_cache_lines, pad_slot_id, stream));
        } else {
            _DISPATCH(launch_fused<scalar_t, 256, DWB, GWB, CT>(
                _PTR(new_token), _PTR(dw_weight), _OPTR(dwb, has_dw_bias),
                _PTR(conv_state), state_indices.data_ptr<int64_t>(),
                _PTR(gw_weight), _OPTR(gwb, has_gw_bias),
                _PTR(qk_mean), tmp.data_ptr<float>(),
                _PTR(output),
                B, G, D, E, state_len,
                static_cast<int>(num_q_heads),
                static_cast<float>(sqrt_head_dim),
                num_cache_lines, pad_slot_id, stream));
        }
    };

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
