/*
 * CCA Decode Fused Kernel — optimised for MI300X decode serving.
 *
 * Runtime shapes (from server.log):
 *   B=512  E=1280  G=10  D=128  state_len=2  qh=8  kh=2
 *   → grid(512,10)  block(128)  2 wavefronts on CDNA3
 *
 * Fuses in a SINGLE kernel launch per (batch, group):
 *
 *   Phase 1 — Depthwise causal conv1d update  (K=2)
 *   Phase 2 — Grouped conv1d  (W=2)   GEMV through LDS
 *   Phase 3 — qk_mean + L2-norm + scale + temperature
 *   Phase 4 — Write to output q/k buffer
 *
 * Optimisations vs. baseline:
 *   - Warp-shuffle L2-norm reduction: 3 barriers instead of 10
 *   - LDS reduced from (D*2+BLOCK_D) to (D*2+NUM_WARPS) floats
 *   - __launch_bounds__ for better register allocation
 *   - Explicit vectorised GEMV inner loop (#pragma unroll)
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

// Warp-shuffle butterfly reduction.  After this call **every lane** in the
// warp holds the full partial sum of its warp.
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SZ / 2; offset > 0; offset >>= 1)
        val += VLLM_SHFL_XOR_SYNC(val, offset);
    return val;
}

// Block-level sum using warp-shuffle + one LDS exchange.
// NUM_WARPS = ceil(BLOCK_D / WARP_SZ).  Returns the sum on ALL threads.
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
    __syncthreads();          // guard smem reuse
    return total;
}

// =======================================================================
// Fused kernel
//
// Grid  (B, G)      Block (BLOCK_D)
// LDS   D*2 floats (phase-1)  +  NUM_WARPS floats (reduction scratch)
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
    scalar_t*       __restrict__ output,      // [B, G*D]
    // Dimensions
    const int G, const int D,
    const int E,                               // = G * D
    const int state_len,
    const int num_q_heads,
    const float sqrt_head_dim,
    const int num_cache_lines,
    const int64_t pad_slot_id)
{
    const int i_n   = blockIdx.x;   // batch
    const int i_g   = blockIdx.y;   // group  (0..qh-1 = Q, qh..G-1 = K)
    const int d_out = threadIdx.x;  // output channel within group

    const int DW = D * 2;
    const int GD = G * D;

    // LDS layout: [D*2] phase1 output  |  [NUM_WARPS] reduce scratch
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

        #pragma unroll
        for (int p = 0; p < n_full; ++p) {
            int4 pk = w_vec[p];
            const scalar_t* e = reinterpret_cast<const scalar_t*>(&pk);
            const int base = p * PACK;
            #pragma unroll
            for (int j = 0; j < PACK; ++j)
                acc = fmaf(static_cast<float>(e[j]), s_phase1[base + j], acc);
        }
        #pragma unroll
        for (int k = n_full * PACK; k < DW; ++k)
            acc = fmaf(static_cast<float>(gw_row[k]), s_phase1[k], acc);
    } else {
        constexpr int PACK = 4;
        const int n_full      = DW / PACK;
        const float4* w_vec   = reinterpret_cast<const float4*>(gw_row);

        #pragma unroll
        for (int p = 0; p < n_full; ++p) {
            float4 v = w_vec[p];
            const int base = p * PACK;
            acc = fmaf(v.x, s_phase1[base + 0], acc);
            acc = fmaf(v.y, s_phase1[base + 1], acc);
            acc = fmaf(v.z, s_phase1[base + 2], acc);
            acc = fmaf(v.w, s_phase1[base + 3], acc);
        }
        #pragma unroll
        for (int k = n_full * PACK; k < DW; ++k)
            acc = fmaf(gw_row[k], s_phase1[k], acc);
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
    // Phase 4: Write to output[batch, ...]
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
            gw_weight, gw_bias, qk_mean, temp, output,
            G, D, E, state_len, num_q_heads, sqrt_head_dim,
            num_cache_lines, pad_slot_id);
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
    bool clamp_temp,
    int64_t pad_slot_id)
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

    auto output = torch::empty({B, G * D}, nt_c.options());

    const at::cuda::OptionalCUDAGuard guard(new_token.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

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
                num_cache_lines, pad_slot_id, stream));
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
                num_cache_lines, pad_slot_id, stream));
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
