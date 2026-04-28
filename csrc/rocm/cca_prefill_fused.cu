/*
 * CCA Prefill Fused Kernels — HIP/CUDA implementation.
 *
 * Fuses hs2-shift + depthwise conv1d + grouped conv1d + state updates
 * into 3 lightweight kernel launches (vs. N*~10 launches in the Python loop).
 *
 * Key design:  The conv kernel (Kernel 2) computes depthwise conv inline
 * per-thread and feeds results through LDS to the grouped GEMV — NO
 * intermediate global-memory buffer is needed.
 *
 *   Kernel 1 — hs2 shift + prev_hs update       Grid(T, ceil(H/BH))
 *   Kernel 2 — Fused DW conv + Grouped GEMV      Grid(T, G) Block(D)
 *   Kernel 3 — conv_state save                    Grid(N, ceil(E/BE))
 */

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#ifdef USE_ROCM
  #include <hip/hip_runtime.h>
  #define WARP_SZ 64
#else
  #include <cuda_runtime.h>
  #define WARP_SZ 32
#endif

namespace {

// ====================================================================
// Kernel 1 — hs2 shift + prev_hs update
// Grid(T, ceil(H/BLOCK_H))   Block(BLOCK_H)
// ====================================================================
template <typename scalar_t, int BLOCK_H>
__global__ void cca_prefill_hs2_shift_kernel(
    const scalar_t* __restrict__ hs_p,         // [T, H]
    scalar_t*       __restrict__ prev_hs,      // [num_cache, H] — mutated
    scalar_t*       __restrict__ hs2,          // [T, H]
    const int64_t*  __restrict__ qsl,          // [num_prefills+1]
    const int32_t*  __restrict__ has_initial,  // [num_prefills] (0/1)
    const int64_t*  __restrict__ state_idx,    // [num_prefills]
    const int32_t*  __restrict__ req_idx,      // [T]
    const int T, const int H,
    const int64_t prev_hs_stride_row)
{
    const int t = blockIdx.x;
    const int h = blockIdx.y * BLOCK_H + threadIdx.x;
    if (t >= T || h >= H) return;

    const int ri = req_idx[t];
    const int64_t start_i = qsl[ri];
    const int64_t end_i   = qsl[ri + 1];
    const int64_t slot    = state_idx[ri];

    float val;
    if (t == static_cast<int>(start_i)) {
        val = has_initial[ri]
            ? static_cast<float>(prev_hs[slot * prev_hs_stride_row + h])
            : 0.0f;
    } else {
        val = static_cast<float>(hs_p[(t - 1) * H + h]);
    }
    hs2[t * H + h] = static_cast<scalar_t>(val);

    if (t == static_cast<int>(end_i - 1))
        prev_hs[slot * prev_hs_stride_row + h] = hs_p[t * H + h];
}

// ====================================================================
// Kernel 2 — Fused depthwise conv1d (K=2) + grouped conv1d (K=2)
//
// Grid(T, G)   Block(BLOCK_D)
// LDS: D*2 floats (depthwise output for hist/curr)
//
// Each thread handles one output channel within its group.
// Phase 1: compute DW conv inline (3 padded-input loads + 2 FMA)
// Phase 2: grouped GEMV through LDS (vectorised inner loop)
// ====================================================================
template <typename scalar_t, int BLOCK_D,
          bool HAS_DW_BIAS, bool HAS_GW_BIAS>
__global__ void __launch_bounds__(BLOCK_D)
cca_prefill_conv_fused_kernel(
    const scalar_t* __restrict__ qk_p,         // [T, E]
    const scalar_t* __restrict__ dw_weight,    // [E, 2]
    const scalar_t* __restrict__ dw_bias,      // [E]
    const scalar_t* __restrict__ conv_state,   // [num_cache, E, state_len]
    const scalar_t* __restrict__ gw_weight,    // [G, D, D, 2]
    const scalar_t* __restrict__ gw_bias,      // [G, D]
    const int64_t*  __restrict__ qsl,
    const int32_t*  __restrict__ has_initial,
    const int64_t*  __restrict__ state_idx,
    const int32_t*  __restrict__ req_idx,
    scalar_t*       __restrict__ output,       // [T, E]
    const int G, const int D, const int E,
    const int state_len, const int T,
    const int64_t cs_stride_row,
    const int64_t cs_stride_ch)
{
    const int t_out = blockIdx.x;
    const int g     = blockIdx.y;
    const int d     = threadIdx.x;
    if (t_out >= T || d >= D) return;

    const int ch = g * D + d;
    const int DW = D * 2;

    const int ri = req_idx[t_out];
    const int64_t start_i = qsl[ri];
    const int64_t slot    = state_idx[ri];
    const int has_init    = has_initial[ri];
    const int t_local     = t_out - static_cast<int>(start_i);

    // ---- Phase 1: depthwise conv (K=2) ----
    // Padded input: [state[0], state[1], tok_0, tok_1, …]
    // We need positions t_local, t_local+1, t_local+2
    auto load_padded = [&](int p) -> float {
        if (p < 2)
            return has_init
                ? static_cast<float>(conv_state[slot * cs_stride_row + ch * cs_stride_ch + p])
                : 0.0f;
        return static_cast<float>(qk_p[(start_i + p - 2) * E + ch]);
    };

    const float inp0 = load_padded(t_local);
    const float inp1 = load_padded(t_local + 1);
    const float inp2 = load_padded(t_local + 2);

    const float w0 = static_cast<float>(dw_weight[ch * 2 + 0]);
    const float w1 = static_cast<float>(dw_weight[ch * 2 + 1]);
    float db = 0.0f;
    if constexpr (HAS_DW_BIAS)
        db = static_cast<float>(dw_bias[ch]);

    const float dw_hist = fmaf(w0, inp0, fmaf(w1, inp1, db));
    const float dw_curr = fmaf(w0, inp1, fmaf(w1, inp2, db));

    // ---- Store to LDS ----
    extern __shared__ char raw_smem[];
    float* smem = reinterpret_cast<float*>(raw_smem);
    smem[d * 2 + 0] = dw_hist;
    smem[d * 2 + 1] = dw_curr;
    __syncthreads();

    // ---- Phase 2: grouped GEMV ----
    // gw_weight layout [G, D_out, D_in, 2] — contiguous for (d_in, k)
    const scalar_t* gw_row = gw_weight
        + static_cast<int64_t>(g) * D * DW + d * DW;

    float acc = 0.0f;

    if constexpr (sizeof(scalar_t) <= 2) {
        constexpr int PACK = 8;
        const int n_full  = DW / PACK;
        const int4* w_vec = reinterpret_cast<const int4*>(gw_row);
        #pragma unroll
        for (int p = 0; p < n_full; ++p) {
            int4 pk = w_vec[p];
            const scalar_t* e = reinterpret_cast<const scalar_t*>(&pk);
            const int base = p * PACK;
            #pragma unroll
            for (int j = 0; j < PACK; ++j)
                acc = fmaf(static_cast<float>(e[j]), smem[base + j], acc);
        }
        #pragma unroll
        for (int k = n_full * PACK; k < DW; ++k)
            acc = fmaf(static_cast<float>(gw_row[k]), smem[k], acc);
    } else {
        constexpr int PACK = 4;
        const int n_full      = DW / PACK;
        const float4* w_vec   = reinterpret_cast<const float4*>(gw_row);
        #pragma unroll
        for (int p = 0; p < n_full; ++p) {
            float4 v = w_vec[p];
            const int base = p * PACK;
            acc = fmaf(v.x, smem[base + 0], acc);
            acc = fmaf(v.y, smem[base + 1], acc);
            acc = fmaf(v.z, smem[base + 2], acc);
            acc = fmaf(v.w, smem[base + 3], acc);
        }
        #pragma unroll
        for (int k = n_full * PACK; k < DW; ++k)
            acc = fmaf(gw_row[k], smem[k], acc);
    }

    if constexpr (HAS_GW_BIAS)
        acc += static_cast<float>(gw_bias[g * D + d]);

    output[t_out * E + ch] = static_cast<scalar_t>(acc);
}

// ====================================================================
// Kernel 3 — Save conv_states (last 2 raw input values per request)
// Grid(num_prefills, ceil(E/BLOCK_E))   Block(BLOCK_E)
// ====================================================================
template <typename scalar_t, int BLOCK_E>
__global__ void cca_prefill_state_save_kernel(
    const scalar_t* __restrict__ qk_p,         // [T, E]
    const scalar_t* __restrict__ conv_state_in, // [num_cache, E, sl]
    scalar_t*       __restrict__ conv_state_out,// [num_cache, E, sl] (may alias in)
    const int64_t*  __restrict__ qsl,
    const int32_t*  __restrict__ has_initial,
    const int64_t*  __restrict__ state_idx,
    const int N, const int E, const int sl,
    const int64_t cs_stride_row,
    const int64_t cs_stride_ch)
{
    const int ri = blockIdx.x;
    const int ch = blockIdx.y * BLOCK_E + threadIdx.x;
    if (ri >= N || ch >= E) return;

    const int64_t start_i = qsl[ri];
    const int64_t end_i   = qsl[ri + 1];
    const int64_t slot    = state_idx[ri];
    const int s_cur       = static_cast<int>(end_i - start_i);

    float v0, v1;
    if (s_cur >= 2) {
        v0 = static_cast<float>(qk_p[(end_i - 2) * E + ch]);
        v1 = static_cast<float>(qk_p[(end_i - 1) * E + ch]);
    } else {
        v0 = has_initial[ri]
           ? static_cast<float>(conv_state_in[slot * cs_stride_row + ch * cs_stride_ch + 1])
           : 0.0f;
        v1 = static_cast<float>(qk_p[start_i * E + ch]);
    }

    const int64_t base = slot * cs_stride_row + ch * cs_stride_ch;
    conv_state_out[base + 0] = static_cast<scalar_t>(v0);
    conv_state_out[base + 1] = static_cast<scalar_t>(v1);
}

// ====================================================================
// Typed launcher helpers (avoid <<<>>> inside macro args)
// ====================================================================
template <typename scalar_t, int BH>
void launch_hs2_shift(
    const scalar_t* hs_p, scalar_t* prev_hs, scalar_t* hs2,
    const int64_t* qsl, const int32_t* hi, const int64_t* si,
    const int32_t* ri, int T, int H, int64_t prev_hs_stride_row,
    cudaStream_t stream)
{
    dim3 grid(T, (H + BH - 1) / BH);
    cca_prefill_hs2_shift_kernel<scalar_t, BH>
        <<<grid, BH, 0, stream>>>(hs_p, prev_hs, hs2, qsl, hi, si, ri, T, H,
                                   prev_hs_stride_row);
}

template <typename scalar_t, int BLOCK_D,
          bool HAS_DW_BIAS, bool HAS_GW_BIAS>
void launch_prefill_conv(
    const scalar_t* qk_p, const scalar_t* dw_w, const scalar_t* dw_b,
    const scalar_t* cs, const scalar_t* gw_w, const scalar_t* gw_b,
    const int64_t* qsl, const int32_t* hi, const int64_t* si,
    const int32_t* ri, scalar_t* out,
    int G, int D, int E, int sl, int T,
    int64_t cs_stride_row, int64_t cs_stride_ch,
    cudaStream_t stream)
{
    dim3 grid(T, G);
    int smem = D * 2 * static_cast<int>(sizeof(float));
    cca_prefill_conv_fused_kernel<scalar_t, BLOCK_D,
                                  HAS_DW_BIAS, HAS_GW_BIAS>
        <<<grid, BLOCK_D, smem, stream>>>(
            qk_p, dw_w, dw_b, cs, gw_w, gw_b,
            qsl, hi, si, ri, out,
            G, D, E, sl, T,
            cs_stride_row, cs_stride_ch);
}

template <typename scalar_t, int BE>
void launch_state_save(
    const scalar_t* qk_p, const scalar_t* cs_in, scalar_t* cs_out,
    const int64_t* qsl, const int32_t* hi, const int64_t* si,
    int N, int E, int sl,
    int64_t cs_stride_row, int64_t cs_stride_ch,
    cudaStream_t stream)
{
    dim3 grid(N, (E + BE - 1) / BE);
    cca_prefill_state_save_kernel<scalar_t, BE>
        <<<grid, BE, 0, stream>>>(qk_p, cs_in, cs_out, qsl, hi, si, N, E, sl,
                                   cs_stride_row, cs_stride_ch);
}

}  // anonymous namespace

// ====================================================================
// Public entry point
// ====================================================================
std::vector<torch::Tensor> cca_prefill_fused_hip(
    const torch::Tensor& hs_p,              // [T, H]
    const torch::Tensor& qk_packed0_p,      // [T, E]
    torch::Tensor& prev_hs,                 // [num_cache, H] — mutated
    torch::Tensor& conv_states,             // [num_cache, E, sl] — mutated
    const torch::Tensor& query_start_loc,   // [N+1]
    const torch::Tensor& has_initial_i32,   // [N] int32 (0/1)
    const torch::Tensor& state_indices,     // [N] int64
    const torch::Tensor& req_idx,           // [T] int32
    const torch::Tensor& dw_weight,         // [E, 2]
    const std::optional<torch::Tensor>& dw_bias_opt,
    const torch::Tensor& gw_weight,         // [G, D, D, 2]
    const std::optional<torch::Tensor>& gw_bias_opt)
{
    TORCH_CHECK(hs_p.is_cuda(), "hs_p must be on CUDA/HIP");

    const int T = hs_p.size(0);
    const int H = hs_p.size(1);
    const int E = qk_packed0_p.size(1);
    const int N = query_start_loc.size(0) - 1;
    const int G = gw_weight.size(0);
    const int D = gw_weight.size(1);
    const int sl = conv_states.size(2);

    const bool has_dw_bias = dw_bias_opt.has_value() && dw_bias_opt->defined();
    const bool has_gw_bias = gw_bias_opt.has_value() && gw_bias_opt->defined();

    TORCH_CHECK(conv_states.dim() == 3,
                "conv_state must be 3-D [num_cache, E, state_len]");
    TORCH_CHECK(conv_states.stride(2) == 1,
                "conv_state innermost (state_len) dim must be contiguous, "
                "got stride(2)=", conv_states.stride(2));
    TORCH_CHECK(prev_hs.dim() == 2,
                "prev_hs must be 2-D [num_cache, H]");
    TORCH_CHECK(prev_hs.stride(1) == 1,
                "prev_hs innermost (H) dim must be contiguous, "
                "got stride(1)=", prev_hs.stride(1));

    const int64_t cs_stride_row = conv_states.stride(0);
    const int64_t cs_stride_ch  = conv_states.stride(1);
    const int64_t prev_hs_stride_row = prev_hs.stride(0);

    auto hs_c  = hs_p.contiguous();
    auto qk_c  = qk_packed0_p.contiguous();
    auto dww   = dw_weight.contiguous();
    auto gww   = gw_weight.contiguous();
    auto qsl   = query_start_loc.to(torch::kInt64).contiguous();
    auto hi    = has_initial_i32.to(torch::kInt32).contiguous();
    auto si    = state_indices.to(torch::kInt64).contiguous();
    auto ri    = req_idx.to(torch::kInt32).contiguous();

    torch::Tensor dwb, gwb;
    if (has_dw_bias) dwb = dw_bias_opt->contiguous();
    if (has_gw_bias) gwb = gw_bias_opt->contiguous();

    auto hs2    = torch::empty({T, H}, hs_c.options());
    auto output = torch::empty({T, E}, qk_c.options());

    const at::cuda::OptionalCUDAGuard guard(hs_p.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // ---- macros (same style as cca_decode_fused) ----
#define _DISPATCH(...)                                                        \
    AT_DISPATCH_SWITCH(hs_c.scalar_type(), "cca_prefill_fused",               \
        AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] { __VA_ARGS__; })     \
        AT_DISPATCH_CASE(at::ScalarType::Half,     [&] { __VA_ARGS__; })     \
        AT_DISPATCH_CASE(at::ScalarType::Float,    [&] { __VA_ARGS__; })     \
    )
#define _PTR(t) (t).data_ptr<scalar_t>()
#define _OPTR(t, c) ((c) ? (t).data_ptr<scalar_t>() : static_cast<scalar_t*>(nullptr))

    // ---- Kernel 1: hs2 shift ----
    {
        constexpr int BH = 256;
        _DISPATCH(launch_hs2_shift<scalar_t, BH>(
            _PTR(hs_c), _PTR(prev_hs), _PTR(hs2),
            qsl.data_ptr<int64_t>(), hi.data_ptr<int32_t>(),
            si.data_ptr<int64_t>(), ri.data_ptr<int32_t>(),
            T, H, prev_hs_stride_row, stream));
    }

    // ---- Kernel 2: fused DW + grouped conv ----
    {
        auto run = [&](auto dw_tag, auto gw_tag) {
            constexpr bool DWB = decltype(dw_tag)::value;
            constexpr bool GWB = decltype(gw_tag)::value;
            if (D <= 64) {
                _DISPATCH(launch_prefill_conv<scalar_t, 64, DWB, GWB>(
                    _PTR(qk_c), _PTR(dww), _OPTR(dwb, has_dw_bias),
                    _PTR(conv_states), _PTR(gww), _OPTR(gwb, has_gw_bias),
                    qsl.data_ptr<int64_t>(), hi.data_ptr<int32_t>(),
                    si.data_ptr<int64_t>(), ri.data_ptr<int32_t>(),
                    _PTR(output), G, D, E, sl, T,
                    cs_stride_row, cs_stride_ch, stream));
            } else if (D <= 128) {
                _DISPATCH(launch_prefill_conv<scalar_t, 128, DWB, GWB>(
                    _PTR(qk_c), _PTR(dww), _OPTR(dwb, has_dw_bias),
                    _PTR(conv_states), _PTR(gww), _OPTR(gwb, has_gw_bias),
                    qsl.data_ptr<int64_t>(), hi.data_ptr<int32_t>(),
                    si.data_ptr<int64_t>(), ri.data_ptr<int32_t>(),
                    _PTR(output), G, D, E, sl, T,
                    cs_stride_row, cs_stride_ch, stream));
            } else {
                _DISPATCH(launch_prefill_conv<scalar_t, 256, DWB, GWB>(
                    _PTR(qk_c), _PTR(dww), _OPTR(dwb, has_dw_bias),
                    _PTR(conv_states), _PTR(gww), _OPTR(gwb, has_gw_bias),
                    qsl.data_ptr<int64_t>(), hi.data_ptr<int32_t>(),
                    si.data_ptr<int64_t>(), ri.data_ptr<int32_t>(),
                    _PTR(output), G, D, E, sl, T,
                    cs_stride_row, cs_stride_ch, stream));
            }
        };
        if (has_dw_bias) {
            if (has_gw_bias) run(std::true_type{}, std::true_type{});
            else             run(std::true_type{}, std::false_type{});
        } else {
            if (has_gw_bias) run(std::false_type{}, std::true_type{});
            else             run(std::false_type{}, std::false_type{});
        }
    }

    // ---- Kernel 3: save conv_states ----
    {
        constexpr int BE = 256;
        _DISPATCH(launch_state_save<scalar_t, BE>(
            _PTR(qk_c), _PTR(conv_states), _PTR(conv_states),
            qsl.data_ptr<int64_t>(), hi.data_ptr<int32_t>(),
            si.data_ptr<int64_t>(), N, E, sl,
            cs_stride_row, cs_stride_ch, stream));
    }

#undef _DISPATCH
#undef _PTR
#undef _OPTR

    return {hs2, output};
}
