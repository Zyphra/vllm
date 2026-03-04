/*
 * HIP/CUDA kernel: Grouped 1D Convolution with W=2 for Decode
 *
 * Equivalent to Triton `_grouped_conv1d_w2_decode_kernel` in cca.py.
 *
 * Per (batch b, group g):
 *   y[b,g,d] = Σ_k weight[g,d,k,0]*x[b,g,k,0]
 *            + Σ_k weight[g,d,k,1]*x[b,g,k,1]
 *            + bias[g,d]
 *
 * Batched GEMV view:
 *   y = weight.reshape(G, D, D*2) @ x.reshape(B, G, D*2) + bias
 *
 * Contiguous layouts:
 *   x      [B, G, D, 2]    weight [G, D, D, 2]
 *   bias   [G, D]           y      [B, G, D]
 */

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#ifdef USE_ROCM
  #include <hip/hip_runtime.h>
#else
  #include <cuda_runtime.h>
#endif

namespace {

// ===================================================================
// Kernel 1 – Basic: one thread per output element
//
// Grid  (B, G)      Block (BLOCK_D)
// LDS   D*2 floats  (shared input vector)
// ===================================================================
template <typename scalar_t, int BLOCK_D, bool HAS_BIAS>
__global__ void gconv1d_w2_basic(
    const scalar_t* __restrict__ x,
    scalar_t*       __restrict__ y,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const int G, const int D)
{
    const int i_n   = blockIdx.x;
    const int i_g   = blockIdx.y;
    const int d_out = threadIdx.x;

    const int DW = D * 2;
    const int GD = G * D;

    extern __shared__ char smem[];
    float* s_x = reinterpret_cast<float*>(smem);

    // Cooperatively load x[b, g, :, :] → LDS
    const scalar_t* x_grp =
        x + static_cast<int64_t>(i_n) * GD * 2 + i_g * DW;

    for (int i = threadIdx.x; i < DW; i += BLOCK_D)
        s_x[i] = static_cast<float>(x_grp[i]);
    __syncthreads();

    if (d_out >= D) return;

    // Pointer to weight row: weight[g, d_out, :, :] (DW contiguous elements)
    const scalar_t* w_row =
        weight + static_cast<int64_t>(i_g) * D * DW + d_out * DW;

    float acc = 0.0f;

    // --- Vectorised inner-product ---
    if constexpr (sizeof(scalar_t) <= 2) {
        // bf16 / fp16: 128-bit loads → 8 elements
        constexpr int PACK = 8;
        const int n_full   = DW / PACK;
        const int4* w_vec  = reinterpret_cast<const int4*>(w_row);

        for (int p = 0; p < n_full; ++p) {
            int4 pk = w_vec[p];
            const scalar_t* e = reinterpret_cast<const scalar_t*>(&pk);
            const int base = p * PACK;
            #pragma unroll
            for (int j = 0; j < PACK; ++j)
                acc = fmaf(static_cast<float>(e[j]), s_x[base + j], acc);
        }
        for (int k = n_full * PACK; k < DW; ++k)
            acc = fmaf(static_cast<float>(w_row[k]), s_x[k], acc);
    } else {
        // fp32: 128-bit loads → 4 elements
        constexpr int PACK = 4;
        const int n_full      = DW / PACK;
        const float4* w_vec   = reinterpret_cast<const float4*>(w_row);

        for (int p = 0; p < n_full; ++p) {
            float4 v = w_vec[p];
            const int base = p * PACK;
            acc = fmaf(v.x, s_x[base + 0], acc);
            acc = fmaf(v.y, s_x[base + 1], acc);
            acc = fmaf(v.z, s_x[base + 2], acc);
            acc = fmaf(v.w, s_x[base + 3], acc);
        }
        for (int k = n_full * PACK; k < DW; ++k)
            acc = fmaf(w_row[k], s_x[k], acc);
    }

    if constexpr (HAS_BIAS)
        acc += static_cast<float>(bias[i_g * D + d_out]);

    y[static_cast<int64_t>(i_n) * GD + i_g * D + d_out] =
        static_cast<scalar_t>(acc);
}

// ===================================================================
// Kernel 2 – Tiled: TILES_K threads cooperate per output element
//
// Higher occupancy when B*G is small.
//
// Grid  (B, G)      Block (BLOCK_D, TILES_K)
// LDS   D*2 floats + BLOCK_D*TILES_K floats (partial sums)
// ===================================================================
template <typename scalar_t, int BLOCK_D, int TILES_K, bool HAS_BIAS>
__global__ void gconv1d_w2_tiled(
    const scalar_t* __restrict__ x,
    scalar_t*       __restrict__ y,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const int G, const int D)
{
    const int i_n   = blockIdx.x;
    const int i_g   = blockIdx.y;
    const int d_out = threadIdx.x;
    const int t_k   = threadIdx.y;
    const int tid   = t_k * BLOCK_D + d_out;
    const int nthds = BLOCK_D * TILES_K;

    const int DW = D * 2;
    const int GD = G * D;

    extern __shared__ char smem[];
    float* s_x       = reinterpret_cast<float*>(smem);
    float* s_partial = s_x + DW;  // [BLOCK_D * TILES_K]

    const scalar_t* x_grp =
        x + static_cast<int64_t>(i_n) * GD * 2 + i_g * DW;

    for (int i = tid; i < DW; i += nthds)
        s_x[i] = static_cast<float>(x_grp[i]);
    __syncthreads();

    if (d_out >= D) return;

    const scalar_t* w_row =
        weight + static_cast<int64_t>(i_g) * D * DW + d_out * DW;

    // Each tile handles a contiguous slice of the reduction dimension
    const int chunk   = (DW + TILES_K - 1) / TILES_K;
    const int k_start = t_k * chunk;
    const int k_end   = min(k_start + chunk, DW);

    float partial = 0.0f;
    #pragma unroll 8
    for (int k = k_start; k < k_end; ++k)
        partial = fmaf(static_cast<float>(w_row[k]), s_x[k], partial);

    s_partial[d_out * TILES_K + t_k] = partial;
    __syncthreads();

    // First tile reduces and writes the result
    if (t_k == 0) {
        float acc = s_partial[d_out * TILES_K];
        #pragma unroll
        for (int t = 1; t < TILES_K; ++t)
            acc += s_partial[d_out * TILES_K + t];

        if constexpr (HAS_BIAS)
            acc += static_cast<float>(bias[i_g * D + d_out]);

        y[static_cast<int64_t>(i_n) * GD + i_g * D + d_out] =
            static_cast<scalar_t>(acc);
    }
}

// ===================================================================
// Typed launcher helpers
// ===================================================================
template <typename scalar_t, int BLOCK_D, bool HAS_BIAS>
void launch_basic(const scalar_t* x, scalar_t* y,
                  const scalar_t* w, const scalar_t* b,
                  int B, int G, int D, cudaStream_t s)
{
    dim3 grid(B, G);
    int  smem = D * 2 * static_cast<int>(sizeof(float));
    gconv1d_w2_basic<scalar_t, BLOCK_D, HAS_BIAS>
        <<<grid, BLOCK_D, smem, s>>>(x, y, w, b, G, D);
}

template <typename scalar_t, int BLOCK_D, int TILES_K, bool HAS_BIAS>
void launch_tiled(const scalar_t* x, scalar_t* y,
                  const scalar_t* w, const scalar_t* b,
                  int B, int G, int D, cudaStream_t s)
{
    dim3 grid(B, G);
    dim3 block(BLOCK_D, TILES_K);
    int  smem = (D * 2 + BLOCK_D * TILES_K) * static_cast<int>(sizeof(float));
    gconv1d_w2_tiled<scalar_t, BLOCK_D, TILES_K, HAS_BIAS>
        <<<grid, block, smem, s>>>(x, y, w, b, G, D);
}

}  // anonymous namespace

// ===================================================================
// Public entry point
// ===================================================================
torch::Tensor grouped_conv1d_w2_decode(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const std::optional<torch::Tensor>& bias_opt)
{
    TORCH_CHECK(x.is_cuda(),         "x must be on CUDA/HIP");
    TORCH_CHECK(weight.is_cuda(),    "weight must be on CUDA/HIP");
    TORCH_CHECK(x.dim() == 4,        "x must be 4-D [B,G,D,W]");
    TORCH_CHECK(weight.dim() == 4,   "weight must be 4-D [G,D,D,W]");
    TORCH_CHECK(x.size(3) == 2,      "x W-dim must be 2");
    TORCH_CHECK(weight.size(3) == 2, "weight W-dim must be 2");

    const int B = x.size(0);
    const int G = x.size(1);
    const int D = x.size(2);
    TORCH_CHECK(weight.size(0) == G && weight.size(1) == D &&
                weight.size(2) == D, "weight shape mismatch");

    const bool has_bias = bias_opt.has_value() && bias_opt->defined();
    auto x_c = x.contiguous();
    auto w_c = weight.contiguous();
    torch::Tensor b_c;
    if (has_bias) {
        b_c = bias_opt->contiguous();
        TORCH_CHECK(b_c.dim() == 2 && b_c.size(0) == G && b_c.size(1) == D,
                     "bias shape must be [G, D]");
    }

    auto y = torch::empty({B, G, D}, x_c.options());
    const at::cuda::OptionalCUDAGuard guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const bool use_tiled = (B * G < 128) && (D >= 64);

    // ---- Dispatch (BLOCK_D, bias, dtype) ----
    // Use variadic macro to avoid comma-in-template-args issues.
#define _DISPATCH_DTYPE(...)                                                \
    AT_DISPATCH_SWITCH(x_c.scalar_type(), "grouped_conv1d_w2_decode",       \
        AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&] { __VA_ARGS__; })   \
        AT_DISPATCH_CASE(at::ScalarType::Half,     [&] { __VA_ARGS__; })   \
        AT_DISPATCH_CASE(at::ScalarType::Float,    [&] { __VA_ARGS__; })   \
    )

#define _ARGS                                                               \
    x_c.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(),                      \
    w_c.data_ptr<scalar_t>(),                                              \
    has_bias ? b_c.data_ptr<scalar_t>() : static_cast<scalar_t*>(nullptr), \
    B, G, D, stream

    auto run = [&](auto bias_tag) {
        constexpr bool HB = decltype(bias_tag)::value;
        if (use_tiled) {
            if (D <= 64) {
                _DISPATCH_DTYPE(launch_tiled<scalar_t, 64, 4, HB>(_ARGS));
            } else if (D <= 128) {
                _DISPATCH_DTYPE(launch_tiled<scalar_t, 128, 4, HB>(_ARGS));
            } else {
                _DISPATCH_DTYPE(launch_tiled<scalar_t, 256, 4, HB>(_ARGS));
            }
        } else {
            if (D <= 64) {
                _DISPATCH_DTYPE(launch_basic<scalar_t, 64, HB>(_ARGS));
            } else if (D <= 128) {
                _DISPATCH_DTYPE(launch_basic<scalar_t, 128, HB>(_ARGS));
            } else if (D <= 256) {
                _DISPATCH_DTYPE(launch_basic<scalar_t, 256, HB>(_ARGS));
            } else if (D <= 512) {
                _DISPATCH_DTYPE(launch_basic<scalar_t, 512, HB>(_ARGS));
            } else {
                _DISPATCH_DTYPE(launch_basic<scalar_t, 1024, HB>(_ARGS));
            }
        }
    };

    if (has_bias) run(std::true_type{});
    else          run(std::false_type{});

#undef _ARGS
#undef _DISPATCH_DTYPE

    return y;
}
