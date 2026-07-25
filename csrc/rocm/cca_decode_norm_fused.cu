// csrc/gfx942/cca_decode.cu
// Fused CCA decode (dev binding exposing BOTH qk_mean variants):
//   depthwise w2 conv-state update -> grouped w2 conv
//   -> qk_mean (folded from raw token, OR precomputed) -> L2 norm + sqrt_head_dim + per-k-head temp.
// Both variants must produce identical output; oracle is forward_triton's decode math.
// The [y1,y2] intermediate lives in LDS and never touches HBM. Target: gfx942 (MI300).
//
// CONFIG: K0==K1==2 -> conv_states width 2; head_dim D power of two.
// Grid (B, G), block D.  G = num_q_heads + num_k_heads (q first).  E = G*D.
#include <type_traits>
#include <cstdlib>
#include <cstdint>
#include <atomic>
#include <cstdio>

#if defined(__HIPCC__) && defined(__HIP__MI3XX__)
  #define CCA_USE_DIRECT_MFMA 1
#else
  #define CCA_USE_DIRECT_MFMA 0
#endif

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

static bool cca_env_enabled(const char* name, bool default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr) return default_value;
    return value[0] == '1' || value[0] == 't' || value[0] == 'T' ||
           value[0] == 'y' || value[0] == 'Y';
}

static bool cca_debug_launch_enabled() {
    return cca_env_enabled("VLLM_CCA_NORM_FUSED_DEBUG", false) ||
           cca_env_enabled("VLLM_CCA_DECODE_NORM_FUSED_DEBUG_LAUNCH", false);
}

static bool cca_debug_launch_should_log() {
    static std::atomic<int> count{0};
    return cca_debug_launch_enabled() && count.fetch_add(1) < 24;
}

template <typename scalar_t, int D, int GQA,
          bool HAS_DW_BIAS, bool HAS_GW_BIAS, bool CLAMP_TEMP,
          bool FOLD_QK_MEAN, bool APPLY_ROPE_KCACHE, bool IS_NEOX>
__global__ void cca_decode_fused_kernel(
        const scalar_t* __restrict__ first_input,   // (B, E)
        const scalar_t* __restrict__ dw_weight,     // (E, 2)
        const scalar_t* __restrict__ dw_bias,       // (E,) or nullptr
        scalar_t*       __restrict__ conv_states,   // (num_cache_lines, E, 2)
        const int*      __restrict__ state_indices, // (B,)
        const scalar_t* __restrict__ gw_weight,     // (G, D*2, D) repacked [g, ic*2+w, oc]
        const scalar_t* __restrict__ gw_bias,       // (G, D) or nullptr
        const scalar_t* __restrict__ qk_mean,       // (B, G, D) used only if !FOLD_QK_MEAN
        const float*    __restrict__ temp,          // (num_k_heads,) fp32
        scalar_t*       __restrict__ out,           // (B, E)
        const int64_t*  __restrict__ positions,     // (B,), optional RoPE positions
        const scalar_t* __restrict__ cos_sin_cache, // (max_pos, rotary_dim)
        const int64_t*  __restrict__ kv_slot_mapping, // (B,)
        scalar_t*       __restrict__ key_cache,     // (num_blocks, block, kv_heads, D)
        int rotary_dim, int key_cache_block_size,
        long key_cache_stride_block, long key_cache_stride_slot,
        long key_cache_stride_head, long key_cache_stride_channel,
        int B, int G, int E, int num_q_heads, int pad_slot_id,
        long first_stride_row, long first_stride_channel,
        long conv_stride_row, long conv_stride_channel, long conv_stride_token,
        long qk_mean_stride_row, long qk_mean_stride_group, long qk_mean_stride_channel,
        long out_stride_row, float sqrt_head_dim)
{
    constexpr float NORM_EPS = 1e-12f;

    const int b  = blockIdx.x;
    const int g  = blockIdx.y;
    const int oc = threadIdx.x;
    const int c  = g * D + oc;
    const bool is_query = (g < num_q_heads);

    __shared__ float y1[D];
    __shared__ float y2[D];
    __shared__ float sh_reduce[D];

    const long row = static_cast<long>(state_indices[b]);
    if (row == pad_slot_id) { out[(long)b * out_stride_row + c] = scalar_t(0); return; }

    scalar_t* st = conv_states + row * conv_stride_row + c * conv_stride_channel;
    const float x0 = static_cast<float>(st[0 * conv_stride_token]);
    const float x1 = static_cast<float>(st[1 * conv_stride_token]);
    const float x2 = static_cast<float>(
        first_input[(long)b * first_stride_row + c * first_stride_channel]);
    const float w0 = static_cast<float>(dw_weight[c * 2 + 0]);
    const float w1 = static_cast<float>(dw_weight[c * 2 + 1]);
    const float bd = HAS_DW_BIAS ? static_cast<float>(dw_bias[c]) : 0.0f;

    y1[oc] = bd + w0 * x0 + w1 * x1;
    y2[oc] = bd + w0 * x1 + w1 * x2;

    st[0 * conv_stride_token] = static_cast<scalar_t>(x1);
    st[1 * conv_stride_token] = static_cast<scalar_t>(x2);
    __syncthreads();

    const scalar_t* gwg = gw_weight + (long)g * (D * 2) * D;
    float acc = HAS_GW_BIAS ? static_cast<float>(gw_bias[g * D + oc]) : 0.0f;
    #pragma unroll 4
    for (int ic = 0; ic < D; ++ic) {
        acc += static_cast<float>(gwg[(ic * 2 + 0) * D + oc]) * y1[ic];
        acc += static_cast<float>(gwg[(ic * 2 + 1) * D + oc]) * y2[ic];
    }

    if (FOLD_QK_MEAN) {
        const scalar_t* fi = first_input + (long)b * first_stride_row;
        float qkm;
        if (is_query) {
            const int kc = num_q_heads * D + (g / GQA) * D + oc;
            qkm = 0.5f * (x2 + static_cast<float>(fi[kc * first_stride_channel]));
        } else {
            const int k = g - num_q_heads;
            float qsum = 0.0f;
            #pragma unroll
            for (int j = 0; j < GQA; ++j)
                qsum += static_cast<float>(
                    fi[((k * GQA + j) * D + oc) * first_stride_channel]);
            qkm = 0.5f * (qsum / GQA) + 0.5f * x2;
        }
        acc += qkm;
    } else {
        acc += static_cast<float>(
            qk_mean[(long)b * qk_mean_stride_row
                    + (long)g * qk_mean_stride_group
                    + (long)oc * qk_mean_stride_channel]);
    }

    sh_reduce[oc] = acc * acc;
    __syncthreads();
    #pragma unroll
    for (int s = D >> 1; s > 0; s >>= 1) {
        if (oc < s) sh_reduce[oc] += sh_reduce[oc + s];
        __syncthreads();
    }
    const float inv = rsqrtf(sh_reduce[0] + NORM_EPS);

    float scale = sqrt_head_dim;
    if (!is_query) {
        float t = temp[g - num_q_heads];
        if (CLAMP_TEMP) t = expf(fminf(fmaxf(t, 1e-7f), 2.0f));
        scale *= t;
    }
    float value = acc * inv * scale;
    sh_reduce[oc] = value;
    __syncthreads();

    if constexpr (APPLY_ROPE_KCACHE) {
        if (rotary_dim > 0 && oc < rotary_dim) {
            const int64_t pos = positions[b];
            const int embed_dim = rotary_dim >> 1;
            const scalar_t* cache_ptr = cos_sin_cache + pos * rotary_dim;
            int pair_idx;
            float x;
            float y;
            if constexpr (IS_NEOX) {
                if (oc < embed_dim) {
                    pair_idx = oc;
                    x = sh_reduce[oc];
                    y = sh_reduce[oc + embed_dim];
                    const float cos_val = static_cast<float>(cache_ptr[pair_idx]);
                    const float sin_val = static_cast<float>(
                        cache_ptr[pair_idx + embed_dim]);
                    value = x * cos_val - y * sin_val;
                } else {
                    pair_idx = oc - embed_dim;
                    x = sh_reduce[pair_idx];
                    y = sh_reduce[oc];
                    const float cos_val = static_cast<float>(cache_ptr[pair_idx]);
                    const float sin_val = static_cast<float>(
                        cache_ptr[pair_idx + embed_dim]);
                    value = y * cos_val + x * sin_val;
                }
            } else {
                pair_idx = oc >> 1;
                x = sh_reduce[pair_idx * 2];
                y = sh_reduce[pair_idx * 2 + 1];
                const float cos_val = static_cast<float>(cache_ptr[pair_idx]);
                const float sin_val = static_cast<float>(
                    cache_ptr[pair_idx + embed_dim]);
                value = (oc & 1) ? (y * cos_val + x * sin_val)
                                 : (x * cos_val - y * sin_val);
            }
        }
    }

    out[(long)b * out_stride_row + c] = static_cast<scalar_t>(value);

    if constexpr (APPLY_ROPE_KCACHE) {
        if (!is_query) {
            const int64_t slot = kv_slot_mapping[b];
            if (slot >= 0) {
                const int64_t block_idx = slot / key_cache_block_size;
                const int64_t block_offset = slot % key_cache_block_size;
                const int local_k = g - num_q_heads;
                key_cache[block_idx * key_cache_stride_block
                          + block_offset * key_cache_stride_slot
                          + (long)local_k * key_cache_stride_head
                          + (long)oc * key_cache_stride_channel] =
                    static_cast<scalar_t>(value);
            }
        }
    }
}

#if CCA_USE_DIRECT_MFMA
using cca_floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using cca_b16x4 =
    __attribute__((__vector_size__(4 * sizeof(uint16_t)))) uint16_t;

__device__ __forceinline__ cca_b16x4 cca_make_b16x4(
    uint16_t x0, uint16_t x1, uint16_t x2, uint16_t x3) {
    cca_b16x4 result;
    result[0] = x0;
    result[1] = x1;
    result[2] = x2;
    result[3] = x3;
    return result;
}

template <typename scalar_t>
__device__ __forceinline__ uint16_t cca_bf16_bits(float value) {
    union {
        scalar_t scalar;
        uint16_t bits;
    } cvt;
    cvt.scalar = static_cast<scalar_t>(value);
    return cvt.bits;
}

template <typename scalar_t, int D, int GQA,
          bool HAS_DW_BIAS, bool HAS_GW_BIAS, bool CLAMP_TEMP,
          bool FOLD_QK_MEAN, bool APPLY_ROPE_KCACHE, bool IS_NEOX>
__global__ void cca_decode_fused_tile16_mfma_kernel(
        const scalar_t* __restrict__ first_input,
        const scalar_t* __restrict__ dw_weight,
        const scalar_t* __restrict__ dw_bias,
        scalar_t*       __restrict__ conv_states,
        const int*      __restrict__ state_indices,
        const scalar_t* __restrict__ gw_weight,
        const scalar_t* __restrict__ gw_bias,
        const scalar_t* __restrict__ qk_mean,
        const float*    __restrict__ temp,
        scalar_t*       __restrict__ out,
        const int64_t*  __restrict__ positions,
        const scalar_t* __restrict__ cos_sin_cache,
        const int64_t*  __restrict__ kv_slot_mapping,
        scalar_t*       __restrict__ key_cache,
        int rotary_dim, int key_cache_block_size,
        long key_cache_stride_block, long key_cache_stride_slot,
        long key_cache_stride_head, long key_cache_stride_channel,
        int B, int G, int E, int num_q_heads, int pad_slot_id,
        long first_stride_row, long first_stride_channel,
        long conv_stride_row, long conv_stride_channel, long conv_stride_token,
        long qk_mean_stride_row, long qk_mean_stride_group, long qk_mean_stride_channel,
        long out_stride_row, float sqrt_head_dim)
{
    static_assert(D == 128, "tile16 MFMA path is currently specialized for D=128");
    constexpr int kTokenTile = 16;
    constexpr int kK = 2 * D;
    constexpr int kBlockSize = 256;
    constexpr int kWaveSize = 64;
    constexpr int kWaves = kBlockSize / kWaveSize;
    constexpr float NORM_EPS = 1e-12f;

    __shared__ uint16_t a_tile[kTokenTile * kK];
    __shared__ float head_values[kTokenTile * D];
    __shared__ float row_sums[kTokenTile];

    const int tile_idx = blockIdx.x;
    const int g = blockIdx.y;
    const int token_base = tile_idx * kTokenTile;
    const int tid = threadIdx.x;
    const int wave_id = tid / kWaveSize;
    const int lane_id = tid & (kWaveSize - 1);
    const int lane16_id = lane_id & 15;
    const int row4_id = lane_id >> 4;
    const bool is_query = (g < num_q_heads);
    const int key_head_idx = is_query ? (g / GQA) : (g - num_q_heads);
    const int channel_base = g * D;
    const int latent_q_dim = num_q_heads * D;
    const int key_channel_base = latent_q_dim + key_head_idx * D;

    for (int idx = tid; idx < kTokenTile * kK; idx += blockDim.x) {
        const int row_in_tile = idx / kK;
        const int k_idx = idx - row_in_tile * kK;
        const int in_dim = k_idx >> 1;
        const int time_idx = k_idx & 1;
        const int b = token_base + row_in_tile;
        float value = 0.0f;
        if (b < B) {
            const int state_row = state_indices[b];
            if (state_row != pad_slot_id && state_row >= 0) {
                const int c = channel_base + in_dim;
                scalar_t* st = conv_states + (long)state_row * conv_stride_row
                               + (long)c * conv_stride_channel;
                const float x0 = static_cast<float>(st[0 * conv_stride_token]);
                const float x1 = static_cast<float>(st[1 * conv_stride_token]);
                const float x2 = static_cast<float>(
                    first_input[(long)b * first_stride_row
                                + (long)c * first_stride_channel]);
                const float w0 = static_cast<float>(dw_weight[c * 2 + 0]);
                const float w1 = static_cast<float>(dw_weight[c * 2 + 1]);
                const float bd = HAS_DW_BIAS ? static_cast<float>(dw_bias[c]) : 0.0f;
                value = (time_idx == 0) ? (bd + w0 * x0 + w1 * x1)
                                        : (bd + w0 * x1 + w1 * x2);
            }
        }
        a_tile[idx] = cca_bf16_bits<scalar_t>(value);
    }
    __syncthreads();

    const uint16_t* gwg_bits = reinterpret_cast<const uint16_t*>(
        gw_weight + (long)g * kK * D);
    for (int col_tile = wave_id; col_tile < D / 16; col_tile += kWaves) {
        const int col_base = col_tile * 16;
        cca_floatx4 acc = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int k_base = 0; k_base < kK; k_base += 16) {
            const int k_lane_base = k_base + row4_id * 4;
            const cca_b16x4 a_frag = cca_make_b16x4(
                a_tile[(long)lane16_id * kK + k_lane_base + 0],
                a_tile[(long)lane16_id * kK + k_lane_base + 1],
                a_tile[(long)lane16_id * kK + k_lane_base + 2],
                a_tile[(long)lane16_id * kK + k_lane_base + 3]);
            const int b_col = col_base + lane16_id;
            const cca_b16x4 b_frag = cca_make_b16x4(
                gwg_bits[(long)(k_lane_base + 0) * D + b_col],
                gwg_bits[(long)(k_lane_base + 1) * D + b_col],
                gwg_bits[(long)(k_lane_base + 2) * D + b_col],
                gwg_bits[(long)(k_lane_base + 3) * D + b_col]);
            acc = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(
                a_frag, b_frag, acc, 0, 0, 0);
        }
        for (int i = 0; i < 4; ++i) {
            const int row_in_tile = row4_id * 4 + i;
            head_values[(long)row_in_tile * D + col_base + lane16_id] = acc[i];
        }
    }
    __syncthreads();

    if (tid < kTokenTile) {
        row_sums[tid] = 0.0f;
    }
    __syncthreads();

    for (int idx = tid; idx < kTokenTile * D; idx += blockDim.x) {
        const int row_in_tile = idx / D;
        const int oc = idx - row_in_tile * D;
        const int b = token_base + row_in_tile;
        float value = 0.0f;
        if (b < B) {
            const int state_row = state_indices[b];
            if (state_row != pad_slot_id && state_row >= 0) {
                const int c = channel_base + oc;
                value = head_values[idx];
                if (HAS_GW_BIAS) {
                    value += static_cast<float>(gw_bias[g * D + oc]);
                }
                if (FOLD_QK_MEAN) {
                    const scalar_t* fi = first_input + (long)b * first_stride_row;
                    if (is_query) {
                        const int kc = key_channel_base + oc;
                        const float query_pre = static_cast<float>(
                            fi[(long)c * first_stride_channel]);
                        const float key_pre = static_cast<float>(
                            fi[(long)kc * first_stride_channel]);
                        value += 0.5f * query_pre + 0.5f * key_pre;
                    } else {
                        float qsum = 0.0f;
                        const int first_q_head = key_head_idx * GQA;
                        #pragma unroll
                        for (int j = 0; j < GQA; ++j) {
                            const int qc = (first_q_head + j) * D + oc;
                            qsum += static_cast<float>(
                                fi[(long)qc * first_stride_channel]);
                        }
                        const float key_pre = static_cast<float>(
                            fi[(long)c * first_stride_channel]);
                        value += 0.5f * (qsum / GQA) + 0.5f * key_pre;
                    }
                } else {
                    value += static_cast<float>(
                        qk_mean[(long)b * qk_mean_stride_row
                                + (long)g * qk_mean_stride_group
                                + (long)oc * qk_mean_stride_channel]);
                }
            }
        }
        head_values[idx] = value;
        if (b < B && state_indices[b] != pad_slot_id && state_indices[b] >= 0) {
            atomicAdd(&row_sums[row_in_tile], value * value);
        }
    }
    __syncthreads();

    for (int idx = tid; idx < kTokenTile * D; idx += blockDim.x) {
        const int row_in_tile = idx / D;
        const int b = token_base + row_in_tile;
        float value = 0.0f;
        if (b < B) {
            const int state_row = state_indices[b];
            if (state_row != pad_slot_id && state_row >= 0) {
                float scale = sqrt_head_dim * rsqrtf(row_sums[row_in_tile] + NORM_EPS);
                if (!is_query) {
                    float t = temp[key_head_idx];
                    if (CLAMP_TEMP) t = expf(fminf(fmaxf(t, 1e-7f), 2.0f));
                    scale *= t;
                }
                value = head_values[idx] * scale;
            }
        }
        head_values[idx] = value;
    }
    __syncthreads();

    for (int idx = tid; idx < kTokenTile * D; idx += blockDim.x) {
        const int row_in_tile = idx / D;
        const int oc = idx - row_in_tile * D;
        const int b = token_base + row_in_tile;
        if (b >= B) continue;
        const int c = channel_base + oc;
        float value = head_values[idx];

        if constexpr (APPLY_ROPE_KCACHE) {
            if (rotary_dim > 0 && oc < rotary_dim) {
                const int64_t pos = positions[b];
                const int embed_dim = rotary_dim >> 1;
                const scalar_t* cache_ptr = cos_sin_cache + pos * rotary_dim;
                if constexpr (IS_NEOX) {
                    int pair_idx;
                    float x;
                    float y;
                    if (oc < embed_dim) {
                        pair_idx = oc;
                        x = head_values[row_in_tile * D + oc];
                        y = head_values[row_in_tile * D + oc + embed_dim];
                        const float cos_val = static_cast<float>(cache_ptr[pair_idx]);
                        const float sin_val = static_cast<float>(
                            cache_ptr[pair_idx + embed_dim]);
                        value = x * cos_val - y * sin_val;
                    } else {
                        pair_idx = oc - embed_dim;
                        x = head_values[row_in_tile * D + pair_idx];
                        y = head_values[row_in_tile * D + oc];
                        const float cos_val = static_cast<float>(cache_ptr[pair_idx]);
                        const float sin_val = static_cast<float>(
                            cache_ptr[pair_idx + embed_dim]);
                        value = y * cos_val + x * sin_val;
                    }
                } else {
                    const int pair_idx = oc >> 1;
                    const float x = head_values[row_in_tile * D + pair_idx * 2];
                    const float y = head_values[row_in_tile * D + pair_idx * 2 + 1];
                    const float cos_val = static_cast<float>(cache_ptr[pair_idx]);
                    const float sin_val = static_cast<float>(
                        cache_ptr[pair_idx + embed_dim]);
                    value = (oc & 1) ? (y * cos_val + x * sin_val)
                                     : (x * cos_val - y * sin_val);
                }
            }
        }

        out[(long)b * out_stride_row + c] = static_cast<scalar_t>(value);

        if constexpr (APPLY_ROPE_KCACHE) {
            if (!is_query) {
                const int64_t slot = kv_slot_mapping[b];
                if (slot >= 0) {
                    const int64_t block_idx = slot / key_cache_block_size;
                    const int64_t block_offset = slot % key_cache_block_size;
                    key_cache[block_idx * key_cache_stride_block
                              + block_offset * key_cache_stride_slot
                              + (long)key_head_idx * key_cache_stride_head
                              + (long)oc * key_cache_stride_channel] =
                        static_cast<scalar_t>(value);
                }
            }
        }
    }
    __syncthreads();

    for (int idx = tid; idx < kTokenTile * D; idx += blockDim.x) {
        const int row_in_tile = idx / D;
        const int oc = idx - row_in_tile * D;
        const int b = token_base + row_in_tile;
        if (b >= B) continue;
        const int state_row = state_indices[b];
        if (state_row == pad_slot_id || state_row < 0) continue;
        const int c = channel_base + oc;
        scalar_t* st = conv_states + (long)state_row * conv_stride_row
                       + (long)c * conv_stride_channel;
        const float x1 = static_cast<float>(st[1 * conv_stride_token]);
        const float x2 = static_cast<float>(
            first_input[(long)b * first_stride_row
                        + (long)c * first_stride_channel]);
        st[0 * conv_stride_token] = static_cast<scalar_t>(x1);
        st[1 * conv_stride_token] = static_cast<scalar_t>(x2);
    }
}
#endif

template <typename scalar_t, int D, int GQA, bool APPLY_ROPE_KCACHE, bool IS_NEOX>
static bool launch_tile16_mfma_flags(
        const scalar_t* fi, const scalar_t* dw_w, const scalar_t* dw_b,
        scalar_t* cs, const int* idx, const scalar_t* gw_w, const scalar_t* gw_b,
        const scalar_t* qkm, const float* temp, scalar_t* out,
        const int64_t* positions, const scalar_t* cos_sin_cache,
        const int64_t* kv_slot_mapping, scalar_t* key_cache,
        int rotary_dim, int key_cache_block_size,
        long key_cache_stride_block, long key_cache_stride_slot,
        long key_cache_stride_head, long key_cache_stride_channel,
        int B, int G, int E, int nq, int pad_slot_id, long out_stride_row,
        long first_stride_row, long first_stride_channel,
        long conv_stride_row, long conv_stride_channel, long conv_stride_token,
        long qk_mean_stride_row, long qk_mean_stride_group, long qk_mean_stride_channel,
        float sqrt_hd,
        bool has_dw_b, bool has_gw_b, bool clamp_temp, bool fold, cudaStream_t s)
{
#if CCA_USE_DIRECT_MFMA
    if constexpr (D == 128 && (std::is_same_v<scalar_t, c10::BFloat16> ||
                               std::is_same_v<scalar_t, at::BFloat16>)) {
        const dim3 grid((B + 15) / 16, G);
        const dim3 block(256);
        #define LAUNCH_TILE(DW,GW,CT,FQ) \
            cca_decode_fused_tile16_mfma_kernel<scalar_t,D,GQA,DW,GW,CT,FQ,APPLY_ROPE_KCACHE,IS_NEOX><<<grid,block,0,s>>>( \
                fi,dw_w,dw_b,cs,idx,gw_w,gw_b,qkm,temp,out, \
                positions,cos_sin_cache,kv_slot_mapping,key_cache,rotary_dim, \
                key_cache_block_size,key_cache_stride_block,key_cache_stride_slot, \
                key_cache_stride_head,key_cache_stride_channel, \
                B,G,E,nq,pad_slot_id, \
                first_stride_row,first_stride_channel,conv_stride_row,conv_stride_channel, \
                conv_stride_token,qk_mean_stride_row,qk_mean_stride_group, \
                qk_mean_stride_channel,out_stride_row,sqrt_hd)
        const int sel=(has_dw_b?8:0)|(has_gw_b?4:0)|(clamp_temp?2:0)|(fold?1:0);
        switch (sel) {
            case 0:  LAUNCH_TILE(false,false,false,false); break;
            case 1:  LAUNCH_TILE(false,false,false,true ); break;
            case 2:  LAUNCH_TILE(false,false,true ,false); break;
            case 3:  LAUNCH_TILE(false,false,true ,true ); break;
            case 4:  LAUNCH_TILE(false,true ,false,false); break;
            case 5:  LAUNCH_TILE(false,true ,false,true ); break;
            case 6:  LAUNCH_TILE(false,true ,true ,false); break;
            case 7:  LAUNCH_TILE(false,true ,true ,true ); break;
            case 8:  LAUNCH_TILE(true ,false,false,false); break;
            case 9:  LAUNCH_TILE(true ,false,false,true ); break;
            case 10: LAUNCH_TILE(true ,false,true ,false); break;
            case 11: LAUNCH_TILE(true ,false,true ,true ); break;
            case 12: LAUNCH_TILE(true ,true ,false,false); break;
            case 13: LAUNCH_TILE(true ,true ,false,true ); break;
            case 14: LAUNCH_TILE(true ,true ,true ,false); break;
            case 15: LAUNCH_TILE(true ,true ,true ,true ); break;
        }
        #undef LAUNCH_TILE
        return true;
    }
#endif
    return false;
}

template <typename scalar_t, int D, int GQA, bool APPLY_ROPE_KCACHE, bool IS_NEOX>
static void launch_flags(
        const scalar_t* fi, const scalar_t* dw_w, const scalar_t* dw_b,
        scalar_t* cs, const int* idx, const scalar_t* gw_w, const scalar_t* gw_b,
        const scalar_t* qkm, const float* temp, scalar_t* out,
        const int64_t* positions, const scalar_t* cos_sin_cache,
        const int64_t* kv_slot_mapping, scalar_t* key_cache,
        int rotary_dim, int key_cache_block_size,
        long key_cache_stride_block, long key_cache_stride_slot,
        long key_cache_stride_head, long key_cache_stride_channel,
        int B, int G, int E, int nq, int pad_slot_id, long out_stride_row,
        long first_stride_row, long first_stride_channel,
        long conv_stride_row, long conv_stride_channel, long conv_stride_token,
        long qk_mean_stride_row, long qk_mean_stride_group, long qk_mean_stride_channel,
        float sqrt_hd,
        bool has_dw_b, bool has_gw_b, bool clamp_temp, bool fold, cudaStream_t s,
        bool use_tile16_mfma)
{
    if (use_tile16_mfma && B >= 16) {
        const bool launched = launch_tile16_mfma_flags<scalar_t,D,GQA,APPLY_ROPE_KCACHE,IS_NEOX>(
            fi,dw_w,dw_b,cs,idx,gw_w,gw_b,qkm,temp,out,
            positions,cos_sin_cache,kv_slot_mapping,key_cache,rotary_dim,
            key_cache_block_size,key_cache_stride_block,key_cache_stride_slot,
            key_cache_stride_head,key_cache_stride_channel,
            B,G,E,nq,pad_slot_id,out_stride_row,
            first_stride_row,first_stride_channel,
            conv_stride_row,conv_stride_channel,conv_stride_token,
            qk_mean_stride_row,qk_mean_stride_group,qk_mean_stride_channel,
            sqrt_hd,has_dw_b,has_gw_b,clamp_temp,fold,s);
        if (cca_debug_launch_should_log()) {
            std::fprintf(stderr,
                "[cca_decode_norm_fused] B=%d G=%d E=%d D=%d GQA=%d "
                "rope_kcache=%d neox=%d tile16_env=1 tile16_launched=%d "
                "dw_bias=%d gw_bias=%d clamp=%d fold_qk_mean=%d\n",
                B, G, E, D, GQA, (int)APPLY_ROPE_KCACHE, (int)IS_NEOX,
                (int)launched, (int)has_dw_b, (int)has_gw_b,
                (int)clamp_temp, (int)fold);
        }
        if (launched) return;
    }
    if (cca_debug_launch_should_log()) {
        std::fprintf(stderr,
            "[cca_decode_norm_fused] B=%d G=%d E=%d D=%d GQA=%d "
            "rope_kcache=%d neox=%d tile16_env=%d tile16_launched=0 "
            "dw_bias=%d gw_bias=%d clamp=%d fold_qk_mean=%d\n",
            B, G, E, D, GQA, (int)APPLY_ROPE_KCACHE, (int)IS_NEOX,
            (int)use_tile16_mfma, (int)has_dw_b, (int)has_gw_b,
            (int)clamp_temp, (int)fold);
    }

    const dim3 grid(B, G);
    const dim3 block(D);
    #define LAUNCH(DW,GW,CT,FQ) \
        cca_decode_fused_kernel<scalar_t,D,GQA,DW,GW,CT,FQ,APPLY_ROPE_KCACHE,IS_NEOX><<<grid,block,0,s>>>( \
            fi,dw_w,dw_b,cs,idx,gw_w,gw_b,qkm,temp,out, \
            positions,cos_sin_cache,kv_slot_mapping,key_cache,rotary_dim, \
            key_cache_block_size,key_cache_stride_block,key_cache_stride_slot, \
            key_cache_stride_head,key_cache_stride_channel, \
            B,G,E,nq,pad_slot_id, \
            first_stride_row,first_stride_channel,conv_stride_row,conv_stride_channel, \
            conv_stride_token,qk_mean_stride_row,qk_mean_stride_group, \
            qk_mean_stride_channel,out_stride_row,sqrt_hd)
    const int sel=(has_dw_b?8:0)|(has_gw_b?4:0)|(clamp_temp?2:0)|(fold?1:0);
    switch (sel) {
        case 0:  LAUNCH(false,false,false,false); break;
        case 1:  LAUNCH(false,false,false,true ); break;
        case 2:  LAUNCH(false,false,true ,false); break;
        case 3:  LAUNCH(false,false,true ,true ); break;
        case 4:  LAUNCH(false,true ,false,false); break;
        case 5:  LAUNCH(false,true ,false,true ); break;
        case 6:  LAUNCH(false,true ,true ,false); break;
        case 7:  LAUNCH(false,true ,true ,true ); break;
        case 8:  LAUNCH(true ,false,false,false); break;
        case 9:  LAUNCH(true ,false,false,true ); break;
        case 10: LAUNCH(true ,false,true ,false); break;
        case 11: LAUNCH(true ,false,true ,true ); break;
        case 12: LAUNCH(true ,true ,false,false); break;
        case 13: LAUNCH(true ,true ,false,true ); break;
        case 14: LAUNCH(true ,true ,true ,false); break;
        case 15: LAUNCH(true ,true ,true ,true ); break;
    }
    #undef LAUNCH
}

torch::Tensor cca_decode_fused(
        torch::Tensor first_input, torch::Tensor dw_weight,
        torch::optional<torch::Tensor> dw_bias, torch::Tensor conv_states,
        torch::Tensor state_indices, torch::Tensor gw_weight,
        torch::optional<torch::Tensor> gw_bias, torch::optional<torch::Tensor> qk_mean,
        torch::Tensor temp, int64_t num_q_heads, int64_t head_dim, int64_t gqa_groups,
        int64_t pad_slot_id, double sqrt_head_dim, bool clamp_temp, bool fold_qk_mean,
        torch::optional<torch::Tensor> out_opt)
{
    if (first_input.dim() == 3) first_input = first_input.squeeze(-1);
    TORCH_CHECK(first_input.dim() == 2, "first_input must be [B, E]");
    TORCH_CHECK(conv_states.size(2) == 2, "kernel specialized for total_padding==2");
    if (!fold_qk_mean) TORCH_CHECK(qk_mean.has_value(), "no-fold path requires qk_mean");

    const int B  = first_input.size(0);
    const int E  = first_input.size(1);
    const int G  = gw_weight.size(0);
    const int nq = (int)num_q_heads;
    TORCH_CHECK((long)head_dim * G == E, "E must equal G*head_dim");
    TORCH_CHECK(first_input.stride(1) == 1,
                "first_input innermost dim must be contiguous");
    if (!fold_qk_mean) {
        TORCH_CHECK(qk_mean->dim() == 3, "qk_mean must be [B, G, D]");
        TORCH_CHECK(qk_mean->size(0) >= B && qk_mean->size(1) == G
                    && qk_mean->size(2) == head_dim,
                    "qk_mean shape mismatch");
        TORCH_CHECK(qk_mean->stride(2) == 1,
                    "qk_mean innermost dim must be contiguous");
        TORCH_CHECK(qk_mean->scalar_type() == first_input.scalar_type(),
                    "qk_mean dtype must match first_input");
        TORCH_CHECK(qk_mean->device() == first_input.device(),
                    "qk_mean device must match first_input");
    }

    auto idx_i32 = state_indices.to(torch::kInt32).contiguous();
    auto temp_f32 = temp.to(torch::kFloat32).contiguous().view({-1});
    const int64_t first_stride_row = first_input.stride(0);
    const int64_t first_stride_channel = first_input.stride(1);
    const int64_t conv_stride_row = conv_states.stride(0);
    const int64_t conv_stride_channel = conv_states.stride(1);
    const int64_t conv_stride_token = conv_states.stride(2);
    const int64_t qk_mean_stride_row =
        (!fold_qk_mean && qk_mean.has_value()) ? qk_mean->stride(0) : 0;
    const int64_t qk_mean_stride_group =
        (!fold_qk_mean && qk_mean.has_value()) ? qk_mean->stride(1) : 0;
    const int64_t qk_mean_stride_channel =
        (!fold_qk_mean && qk_mean.has_value()) ? qk_mean->stride(2) : 0;

    torch::Tensor out;
    int64_t out_stride_row;
    if (out_opt.has_value() && out_opt->defined()) {
        out = *out_opt;
        TORCH_CHECK(out.dim() == 2, "out_opt must be 2-D");
        TORCH_CHECK(out.size(0) >= B, "out_opt rows < B");
        TORCH_CHECK(out.size(1) >= E, "out_opt cols < E");
        TORCH_CHECK(out.stride(1) == 1, "out_opt innermost dim must be contiguous");
        TORCH_CHECK(out.scalar_type() == first_input.scalar_type(),
                    "out_opt dtype must match first_input");
        TORCH_CHECK(out.device() == first_input.device(),
                    "out_opt device must match first_input");
        out_stride_row = out.stride(0);
    } else {
        out = torch::empty({B, E}, first_input.options());
        out_stride_row = E;
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    const bool use_tile16_mfma = cca_env_enabled(
        "VLLM_CCA_DECODE_NORM_TILE16_MFMA_ENABLED", false);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16,
        first_input.scalar_type(), "cca_decode_fused", [&]{
        const scalar_t* dwb = dw_bias.has_value() ? dw_bias->data_ptr<scalar_t>() : nullptr;
        const scalar_t* gwb = gw_bias.has_value() ? gw_bias->data_ptr<scalar_t>() : nullptr;
        const scalar_t* qkm = qk_mean.has_value() ? qk_mean->data_ptr<scalar_t>() : nullptr;

        bool dispatched = false;
        auto try_case = [&](int dd, int gg, auto D_c, auto GQA_c) {
            if (!dispatched && head_dim == dd && gqa_groups == gg) {
                launch_flags<scalar_t, decltype(D_c)::value,
                             decltype(GQA_c)::value, false, true>(
                    first_input.data_ptr<scalar_t>(), dw_weight.data_ptr<scalar_t>(), dwb,
                    conv_states.data_ptr<scalar_t>(), idx_i32.data_ptr<int>(),
                    gw_weight.data_ptr<scalar_t>(), gwb, qkm, temp_f32.data_ptr<float>(),
                    out.data_ptr<scalar_t>(), nullptr, nullptr, nullptr, nullptr,
                    0, 0, 0, 0, 0, 0, B, G, E, nq, (int)pad_slot_id,
                    out_stride_row,
                    first_stride_row, first_stride_channel,
                    conv_stride_row, conv_stride_channel, conv_stride_token,
                    qk_mean_stride_row, qk_mean_stride_group, qk_mean_stride_channel,
                    (float)sqrt_head_dim, dw_bias.has_value(), gw_bias.has_value(),
                    clamp_temp, fold_qk_mean, stream, use_tile16_mfma);
                dispatched = true;
            }
        };

        using ic = std::integral_constant<int, 0>;  // not used; keeps include obvious
        (void)sizeof(ic);

        try_case(128, 1, std::integral_constant<int,128>{}, std::integral_constant<int,1>{});
        try_case(128, 2, std::integral_constant<int,128>{}, std::integral_constant<int,2>{});
        try_case(128, 4, std::integral_constant<int,128>{}, std::integral_constant<int,4>{});
        try_case(128, 8, std::integral_constant<int,128>{}, std::integral_constant<int,8>{});
        try_case(64,  1, std::integral_constant<int,64>{},  std::integral_constant<int,1>{});
        try_case(64,  2, std::integral_constant<int,64>{},  std::integral_constant<int,2>{});
        try_case(64,  4, std::integral_constant<int,64>{},  std::integral_constant<int,4>{});
        try_case(64,  8, std::integral_constant<int,64>{},  std::integral_constant<int,8>{});

        TORCH_CHECK(dispatched, "no dispatch case for head_dim=", head_dim,
                    " gqa_groups=", gqa_groups);
    });
    return out;
}

torch::Tensor cca_decode_fused_rope_kcache(
        torch::Tensor first_input, torch::Tensor dw_weight,
        torch::optional<torch::Tensor> dw_bias, torch::Tensor conv_states,
        torch::Tensor state_indices, torch::Tensor gw_weight,
        torch::optional<torch::Tensor> gw_bias, torch::optional<torch::Tensor> qk_mean,
        torch::Tensor temp, int64_t num_q_heads, int64_t head_dim, int64_t gqa_groups,
        int64_t pad_slot_id, double sqrt_head_dim, bool clamp_temp, bool fold_qk_mean,
        torch::Tensor positions, torch::Tensor cos_sin_cache,
        torch::Tensor kv_slot_mapping, torch::Tensor key_cache,
        int64_t rotary_dim, bool is_neox,
        torch::optional<torch::Tensor> out_opt)
{
    if (first_input.dim() == 3) first_input = first_input.squeeze(-1);
    TORCH_CHECK(first_input.dim() == 2, "first_input must be [B, E]");
    TORCH_CHECK(conv_states.size(2) == 2, "kernel specialized for total_padding==2");
    if (!fold_qk_mean) TORCH_CHECK(qk_mean.has_value(), "no-fold path requires qk_mean");

    const int B  = first_input.size(0);
    const int E  = first_input.size(1);
    const int G  = gw_weight.size(0);
    const int nq = (int)num_q_heads;
    const int num_k_heads = G - nq;
    TORCH_CHECK(num_k_heads > 0, "expected at least one key head");
    TORCH_CHECK((long)head_dim * G == E, "E must equal G*head_dim");
    TORCH_CHECK(rotary_dim >= 0 && rotary_dim <= head_dim && (rotary_dim % 2) == 0,
                "rotary_dim must be even and <= head_dim");
    TORCH_CHECK(first_input.stride(1) == 1,
                "first_input innermost dim must be contiguous");
    TORCH_CHECK(positions.dim() == 1 && positions.size(0) >= B,
                "positions must be [B] or larger");
    TORCH_CHECK(positions.scalar_type() == torch::kInt64,
                "positions must be int64");
    TORCH_CHECK(kv_slot_mapping.dim() == 1 && kv_slot_mapping.size(0) >= B,
                "kv_slot_mapping must be [B] or larger");
    TORCH_CHECK(kv_slot_mapping.scalar_type() == torch::kInt64,
                "kv_slot_mapping must be int64");
    TORCH_CHECK(cos_sin_cache.dim() == 2 && cos_sin_cache.size(1) == rotary_dim,
                "cos_sin_cache must be [max_position, rotary_dim]");
    TORCH_CHECK(cos_sin_cache.scalar_type() == first_input.scalar_type(),
                "cos_sin_cache dtype must match first_input");
    TORCH_CHECK(cos_sin_cache.device() == first_input.device(),
                "cos_sin_cache device must match first_input");
    TORCH_CHECK(cos_sin_cache.stride(1) == 1,
                "cos_sin_cache innermost dim must be contiguous");
    TORCH_CHECK(key_cache.dim() == 4, "key_cache must be [blocks, block, heads, D]");
    TORCH_CHECK(key_cache.size(2) == num_k_heads && key_cache.size(3) == head_dim,
                "key_cache shape mismatch");
    TORCH_CHECK(key_cache.scalar_type() == first_input.scalar_type(),
                "key_cache dtype must match first_input");
    TORCH_CHECK(key_cache.device() == first_input.device(),
                "key_cache device must match first_input");
    if (!fold_qk_mean) {
        TORCH_CHECK(qk_mean->dim() == 3, "qk_mean must be [B, G, D]");
        TORCH_CHECK(qk_mean->size(0) >= B && qk_mean->size(1) == G
                    && qk_mean->size(2) == head_dim,
                    "qk_mean shape mismatch");
        TORCH_CHECK(qk_mean->stride(2) == 1,
                    "qk_mean innermost dim must be contiguous");
        TORCH_CHECK(qk_mean->scalar_type() == first_input.scalar_type(),
                    "qk_mean dtype must match first_input");
        TORCH_CHECK(qk_mean->device() == first_input.device(),
                    "qk_mean device must match first_input");
    }

    auto idx_i32 = state_indices.to(torch::kInt32).contiguous();
    auto temp_f32 = temp.to(torch::kFloat32).contiguous().view({-1});
    auto positions_i64 = positions.to(torch::kInt64).contiguous();
    auto slots_i64 = kv_slot_mapping.to(torch::kInt64).contiguous();
    const int64_t first_stride_row = first_input.stride(0);
    const int64_t first_stride_channel = first_input.stride(1);
    const int64_t conv_stride_row = conv_states.stride(0);
    const int64_t conv_stride_channel = conv_states.stride(1);
    const int64_t conv_stride_token = conv_states.stride(2);
    const int64_t qk_mean_stride_row =
        (!fold_qk_mean && qk_mean.has_value()) ? qk_mean->stride(0) : 0;
    const int64_t qk_mean_stride_group =
        (!fold_qk_mean && qk_mean.has_value()) ? qk_mean->stride(1) : 0;
    const int64_t qk_mean_stride_channel =
        (!fold_qk_mean && qk_mean.has_value()) ? qk_mean->stride(2) : 0;

    torch::Tensor out;
    int64_t out_stride_row;
    if (out_opt.has_value() && out_opt->defined()) {
        out = *out_opt;
        TORCH_CHECK(out.dim() == 2, "out_opt must be 2-D");
        TORCH_CHECK(out.size(0) >= B, "out_opt rows < B");
        TORCH_CHECK(out.size(1) >= E, "out_opt cols < E");
        TORCH_CHECK(out.stride(1) == 1, "out_opt innermost dim must be contiguous");
        TORCH_CHECK(out.scalar_type() == first_input.scalar_type(),
                    "out_opt dtype must match first_input");
        TORCH_CHECK(out.device() == first_input.device(),
                    "out_opt device must match first_input");
        out_stride_row = out.stride(0);
    } else {
        out = torch::empty({B, E}, first_input.options());
        out_stride_row = E;
    }

    auto stream = at::cuda::getCurrentCUDAStream();
    const bool use_tile16_mfma = cca_env_enabled(
        "VLLM_CCA_DECODE_NORM_TILE16_MFMA_ENABLED", false);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16,
        first_input.scalar_type(), "cca_decode_fused_rope_kcache", [&]{
        const scalar_t* dwb = dw_bias.has_value() ? dw_bias->data_ptr<scalar_t>() : nullptr;
        const scalar_t* gwb = gw_bias.has_value() ? gw_bias->data_ptr<scalar_t>() : nullptr;
        const scalar_t* qkm = qk_mean.has_value() ? qk_mean->data_ptr<scalar_t>() : nullptr;

        bool dispatched = false;
        auto try_case = [&](int dd, int gg, auto D_c, auto GQA_c) {
            if (!dispatched && head_dim == dd && gqa_groups == gg) {
                if (is_neox) {
                    launch_flags<scalar_t, decltype(D_c)::value,
                                 decltype(GQA_c)::value, true, true>(
                        first_input.data_ptr<scalar_t>(), dw_weight.data_ptr<scalar_t>(), dwb,
                        conv_states.data_ptr<scalar_t>(), idx_i32.data_ptr<int>(),
                        gw_weight.data_ptr<scalar_t>(), gwb, qkm, temp_f32.data_ptr<float>(),
                        out.data_ptr<scalar_t>(),
                        positions_i64.data_ptr<int64_t>(),
                        cos_sin_cache.data_ptr<scalar_t>(),
                        slots_i64.data_ptr<int64_t>(),
                        key_cache.data_ptr<scalar_t>(),
                        (int)rotary_dim, (int)key_cache.size(1),
                        key_cache.stride(0), key_cache.stride(1),
                        key_cache.stride(2), key_cache.stride(3),
                        B, G, E, nq, (int)pad_slot_id, out_stride_row,
                        first_stride_row, first_stride_channel,
                        conv_stride_row, conv_stride_channel, conv_stride_token,
                        qk_mean_stride_row, qk_mean_stride_group, qk_mean_stride_channel,
                        (float)sqrt_head_dim, dw_bias.has_value(), gw_bias.has_value(),
                        clamp_temp, fold_qk_mean, stream, use_tile16_mfma);
                } else {
                    launch_flags<scalar_t, decltype(D_c)::value,
                                 decltype(GQA_c)::value, true, false>(
                        first_input.data_ptr<scalar_t>(), dw_weight.data_ptr<scalar_t>(), dwb,
                        conv_states.data_ptr<scalar_t>(), idx_i32.data_ptr<int>(),
                        gw_weight.data_ptr<scalar_t>(), gwb, qkm, temp_f32.data_ptr<float>(),
                        out.data_ptr<scalar_t>(),
                        positions_i64.data_ptr<int64_t>(),
                        cos_sin_cache.data_ptr<scalar_t>(),
                        slots_i64.data_ptr<int64_t>(),
                        key_cache.data_ptr<scalar_t>(),
                        (int)rotary_dim, (int)key_cache.size(1),
                        key_cache.stride(0), key_cache.stride(1),
                        key_cache.stride(2), key_cache.stride(3),
                        B, G, E, nq, (int)pad_slot_id, out_stride_row,
                        first_stride_row, first_stride_channel,
                        conv_stride_row, conv_stride_channel, conv_stride_token,
                        qk_mean_stride_row, qk_mean_stride_group, qk_mean_stride_channel,
                        (float)sqrt_head_dim, dw_bias.has_value(), gw_bias.has_value(),
                        clamp_temp, fold_qk_mean, stream, use_tile16_mfma);
                }
                dispatched = true;
            }
        };

        try_case(128, 1, std::integral_constant<int,128>{}, std::integral_constant<int,1>{});
        try_case(128, 2, std::integral_constant<int,128>{}, std::integral_constant<int,2>{});
        try_case(128, 4, std::integral_constant<int,128>{}, std::integral_constant<int,4>{});
        try_case(128, 8, std::integral_constant<int,128>{}, std::integral_constant<int,8>{});
        try_case(64,  1, std::integral_constant<int,64>{},  std::integral_constant<int,1>{});
        try_case(64,  2, std::integral_constant<int,64>{},  std::integral_constant<int,2>{});
        try_case(64,  4, std::integral_constant<int,64>{},  std::integral_constant<int,4>{});
        try_case(64,  8, std::integral_constant<int,64>{},  std::integral_constant<int,8>{});

        TORCH_CHECK(dispatched, "no dispatch case for head_dim=", head_dim,
                    " gqa_groups=", gqa_groups);
    });
    return out;
}

template <typename scalar_t>
__global__ void cca_cache_value_kernel(
        const scalar_t* __restrict__ value,
        scalar_t* __restrict__ value_cache,
        const int64_t* __restrict__ kv_slot_mapping,
        int B, int num_kv_heads, int head_dim, int block_size,
        long value_stride_token, long value_stride_head, long value_stride_channel,
        long cache_stride_block, long cache_stride_slot,
        long cache_stride_head, long cache_stride_channel)
{
    const int token = blockIdx.x;
    const int head = blockIdx.y;
    const int channel = threadIdx.x;
    if (token >= B || head >= num_kv_heads || channel >= head_dim) return;
    const int64_t slot = kv_slot_mapping[token];
    if (slot < 0) return;
    const int64_t block_idx = slot / block_size;
    const int64_t block_offset = slot % block_size;
    value_cache[block_idx * cache_stride_block
                + block_offset * cache_stride_slot
                + (long)head * cache_stride_head
                + (long)channel * cache_stride_channel] =
        value[(long)token * value_stride_token
              + (long)head * value_stride_head
              + (long)channel * value_stride_channel];
}

void cca_cache_value(
        torch::Tensor value, torch::Tensor kv_slot_mapping,
        torch::Tensor value_cache)
{
    TORCH_CHECK(value.dim() == 3, "value must be [B, num_kv_heads, head_dim]");
    TORCH_CHECK(value_cache.dim() == 4,
                "value_cache must be [blocks, block, num_kv_heads, head_dim]");
    TORCH_CHECK(kv_slot_mapping.dim() == 1
                && kv_slot_mapping.size(0) >= value.size(0),
                "kv_slot_mapping must be [B] or larger");
    TORCH_CHECK(kv_slot_mapping.scalar_type() == torch::kInt64,
                "kv_slot_mapping must be int64");
    TORCH_CHECK(value.scalar_type() == value_cache.scalar_type(),
                "value and value_cache dtype mismatch");
    TORCH_CHECK(value.device() == value_cache.device(),
                "value and value_cache device mismatch");
    TORCH_CHECK(value_cache.size(2) == value.size(1)
                && value_cache.size(3) == value.size(2),
                "value_cache shape mismatch");

    auto slots_i64 = kv_slot_mapping.to(torch::kInt64).contiguous();
    const int B = value.size(0);
    const int num_kv_heads = value.size(1);
    const int head_dim = value.size(2);
    if (cca_debug_launch_should_log()) {
        std::fprintf(stderr,
            "[cca_cache_value] B=%d kv_heads=%d head_dim=%d block_size=%ld\n",
            B, num_kv_heads, head_dim, (long)value_cache.size(1));
    }
    const dim3 grid(B, num_kv_heads);
    const dim3 block(head_dim);
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16,
        value.scalar_type(), "cca_cache_value", [&] {
        cca_cache_value_kernel<scalar_t><<<grid, block, 0, stream>>>(
            value.data_ptr<scalar_t>(),
            value_cache.data_ptr<scalar_t>(),
            slots_i64.data_ptr<int64_t>(),
            B, num_kv_heads, head_dim, (int)value_cache.size(1),
            value.stride(0), value.stride(1), value.stride(2),
            value_cache.stride(0), value_cache.stride(1),
            value_cache.stride(2), value_cache.stride(3));
    });
}

// Zaya production variant: BF16 projections/weights with the FP32 recurrent
// CCA state preserved.  The generic kernel above requires one scalar type for
// both the projection and state tensors, which is not the production policy.
__global__ void cca_decode_fused_mixed_state_kernel(
        const at::BFloat16* first_input,
        const at::BFloat16* dw_weight,
        const at::BFloat16* dw_bias,
        float* conv_states,
        const int* state_indices,
        const at::BFloat16* gw_weight,
        const at::BFloat16* gw_bias,
        const float* temp,
        at::BFloat16* out,
        int B, int G, int E, int num_q_heads, int gqa_groups,
        int pad_slot_id, long first_stride_row, long first_stride_channel,
        long conv_stride_row, long conv_stride_channel,
        long conv_stride_token, long out_stride_row, float sqrt_head_dim,
        bool has_dw_bias, bool has_gw_bias, bool clamp_temp)
{
    constexpr int D = 128;
    constexpr float NORM_EPS = 1e-12f;
    const int b = blockIdx.x;
    const int g = blockIdx.y;
    const int oc = threadIdx.x;
    const int c = g * D + oc;
    if (b >= B || g >= G || c >= E) return;

    __shared__ float y1[D];
    __shared__ float y2[D];
    __shared__ float sh_reduce[D];

    const int row = state_indices[b];
    if (row == pad_slot_id || row < 0) {
        out[(long)b * out_stride_row + c] = at::BFloat16(0);
        return;
    }

    float* state = conv_states + (long)row * conv_stride_row
        + (long)c * conv_stride_channel;
    const float x0 = state[0 * conv_stride_token];
    const float x1 = state[1 * conv_stride_token];
    const float x2 = static_cast<float>(
        first_input[(long)b * first_stride_row
                    + (long)c * first_stride_channel]);
    const float w0 = static_cast<float>(dw_weight[c * 2 + 0]);
    const float w1 = static_cast<float>(dw_weight[c * 2 + 1]);
    const float bd = has_dw_bias ? static_cast<float>(dw_bias[c]) : 0.0f;
    y1[oc] = bd + w0 * x0 + w1 * x1;
    y2[oc] = bd + w0 * x1 + w1 * x2;
    state[0 * conv_stride_token] = x1;
    state[1 * conv_stride_token] = x2;
    __syncthreads();

    const at::BFloat16* gwg = gw_weight + (long)g * (D * 2) * D;
    float acc = has_gw_bias ? static_cast<float>(gw_bias[g * D + oc]) : 0.0f;
    for (int ic = 0; ic < D; ++ic) {
        acc += static_cast<float>(gwg[(ic * 2 + 0) * D + oc]) * y1[ic];
        acc += static_cast<float>(gwg[(ic * 2 + 1) * D + oc]) * y2[ic];
    }

    const at::BFloat16* fi = first_input + (long)b * first_stride_row;
    if (g < num_q_heads) {
        const int kc = num_q_heads * D + (g / gqa_groups) * D + oc;
        acc += 0.5f * (x2 + static_cast<float>(
            fi[(long)kc * first_stride_channel]));
    } else {
        const int k = g - num_q_heads;
        float qsum = 0.0f;
        for (int j = 0; j < gqa_groups; ++j) {
            qsum += static_cast<float>(
                fi[(long)((k * gqa_groups + j) * D + oc)
                   * first_stride_channel]);
        }
        acc += 0.5f * (qsum / gqa_groups) + 0.5f * x2;
    }

    sh_reduce[oc] = acc * acc;
    __syncthreads();
    for (int s = D >> 1; s > 0; s >>= 1) {
        if (oc < s) sh_reduce[oc] += sh_reduce[oc + s];
        __syncthreads();
    }
    const float inv = rsqrtf(sh_reduce[0] + NORM_EPS);
    float scale = sqrt_head_dim;
    if (g >= num_q_heads) {
        float t = temp[g - num_q_heads];
        if (clamp_temp) t = expf(fminf(fmaxf(t, 1e-7f), 2.0f));
        scale *= t;
    }
    sh_reduce[oc] = acc * inv * scale;
    __syncthreads();
    out[(long)b * out_stride_row + c] =
        static_cast<at::BFloat16>(sh_reduce[oc]);
}

torch::Tensor cca_decode_fused_mixed_state(
        torch::Tensor first_input, torch::Tensor dw_weight,
        torch::optional<torch::Tensor> dw_bias, torch::Tensor conv_states,
        torch::Tensor state_indices, torch::Tensor gw_weight,
        torch::optional<torch::Tensor> gw_bias, torch::Tensor temp,
        int64_t num_q_heads, int64_t head_dim, int64_t gqa_groups,
        int64_t pad_slot_id, double sqrt_head_dim, bool clamp_temp,
        torch::optional<torch::Tensor> out_opt)
{
    TORCH_CHECK(first_input.scalar_type() == torch::kBFloat16
                && conv_states.scalar_type() == torch::kFloat,
                "mixed CCA state expects BF16 input and FP32 state");
    TORCH_CHECK(head_dim == 128 && gqa_groups == 4,
                "mixed CCA state is specialized for D=128/GQA=4");
    TORCH_CHECK(conv_states.size(2) == 2, "mixed CCA state expects width 2");
    first_input = first_input.contiguous();
    auto idx_i32 = state_indices.to(torch::kInt32).contiguous();
    auto temp_f32 = temp.to(torch::kFloat32).contiguous().view({-1});
    const int B = first_input.size(0);
    const int E = first_input.size(1);
    const int G = gw_weight.size(0);
    TORCH_CHECK(E == G * head_dim, "mixed CCA E/G/D mismatch");

    torch::Tensor out;
    int64_t out_stride_row;
    if (out_opt.has_value() && out_opt->defined()) {
        out = *out_opt;
        out_stride_row = out.stride(0);
    } else {
        out = torch::empty({B, E}, first_input.options());
        out_stride_row = E;
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    cca_decode_fused_mixed_state_kernel<<<dim3(B, G), dim3(128), 0, stream>>>(
        first_input.data_ptr<at::BFloat16>(),
        dw_weight.data_ptr<at::BFloat16>(),
        dw_bias.has_value() ? dw_bias->data_ptr<at::BFloat16>() : nullptr,
        conv_states.data_ptr<float>(),
        idx_i32.data_ptr<int>(),
        gw_weight.data_ptr<at::BFloat16>(),
        gw_bias.has_value() ? gw_bias->data_ptr<at::BFloat16>() : nullptr,
        temp_f32.data_ptr<float>(), out.data_ptr<at::BFloat16>(),
        B, G, E, (int)num_q_heads, (int)gqa_groups, (int)pad_slot_id,
        first_input.stride(0), first_input.stride(1),
        conv_states.stride(0), conv_states.stride(1), conv_states.stride(2),
        out_stride_row, (float)sqrt_head_dim, dw_bias.has_value(),
        gw_bias.has_value(), clamp_temp);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cca_decode_fused", &cca_decode_fused, "Fused CCA decode (both qk_mean variants)",
          py::arg("first_input"), py::arg("dw_weight"), py::arg("dw_bias"),
          py::arg("conv_states"), py::arg("state_indices"), py::arg("gw_weight"),
          py::arg("gw_bias"), py::arg("qk_mean"), py::arg("temp"),
          py::arg("num_q_heads"), py::arg("head_dim"), py::arg("gqa_groups"),
          py::arg("pad_slot_id"), py::arg("sqrt_head_dim"),
          py::arg("clamp_temp"), py::arg("fold_qk_mean"),
          py::arg("out_opt"));
    m.def("cca_decode_fused_rope_kcache", &cca_decode_fused_rope_kcache,
          "Fused CCA decode with RoPE and direct K-cache write",
          py::arg("first_input"), py::arg("dw_weight"), py::arg("dw_bias"),
          py::arg("conv_states"), py::arg("state_indices"), py::arg("gw_weight"),
          py::arg("gw_bias"), py::arg("qk_mean"), py::arg("temp"),
          py::arg("num_q_heads"), py::arg("head_dim"), py::arg("gqa_groups"),
          py::arg("pad_slot_id"), py::arg("sqrt_head_dim"),
          py::arg("clamp_temp"), py::arg("fold_qk_mean"),
          py::arg("positions"), py::arg("cos_sin_cache"),
          py::arg("kv_slot_mapping"), py::arg("key_cache"),
          py::arg("rotary_dim"), py::arg("is_neox"), py::arg("out_opt"));
    m.def("cca_cache_value", &cca_cache_value, "Direct CCA V-cache write",
          py::arg("value"), py::arg("kv_slot_mapping"),
          py::arg("value_cache"));
    m.def("cca_decode_fused_mixed_state", &cca_decode_fused_mixed_state,
          "Fused CCA decode with BF16 projections and FP32 state",
          py::arg("first_input"), py::arg("dw_weight"),
          py::arg("dw_bias"), py::arg("conv_states"),
          py::arg("state_indices"), py::arg("gw_weight"),
          py::arg("gw_bias"), py::arg("temp"),
          py::arg("num_q_heads"), py::arg("head_dim"),
          py::arg("gqa_groups"), py::arg("pad_slot_id"),
          py::arg("sqrt_head_dim"), py::arg("clamp_temp"),
          py::arg("out_opt"));
}
