// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>
#include <torch/all.h>

#include <cstdint>
#include <limits>
#include <string>

namespace {

constexpr int kNumQueryHeads = 8;
constexpr int kNumKeyHeads = 2;
constexpr int kHeadDim = 128;
constexpr int kGqaGroups = 4;
constexpr int kNumHeads = kNumQueryHeads + kNumKeyHeads;
constexpr int kWidth = 1280;
constexpr float kNormEps = 1e-12f;
constexpr float kSqrtHeadDim = 11.313708305358887f;

static_assert(kNumQueryHeads == kNumKeyHeads * kGqaGroups);
static_assert(kWidth == kNumHeads * kHeadDim);

__global__ void cca_qk_postprocess_kernel(
    const at::BFloat16* grouped, const at::BFloat16* first,
    const at::BFloat16* temp, at::BFloat16* out, int rows,
    int64_t grouped_stride, int64_t first_stride, int64_t out_stride) {
  const int row = blockIdx.x;
  const int head = blockIdx.y;
  const int channel = threadIdx.x;
  if (row >= rows || head >= kNumHeads || channel >= kHeadDim) {
    return;
  }

  __shared__ float squared[kHeadDim];
  const bool is_query = head < kNumQueryHeads;
  const int64_t row_offset = static_cast<int64_t>(row);
  float value = static_cast<float>(
      grouped[row_offset * grouped_stride + head * kHeadDim + channel]);
  const at::BFloat16* input = first + row_offset * first_stride;

  if (is_query) {
    const int key_channel =
        kNumQueryHeads * kHeadDim + (head / kGqaGroups) * kHeadDim + channel;
    value += 0.5f * static_cast<float>(input[head * kHeadDim + channel]);
    value += 0.5f * static_cast<float>(input[key_channel]);
  } else {
    const int key_head = head - kNumQueryHeads;
    float query_sum = 0.0f;
#pragma unroll
    for (int query = 0; query < kGqaGroups; ++query) {
      query_sum += static_cast<float>(
          input[(key_head * kGqaGroups + query) * kHeadDim + channel]);
    }
    value += 0.5f * (query_sum / static_cast<float>(kGqaGroups));
    value += 0.5f * static_cast<float>(input[head * kHeadDim + channel]);
  }

  squared[channel] = value * value;
  __syncthreads();

  // Match ATen on gfx942: reduce each wave64 before combining the waves.
  const int wave_base = (channel / 64) * 64;
  const int lane = channel % 64;
  for (int stride = 32; stride > 0; stride /= 2) {
    if (lane < stride) {
      squared[wave_base + lane] += squared[wave_base + lane + stride];
    }
    __syncthreads();
  }
  if (channel == 0) {
    squared[0] += squared[64];
  }
  __syncthreads();

  // Preserve vector_norm -> square -> rsqrt and each subsequent multiply.
  volatile float norm = sqrtf(squared[0]);
  const float inverse_norm = rsqrtf(norm * norm + kNormEps);
  volatile float normalized = value * inverse_norm;
  normalized = normalized * kSqrtHeadDim;
  if (!is_query) {
    const float key_temp = static_cast<float>(temp[head - kNumQueryHeads]);
    normalized = normalized * key_temp;
  }
  out[row_offset * out_stride + head * kHeadDim + channel] =
      static_cast<at::BFloat16>(normalized);
}

void check_matrix_layout(const torch::Tensor& tensor, int64_t rows,
                         const char* name) {
  TORCH_CHECK(
      tensor.dim() == 2 && tensor.size(0) == rows && tensor.size(1) == kWidth,
      "cca_qk_postprocess: ", name, " must have shape [M, ", kWidth, "]");
  TORCH_CHECK(tensor.stride(1) == 1 && tensor.stride(0) >= kWidth,
              "cca_qk_postprocess: ", name,
              " must have unit inner stride and non-overlapping rows");
}

}  // namespace

void cca_qk_postprocess(const torch::Tensor& grouped,
                        const torch::Tensor& first, const torch::Tensor& temp,
                        torch::Tensor& out) {
  TORCH_CHECK(
      grouped.is_cuda() && first.is_cuda() && temp.is_cuda() && out.is_cuda(),
      "cca_qk_postprocess: all tensors must be CUDA/HIP tensors");
  TORCH_CHECK(grouped.device() == first.device() &&
                  grouped.device() == temp.device() &&
                  grouped.device() == out.device(),
              "cca_qk_postprocess: all tensors must be on the same device");
  TORCH_CHECK(grouped.scalar_type() == torch::kBFloat16 &&
                  first.scalar_type() == torch::kBFloat16 &&
                  temp.scalar_type() == torch::kBFloat16 &&
                  out.scalar_type() == torch::kBFloat16,
              "cca_qk_postprocess: all tensors must be bfloat16");
  TORCH_CHECK(!out.is_alias_of(grouped) && !out.is_alias_of(first) &&
                  !out.is_alias_of(temp),
              "cca_qk_postprocess: out must not share storage with inputs");
  TORCH_CHECK(grouped.dim() == 2,
              "cca_qk_postprocess: grouped must have shape [M, ", kWidth, "]");

  const int64_t rows = grouped.size(0);
  TORCH_CHECK(rows > 0, "cca_qk_postprocess: M must be positive");
  TORCH_CHECK(rows <= std::numeric_limits<int>::max(),
              "cca_qk_postprocess: M exceeds the kernel index range");
  check_matrix_layout(grouped, rows, "grouped");
  check_matrix_layout(first, rows, "first");
  check_matrix_layout(out, rows, "out");
  TORCH_CHECK(temp.dim() == 1 && temp.size(0) == kNumKeyHeads,
              "cca_qk_postprocess: temp must have shape [", kNumKeyHeads, "]");
  TORCH_CHECK(temp.stride(0) == 1,
              "cca_qk_postprocess: temp must have unit stride");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(grouped));
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  const std::string architecture = properties->gcnArchName;
  TORCH_CHECK(architecture.rfind("gfx942", 0) == 0,
              "cca_qk_postprocess: requires gfx942, got ", architecture);
  TORCH_CHECK(rows <= static_cast<int64_t>(properties->maxGridSize[0]),
              "cca_qk_postprocess: M exceeds the device grid-x limit");

  const auto stream = at::cuda::getCurrentCUDAStream();
  cca_qk_postprocess_kernel<<<dim3(static_cast<uint32_t>(rows), kNumHeads),
                              dim3(kHeadDim), 0, stream>>>(
      grouped.data_ptr<at::BFloat16>(), first.data_ptr<at::BFloat16>(),
      temp.data_ptr<at::BFloat16>(), out.data_ptr<at::BFloat16>(),
      static_cast<int>(rows), grouped.stride(0), first.stride(0),
      out.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
