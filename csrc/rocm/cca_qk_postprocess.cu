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
constexpr int kValuesPerLane = 4;
constexpr int kReductionLanes = kHeadDim / kValuesPerLane;
constexpr float kNormEps = 1e-12f;
constexpr float kSqrtHeadDim = 11.313708305358887f;

static_assert(kNumQueryHeads == kNumKeyHeads * kGqaGroups);
static_assert(kWidth == kNumHeads * kHeadDim);
static_assert(kReductionLanes == 32);

__global__ void cca_qk_postprocess_kernel(
    const at::BFloat16* grouped, const at::BFloat16* first,
    const at::BFloat16* temp, at::BFloat16* out, int rows,
    int64_t grouped_stride, int64_t first_stride, int64_t out_stride) {
  const int row = blockIdx.x;
  const int head = blockIdx.y;
  const int lane = threadIdx.x;
  if (row >= rows || head >= kNumHeads || lane >= kReductionLanes) {
    return;
  }

  const bool is_query = head < kNumQueryHeads;
  const int64_t row_offset = static_cast<int64_t>(row);
  const at::BFloat16* input = first + row_offset * first_stride;
  const int channel_base = lane * kValuesPerLane;
  float values[kValuesPerLane];

#pragma unroll
  for (int i = 0; i < kValuesPerLane; ++i) {
    const int channel = channel_base + i;
    float value = static_cast<float>(
        grouped[row_offset * grouped_stride + head * kHeadDim + channel]);
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
    // The source mean path materializes FP32 before vector_norm. Keep the
    // merged value in a VGPR while preventing contraction across that boundary.
    asm volatile("" : "+v"(value));
    values[i] = value;
  }

  // Match the vectorized ROCm vector_norm path: contiguous float4 reduction
  // within each lane, then ascending shuffle offsets across 32 logical lanes.
  float value_list[kValuesPerLane];
#pragma unroll
  for (int i = 0; i < kValuesPerLane; ++i) {
    value_list[i] = 0.0f;
    value_list[i] = value_list[i] + values[i] * values[i];
  }
#pragma unroll
  for (int i = 1; i < kValuesPerLane; ++i) {
    value_list[0] = value_list[0] + value_list[i];
  }
  float sum = value_list[0];
#pragma unroll
  for (int offset = 1; offset < kReductionLanes; offset <<= 1) {
    const float other = __shfl_down(sum, offset);
    sum = sum + other;
  }
  sum = __shfl(sum, 0);

  volatile float norm = sqrtf(sum);
  volatile float norm_squared = norm * norm;
  volatile float stabilized = norm_squared + kNormEps;
  volatile float inverse_norm = ::rsqrt(stabilized);
  const float key_temp =
      is_query ? 1.0f : static_cast<float>(temp[head - kNumQueryHeads]);
#pragma unroll
  for (int i = 0; i < kValuesPerLane; ++i) {
    volatile float normalized = values[i] * inverse_norm;
    normalized = normalized * kSqrtHeadDim;
    if (!is_query) {
      normalized = normalized * key_temp;
    }
    out[row_offset * out_stride + head * kHeadDim + channel_base + i] =
        static_cast<at::BFloat16>(normalized);
  }
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
                              dim3(kReductionLanes), 0, stream>>>(
      grouped.data_ptr<at::BFloat16>(), first.data_ptr<at::BFloat16>(),
      temp.data_ptr<at::BFloat16>(), out.data_ptr<at::BFloat16>(),
      static_cast<int>(rows), grouped.stride(0), first.stride(0),
      out.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
