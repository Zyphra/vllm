// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>
#include <torch/all.h>

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <string>
#include <utility>

namespace {

constexpr int kQkWidth = 1280;
constexpr int kValueWidth = 128;
constexpr int kOutputWidth = 1536;
constexpr int kConvStateWidth = 2;
constexpr int kConvWindowWidth = 3;
constexpr int kThreads = 256;
constexpr int kPadSlotId = -1;

__device__ __forceinline__ int checked_slot(const int slot,
                                            const int num_slots) {
  if (slot < kPadSlotId || slot >= num_slots) {
    __builtin_trap();
  }
  return slot;
}

__global__ void cca_decode_state_prepare_kernel(
    const at::BFloat16* qk0, const at::BFloat16* v_current,
    const uint32_t* conv_state, const float* recurrent_state,
    const int32_t* state_idx, at::BFloat16* conv_window, uint32_t* conv_tail,
    at::BFloat16* qkv_out, int rows, int num_slots, int64_t qk0_stride,
    int64_t v_current_stride, int64_t conv_state_stride0,
    int64_t conv_state_stride1, int64_t recurrent_state_stride,
    int64_t state_idx_stride, int64_t conv_tail_stride,
    int64_t qkv_out_stride) {
  const int row = blockIdx.x;
  const int channel = blockIdx.y * blockDim.x + threadIdx.x;
  if (row >= rows || channel >= kQkWidth) {
    return;
  }

  const int slot = checked_slot(
      state_idx[static_cast<int64_t>(row) * state_idx_stride], num_slots);
  const bool is_pad = slot == kPadSlotId;
  const int64_t row_offset = static_cast<int64_t>(row);
  const int64_t window_offset =
      (row_offset * kQkWidth + channel) * kConvWindowWidth;
  const int64_t tail_offset = row_offset * conv_tail_stride + channel;

  if (is_pad) {
    const at::BFloat16 zero = static_cast<at::BFloat16>(0.0f);
    conv_window[window_offset] = zero;
    conv_window[window_offset + 1] = zero;
    conv_window[window_offset + 2] = zero;
    conv_tail[tail_offset] = 0;
    if (channel < kValueWidth) {
      qkv_out[row_offset * qkv_out_stride + kQkWidth + channel] = zero;
      qkv_out[row_offset * qkv_out_stride + kQkWidth + kValueWidth + channel] =
          zero;
    }
    return;
  }

  const int64_t state_offset =
      static_cast<int64_t>(slot) * conv_state_stride0 +
      static_cast<int64_t>(channel) * conv_state_stride1;
  const uint32_t old0_bits = conv_state[state_offset];
  const uint32_t old1_bits = conv_state[state_offset + 1];
  const float old0 = __uint_as_float(old0_bits);
  const float old1 = __uint_as_float(old1_bits);
  conv_window[window_offset] = static_cast<at::BFloat16>(old0);
  conv_window[window_offset + 1] = static_cast<at::BFloat16>(old1);
  conv_window[window_offset + 2] = qk0[row_offset * qk0_stride + channel];
  conv_tail[tail_offset] = old1_bits;

  if (channel < kValueWidth) {
    qkv_out[row_offset * qkv_out_stride + kQkWidth + channel] =
        v_current[row_offset * v_current_stride + channel];
    qkv_out[row_offset * qkv_out_stride + kQkWidth + kValueWidth + channel] =
        static_cast<at::BFloat16>(recurrent_state[static_cast<int64_t>(slot) *
                                                      recurrent_state_stride +
                                                  channel]);
  }
}

__global__ void cca_decode_state_commit_kernel(
    const at::BFloat16* qk0, const at::BFloat16* v_delayed_current,
    const int32_t* state_idx, const uint32_t* conv_tail, uint32_t* conv_state,
    float* recurrent_state, at::BFloat16* qkv_out, int rows, int num_slots,
    int64_t qk0_stride, int64_t v_delayed_current_stride,
    int64_t state_idx_stride, int64_t conv_tail_stride,
    int64_t conv_state_stride0, int64_t conv_state_stride1,
    int64_t recurrent_state_stride, int64_t qkv_out_stride) {
  const int row = blockIdx.x;
  const int channel = blockIdx.y * blockDim.x + threadIdx.x;
  if (row >= rows || channel >= kOutputWidth) {
    return;
  }

  const int slot = checked_slot(
      state_idx[static_cast<int64_t>(row) * state_idx_stride], num_slots);
  const bool is_pad = slot == kPadSlotId;
  const int safe_slot = is_pad ? 0 : slot;
  const int64_t row_offset = static_cast<int64_t>(row);

  if (channel < kQkWidth) {
    const int64_t state_offset =
        static_cast<int64_t>(safe_slot) * conv_state_stride0 +
        static_cast<int64_t>(channel) * conv_state_stride1;
    if (is_pad) {
      conv_state[state_offset] = 0;
      conv_state[state_offset + 1] = 0;
    } else {
      conv_state[state_offset] =
          conv_tail[row_offset * conv_tail_stride + channel];
      conv_state[state_offset + 1] = __float_as_uint(
          static_cast<float>(qk0[row_offset * qk0_stride + channel]));
    }
  }

  if (channel < kValueWidth) {
    recurrent_state[static_cast<int64_t>(safe_slot) * recurrent_state_stride +
                    channel] =
        is_pad ? 0.0f
               : static_cast<float>(
                     v_delayed_current[row_offset * v_delayed_current_stride +
                                       channel]);
  }

  if (is_pad) {
    qkv_out[row_offset * qkv_out_stride + channel] =
        static_cast<at::BFloat16>(0.0f);
  }
}

void check_matrix(const torch::Tensor& tensor, int64_t rows, int64_t width,
                  torch::ScalarType dtype, const char* op, const char* name) {
  TORCH_CHECK(
      tensor.dim() == 2 && tensor.size(0) == rows && tensor.size(1) == width,
      op, ": ", name, " must have shape [M, ", width, "]");
  TORCH_CHECK(tensor.scalar_type() == dtype, op, ": ", name,
              " has the wrong dtype");
  TORCH_CHECK(tensor.stride(1) == 1 && tensor.stride(0) >= width, op, ": ",
              name, " must have unit inner stride and non-overlapping rows");
}

void check_conv_state(const torch::Tensor& tensor, const char* op) {
  TORCH_CHECK(tensor.dim() == 3 && tensor.size(0) > 0 &&
                  tensor.size(1) == kQkWidth &&
                  tensor.size(2) == kConvStateWidth,
              op, ": conv_state must have shape [N, ", kQkWidth, ", ",
              kConvStateWidth, "]");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, op,
              ": conv_state must be float32");
  TORCH_CHECK(tensor.stride(2) == 1 && tensor.stride(1) >= kConvStateWidth &&
                  tensor.stride(0) >= kQkWidth * tensor.stride(1),
              op,
              ": conv_state must have unit inner stride and "
              "non-overlapping channels and rows");
}

void check_recurrent_state(const torch::Tensor& tensor, int64_t num_slots,
                           const char* op) {
  TORCH_CHECK(tensor.dim() == 2 && tensor.size(0) == num_slots &&
                  tensor.size(1) == kValueWidth,
              op, ": recurrent_state must have shape [N, ", kValueWidth, "]");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, op,
              ": recurrent_state must be float32");
  TORCH_CHECK(tensor.stride(1) == 1 && tensor.stride(0) >= kValueWidth, op,
              ": recurrent_state must have unit inner stride and "
              "non-overlapping rows");
}

void check_state_idx(const torch::Tensor& tensor, int64_t rows,
                     const char* op) {
  TORCH_CHECK(tensor.dim() == 1 && tensor.size(0) == rows, op,
              ": state_idx must have shape [M]");
  TORCH_CHECK(tensor.scalar_type() == torch::kInt32, op,
              ": state_idx must be int32");
  TORCH_CHECK(tensor.stride(0) > 0, op,
              ": state_idx must have positive stride");
}

void check_conv_window(const torch::Tensor& tensor, int64_t rows,
                       const char* op) {
  TORCH_CHECK(tensor.dim() == 3 && tensor.size(0) == rows &&
                  tensor.size(1) == kQkWidth &&
                  tensor.size(2) == kConvWindowWidth,
              op, ": conv_window must have shape [M, ", kQkWidth, ", ",
              kConvWindowWidth, "]");
  TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16, op,
              ": conv_window must be bfloat16");
  TORCH_CHECK(tensor.is_contiguous(), op, ": conv_window must be contiguous");
}

void check_common_device(const torch::Tensor& anchor,
                         std::initializer_list<const torch::Tensor*> tensors,
                         const char* op) {
  TORCH_CHECK(anchor.is_cuda(), op, ": all tensors must be CUDA/HIP tensors");
  for (const auto* tensor : tensors) {
    TORCH_CHECK(tensor->is_cuda(), op,
                ": all tensors must be CUDA/HIP tensors");
    TORCH_CHECK(tensor->device() == anchor.device(), op,
                ": all tensors must be on the same device");
  }
}

void check_gfx942(const torch::Tensor& anchor, int64_t rows, const char* op) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(anchor));
  const auto* properties = at::cuda::getCurrentDeviceProperties();
  const std::string architecture = properties->gcnArchName;
  TORCH_CHECK(architecture.rfind("gfx942", 0) == 0, op,
              ": requires gfx942, got ", architecture);
  TORCH_CHECK(rows <= static_cast<int64_t>(properties->maxGridSize[0]), op,
              ": M exceeds the device grid-x limit");
}

bool byte_ranges_overlap(const torch::Tensor& left,
                         const torch::Tensor& right) {
  const auto byte_range = [](const torch::Tensor& tensor, int64_t first_dim) {
    uintptr_t begin = reinterpret_cast<uintptr_t>(tensor.data_ptr());
    uint64_t last_element_offset = 0;
    for (int64_t dim = first_dim; dim < tensor.dim(); ++dim) {
      TORCH_INTERNAL_ASSERT(tensor.size(dim) > 0);
      TORCH_INTERNAL_ASSERT(tensor.stride(dim) >= 0);
      last_element_offset += static_cast<uint64_t>(tensor.size(dim) - 1) *
                             static_cast<uint64_t>(tensor.stride(dim));
    }
    const uintptr_t end =
        begin + (last_element_offset + 1) * tensor.element_size();
    return std::pair<uintptr_t, uintptr_t>{begin, end};
  };

  const auto [left_begin, left_end] = byte_range(left, 0);
  const auto [right_begin, right_end] = byte_range(right, 0);
  if (left_begin >= right_end || right_begin >= left_end) {
    return false;
  }

  const uint64_t left_row_stride =
      static_cast<uint64_t>(left.stride(0)) * left.element_size();
  const uint64_t right_row_stride =
      static_cast<uint64_t>(right.stride(0)) * right.element_size();
  if (left.size(0) != right.size(0) ||
      left_row_stride != right_row_stride) {
    return true;
  }

  const auto [left_row_begin, left_row_end] = byte_range(left, 1);
  const auto [right_row_begin, right_row_end] = byte_range(right, 1);
  const uint64_t left_row_bytes = left_row_end - left_row_begin;
  const uint64_t right_row_bytes = right_row_end - right_row_begin;
  if (left_begin <= right_begin) {
    const uint64_t offset = right_begin - left_begin;
    return offset > left_row_stride || left_row_bytes > offset ||
           right_row_bytes > left_row_stride - offset;
  }
  const uint64_t offset = left_begin - right_begin;
  return offset > right_row_stride || right_row_bytes > offset ||
         left_row_bytes > right_row_stride - offset;
}

}  // namespace

void cca_decode_state_prepare(
    const torch::Tensor& qk0, const torch::Tensor& v_current,
    const torch::Tensor& conv_state, const torch::Tensor& recurrent_state,
    const torch::Tensor& state_idx, torch::Tensor& conv_window,
    torch::Tensor& conv_tail, torch::Tensor& qkv_out) {
  constexpr const char* op = "cca_decode_state_prepare";
  check_common_device(qk0,
                      {&v_current, &conv_state, &recurrent_state, &state_idx,
                       &conv_window, &conv_tail, &qkv_out},
                      op);
  TORCH_CHECK(qk0.dim() == 2, op, ": qk0 must have shape [M, ", kQkWidth, "]");
  const int64_t rows = qk0.size(0);
  TORCH_CHECK(rows > 0, op, ": M must be positive");
  TORCH_CHECK(rows <= std::numeric_limits<int>::max(), op,
              ": M exceeds the kernel index range");
  check_matrix(qk0, rows, kQkWidth, torch::kBFloat16, op, "qk0");
  check_matrix(v_current, rows, kValueWidth, torch::kBFloat16, op, "v_current");
  check_conv_state(conv_state, op);
  const int64_t num_slots = conv_state.size(0);
  TORCH_CHECK(num_slots <= std::numeric_limits<int>::max(), op,
              ": N exceeds the kernel index range");
  check_recurrent_state(recurrent_state, num_slots, op);
  check_state_idx(state_idx, rows, op);
  check_conv_window(conv_window, rows, op);
  check_matrix(conv_tail, rows, kQkWidth, torch::kFloat32, op, "conv_tail");
  check_matrix(qkv_out, rows, kOutputWidth, torch::kBFloat16, op, "qkv_out");

  TORCH_CHECK(
      !conv_window.is_alias_of(qk0) && !conv_window.is_alias_of(v_current) &&
          !conv_window.is_alias_of(conv_state) &&
          !conv_window.is_alias_of(recurrent_state) &&
          !conv_window.is_alias_of(state_idx) &&
          !conv_window.is_alias_of(conv_tail) &&
          !conv_window.is_alias_of(qkv_out) && !conv_tail.is_alias_of(qk0) &&
          !conv_tail.is_alias_of(v_current) &&
          !conv_tail.is_alias_of(conv_state) &&
          !conv_tail.is_alias_of(recurrent_state) &&
          !conv_tail.is_alias_of(state_idx) &&
          !conv_tail.is_alias_of(qkv_out) && !qkv_out.is_alias_of(qk0) &&
          !qkv_out.is_alias_of(v_current) && !qkv_out.is_alias_of(conv_state) &&
          !qkv_out.is_alias_of(recurrent_state) &&
          !qkv_out.is_alias_of(state_idx),
      op, ": mutable outputs must not share storage with inputs or each other");

  check_gfx942(qk0, rows, op);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(qk0));
  const auto stream = at::cuda::getCurrentCUDAStream();
  cca_decode_state_prepare_kernel<<<dim3(static_cast<uint32_t>(rows),
                                         kQkWidth / kThreads),
                                    dim3(kThreads), 0, stream>>>(
      qk0.data_ptr<at::BFloat16>(), v_current.data_ptr<at::BFloat16>(),
      reinterpret_cast<const uint32_t*>(conv_state.data_ptr<float>()),
      recurrent_state.data_ptr<float>(), state_idx.data_ptr<int32_t>(),
      conv_window.data_ptr<at::BFloat16>(),
      reinterpret_cast<uint32_t*>(conv_tail.data_ptr<float>()),
      qkv_out.data_ptr<at::BFloat16>(), static_cast<int>(rows),
      static_cast<int>(num_slots), qk0.stride(0), v_current.stride(0),
      conv_state.stride(0), conv_state.stride(1), recurrent_state.stride(0),
      state_idx.stride(0), conv_tail.stride(0), qkv_out.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void cca_decode_state_commit(const torch::Tensor& qk0,
                             const torch::Tensor& v_delayed_current,
                             const torch::Tensor& state_idx,
                             const torch::Tensor& conv_tail,
                             torch::Tensor& conv_state,
                             torch::Tensor& recurrent_state,
                             torch::Tensor& qkv_out) {
  constexpr const char* op = "cca_decode_state_commit";
  check_common_device(qk0,
                      {&v_delayed_current, &state_idx, &conv_tail, &conv_state,
                       &recurrent_state, &qkv_out},
                      op);
  TORCH_CHECK(qk0.dim() == 2, op, ": qk0 must have shape [M, ", kQkWidth, "]");
  const int64_t rows = qk0.size(0);
  TORCH_CHECK(rows > 0, op, ": M must be positive");
  TORCH_CHECK(rows <= std::numeric_limits<int>::max(), op,
              ": M exceeds the kernel index range");
  check_matrix(qk0, rows, kQkWidth, torch::kBFloat16, op, "qk0");
  check_matrix(v_delayed_current, rows, kValueWidth, torch::kBFloat16, op,
               "v_delayed_current");
  check_state_idx(state_idx, rows, op);
  check_matrix(conv_tail, rows, kQkWidth, torch::kFloat32, op, "conv_tail");
  check_conv_state(conv_state, op);
  const int64_t num_slots = conv_state.size(0);
  TORCH_CHECK(num_slots <= std::numeric_limits<int>::max(), op,
              ": N exceeds the kernel index range");
  check_recurrent_state(recurrent_state, num_slots, op);
  check_matrix(qkv_out, rows, kOutputWidth, torch::kBFloat16, op, "qkv_out");

  TORCH_CHECK(
      !conv_state.is_alias_of(qk0) &&
          !conv_state.is_alias_of(v_delayed_current) &&
          !conv_state.is_alias_of(state_idx) &&
          !conv_state.is_alias_of(conv_tail) &&
          !byte_ranges_overlap(conv_state, recurrent_state) &&
          !conv_state.is_alias_of(qkv_out) &&
          !recurrent_state.is_alias_of(qk0) &&
          !recurrent_state.is_alias_of(v_delayed_current) &&
          !recurrent_state.is_alias_of(state_idx) &&
          !recurrent_state.is_alias_of(conv_tail) &&
          !recurrent_state.is_alias_of(qkv_out) && !qkv_out.is_alias_of(qk0) &&
          !qkv_out.is_alias_of(v_delayed_current) &&
          !qkv_out.is_alias_of(state_idx) && !qkv_out.is_alias_of(conv_tail),
      op, ": mutable outputs must not share storage with inputs or each other");

  check_gfx942(qk0, rows, op);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(qk0));
  const auto stream = at::cuda::getCurrentCUDAStream();
  cca_decode_state_commit_kernel<<<dim3(static_cast<uint32_t>(rows),
                                        kOutputWidth / kThreads),
                                   dim3(kThreads), 0, stream>>>(
      qk0.data_ptr<at::BFloat16>(), v_delayed_current.data_ptr<at::BFloat16>(),
      state_idx.data_ptr<int32_t>(),
      reinterpret_cast<const uint32_t*>(conv_tail.data_ptr<float>()),
      reinterpret_cast<uint32_t*>(conv_state.data_ptr<float>()),
      recurrent_state.data_ptr<float>(), qkv_out.data_ptr<at::BFloat16>(),
      static_cast<int>(rows), static_cast<int>(num_slots), qk0.stride(0),
      v_delayed_current.stride(0), state_idx.stride(0), conv_tail.stride(0),
      conv_state.stride(0), conv_state.stride(1), recurrent_state.stride(0),
      qkv_out.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
