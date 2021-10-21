/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/util/cuda_half_util.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void TrilGpu(const int64_t elem_cnt, const int64_t num_rows, const int64_t num_cols,
                        const int64_t diagonal, const T* x, const T fill, T* y) {
  const int64_t matrix_size = num_rows * num_cols;
  CUDA_1D_KERNEL_LOOP_T(int64_t, k, elem_cnt) {
    const int64_t offset_in_matrix = k % matrix_size;
    const int64_t i = offset_in_matrix / num_cols;
    const int64_t j = offset_in_matrix - num_cols * i;
    y[k] = j > i + diagonal ? fill : x[k];
  }
}

template<typename T>
__global__ void TrilWarpProcessRowGpu(const int64_t total_rows, const int64_t num_rows,
                                      const int64_t num_cols, const int64_t diagonal, const T* x,
                                      const T fill, T* y) {
  const int64_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / kCudaWarpSize;
  const int64_t lan_id = threadIdx.x % kCudaWarpSize;
  const int64_t num_warp = blockDim.x * gridDim.x / kCudaWarpSize;
  for (int64_t i = warp_id; i < total_rows; i += num_warp) {
    const int64_t row = i % num_rows;
    for (int64_t col = lan_id; col < num_cols; col += kCudaWarpSize) {
      const int64_t idx = i * num_cols + col;
      y[idx] = col > row + diagonal ? fill : x[idx];
    }
  }
}

template<>
__global__ void TrilWarpProcessRowGpu<half>(const int64_t total_rows, const int64_t num_rows,
                                            const int64_t num_cols, const int64_t diagonal,
                                            const half* x, const half fill, half* y) {
  const int64_t h2_num_cols = num_cols / 2;
  const auto* x_h2 = reinterpret_cast<const half2*>(x);
  auto* y_h2 = reinterpret_cast<half2*>(y);

  const int64_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / kCudaWarpSize;
  const int64_t lan_id = threadIdx.x % kCudaWarpSize;
  const int64_t num_warp = blockDim.x * gridDim.x / kCudaWarpSize;
  for (int64_t i = warp_id; i < total_rows; i += num_warp) {
    const int64_t row = i % num_rows;
    for (int64_t col = lan_id; col < h2_num_cols; col += kCudaWarpSize) {
      const int64_t idx = i * h2_num_cols + col;
      const half2 x_val = x_h2[idx];
      half2 y_val;
      y_val.x = (2 * col) > row + diagonal ? fill : x_val.x;
      y_val.y = (2 * col + 1) > row + diagonal ? fill : x_val.y;
      y_h2[idx] = y_val;
    }
  }
}

template<typename T>
__global__ void FusedScaleTrilGpu(const int64_t elem_cnt, const int64_t num_rows,
                                  const int64_t num_cols, const int64_t diagonal, const T scale,
                                  const T* x, const T fill, T* y) {
  const int64_t matrix_size = num_rows * num_cols;
  CUDA_1D_KERNEL_LOOP_T(int64_t, k, elem_cnt) {
    const int64_t offset_in_matrix = k % matrix_size;
    const int64_t i = offset_in_matrix / num_cols;
    const int64_t j = offset_in_matrix - num_cols * i;
    y[k] = j > i + diagonal ? fill : (scale * x[k]);
  }
}

template<typename T>
__global__ void FusedScaleTrilWarpProcessRowGpu(const int64_t total_rows, const int64_t num_rows,
                                                const int64_t num_cols, const int64_t diagonal,
                                                const T scale, const T* x, const T fill, T* y) {
  const int64_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / kCudaWarpSize;
  const int64_t lan_id = threadIdx.x % kCudaWarpSize;
  const int64_t num_warp = blockDim.x * gridDim.x / kCudaWarpSize;
  for (int64_t i = warp_id; i < total_rows; i += num_warp) {
    const int64_t row = i % num_rows;
    for (int64_t col = lan_id; col < num_cols; col += kCudaWarpSize) {
      const int64_t idx = i * num_cols + col;
      y[idx] = col > row + diagonal ? fill : (scale * x[idx]);
    }
  }
}

template<>
__global__ void FusedScaleTrilWarpProcessRowGpu<half>(const int64_t total_rows,
                                                      const int64_t num_rows,
                                                      const int64_t num_cols,
                                                      const int64_t diagonal, const half scale,
                                                      const half* x, const half fill, half* y) {
  const int64_t h2_num_cols = num_cols / 2;
  const auto* x_h2 = reinterpret_cast<const half2*>(x);
  auto* y_h2 = reinterpret_cast<half2*>(y);
  const half2 h2_scale = __half2half2(scale);
  const int64_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / kCudaWarpSize;
  const int64_t lan_id = threadIdx.x % kCudaWarpSize;
  const int64_t num_warp = blockDim.x * gridDim.x / kCudaWarpSize;
  for (int64_t i = warp_id; i < total_rows; i += num_warp) {
    const int64_t row = i % num_rows;
    for (int64_t col = lan_id; col < h2_num_cols; col += kCudaWarpSize) {
      const int64_t idx = i * h2_num_cols + col;
      const half2 scaled_x = __hmul2(h2_scale, x_h2[idx]);
      half2 y_val;
      y_val.x = (2 * col) > row + diagonal ? fill : scaled_x.x;
      y_val.y = (2 * col + 1) > row + diagonal ? fill : scaled_x.y;
      y_h2[idx] = y_val;
    }
  }
}

template<typename T>
T GetAttrVal(bool is_floating_val, double floating_value, int64_t integer_value) {
  return is_floating_val ? static_cast<T>(floating_value) : static_cast<T>(integer_value);
}

template<>
half GetAttrVal<half>(bool is_floating_val, double floating_value, int64_t integer_value) {
  return is_floating_val ? __float2half(floating_value) : __float2half(integer_value);
}

}  // namespace

template<typename T>
class GpuTrilKernel final : public user_op::OpKernel {
 public:
  GpuTrilKernel() = default;
  ~GpuTrilKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("in", 0);
    const auto shape = x->shape();
    const auto diagonal = ctx->Attr<int64_t>("diagonal");
    const int64_t num_rows = shape.At(shape.NumAxes() - 2);
    const int64_t num_cols = shape.At(shape.NumAxes() - 1);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int32_t elem_cnt = shape.elem_cnt();
    const T fill = GetAttrVal<T>(ctx->Attr<bool>("is_floating_fill_value"),
                                 ctx->Attr<double>("floating_fill_value"),
                                 ctx->Attr<int64_t>("integer_fill_value"));
    if (num_cols % (kCudaWarpSize * 2) == 0) {
      const int64_t total_rows = elem_cnt / num_cols;
      TrilWarpProcessRowGpu<<<BlocksNum4ThreadsNum(total_rows * kCudaWarpSize),
                              kCudaThreadsNumPerBlock, 0, ctx->device_ctx()->cuda_stream()>>>(
          total_rows, num_rows, num_cols, diagonal, x->dptr<T>(), fill, y->mut_dptr<T>());
    } else {
      TrilGpu<<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                ctx->device_ctx()->cuda_stream()>>>(elem_cnt, num_rows, num_cols, diagonal,
                                                    x->dptr<T>(), fill, y->mut_dptr<T>());
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GPU_TRIL_KERNEL(dtype)                                                         \
  REGISTER_USER_KERNEL("tril")                                                                  \
      .SetCreateFn<GpuTrilKernel<dtype>>()                                                      \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                                       \
                       & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value))         \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, true));                       \
        return Maybe<void>::Ok();                                                               \
      });

REGISTER_GPU_TRIL_KERNEL(float)
REGISTER_GPU_TRIL_KERNEL(double)
REGISTER_GPU_TRIL_KERNEL(int8_t)
REGISTER_GPU_TRIL_KERNEL(int32_t)
REGISTER_GPU_TRIL_KERNEL(int64_t)
REGISTER_GPU_TRIL_KERNEL(half)

template<typename T>
class GpuFusedScaleTrilKernel final : public user_op::OpKernel {
 public:
  GpuFusedScaleTrilKernel() = default;
  ~GpuFusedScaleTrilKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("in", 0);
    const auto shape = x->shape();
    const auto diagonal = ctx->Attr<int64_t>("diagonal");
    const int32_t num_rows = shape.At(shape.NumAxes() - 2);
    const int32_t num_cols = shape.At(shape.NumAxes() - 1);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int32_t elem_cnt = shape.elem_cnt();
    const T fill = GetAttrVal<T>(ctx->Attr<bool>("is_floating_fill_value"),
                                 ctx->Attr<double>("floating_fill_value"),
                                 ctx->Attr<int64_t>("integer_fill_value"));
    const T scale = GetAttrVal<T>(ctx->Attr<bool>("is_floating_scale_value"),
                                  ctx->Attr<double>("floating_scale_value"),
                                  ctx->Attr<int64_t>("integer_scale_value"));
    if (num_cols % (kCudaWarpSize * 2) == 0) {
      const int64_t total_rows = elem_cnt / num_cols;
      FusedScaleTrilWarpProcessRowGpu<<<BlocksNum4ThreadsNum(total_rows * kCudaWarpSize),
                                        kCudaThreadsNumPerBlock, 0,
                                        ctx->device_ctx()->cuda_stream()>>>(
          total_rows, num_rows, num_cols, diagonal, scale, x->dptr<T>(), fill, y->mut_dptr<T>());
    } else {
      FusedScaleTrilGpu<<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                          ctx->device_ctx()->cuda_stream()>>>(
          elem_cnt, num_rows, num_cols, diagonal, scale, x->dptr<T>(), fill, y->mut_dptr<T>());
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GPU_FUSED_SCALE_TRIL_KERNEL(dtype)                                             \
  REGISTER_USER_KERNEL("fused_scale_tril")                                                      \
      .SetCreateFn<GpuFusedScaleTrilKernel<dtype>>()                                            \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                                       \
                       & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value))         \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, true));                       \
        return Maybe<void>::Ok();                                                               \
      });

REGISTER_GPU_FUSED_SCALE_TRIL_KERNEL(float)
REGISTER_GPU_FUSED_SCALE_TRIL_KERNEL(double)
REGISTER_GPU_FUSED_SCALE_TRIL_KERNEL(int8_t)
REGISTER_GPU_FUSED_SCALE_TRIL_KERNEL(int32_t)
REGISTER_GPU_FUSED_SCALE_TRIL_KERNEL(int64_t)
REGISTER_GPU_FUSED_SCALE_TRIL_KERNEL(half)

}  // namespace oneflow
