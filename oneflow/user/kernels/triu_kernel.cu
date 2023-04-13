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
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void TriuGpu(const int64_t elem_cnt, const int64_t num_rows, const int64_t num_cols,
                        const int64_t diagonal, const T* x, T* y) {
  const int64_t matrix_size = num_rows * num_cols;
  CUDA_1D_KERNEL_LOOP_T(int64_t, k, elem_cnt) {
    const int64_t offset_in_matrix = k % matrix_size;
    const int64_t i = offset_in_matrix / num_cols;
    const int64_t j = offset_in_matrix - num_cols * i;
    y[k] = j < i + diagonal ? static_cast<T>(0) : x[k];
  }
}

template<typename T>
__global__ void TriuWarpProcessRowGpu(const int64_t total_rows, const int64_t num_rows,
                                      const int64_t num_cols, const int64_t diagonal, const T* x,
                                      T* y) {
  const int64_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / kCudaWarpSize;
  const int64_t lan_id = threadIdx.x % kCudaWarpSize;
  const int64_t num_warp = blockDim.x * gridDim.x / kCudaWarpSize;
  for (int64_t i = warp_id; i < total_rows; i += num_warp) {
    const int64_t row = i % num_rows;
    for (int64_t col = lan_id; col < num_cols; col += kCudaWarpSize) {
      const int64_t idx = i * num_cols + col;
      y[idx] = col < row + diagonal ? static_cast<T>(0) : x[idx];
    }
  }
}

template<>
__global__ void TriuWarpProcessRowGpu<half>(const int64_t total_rows, const int64_t num_rows,
                                            const int64_t num_cols, const int64_t diagonal,
                                            const half* x, half* y) {
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
      y_val.x = (2 * col) < row + diagonal ? static_cast<half>(0) : x_val.x;
      y_val.y = (2 * col + 1) < row + diagonal ? static_cast<half>(0) : x_val.y;
      y_h2[idx] = y_val;
    }
  }
}

}  // namespace

template<typename T>
class GpuTriuKernel final : public user_op::OpKernel {
 public:
  GpuTriuKernel() = default;
  ~GpuTriuKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("in", 0);
    const auto shape = x->shape_view();
    const auto diagonal = ctx->Attr<int64_t>("diagonal");
    const int64_t num_rows = shape.At(shape.NumAxes() - 2);
    const int64_t num_cols = shape.At(shape.NumAxes() - 1);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int32_t elem_cnt = shape.elem_cnt();
    if (elem_cnt == 0) { return; }
    if (num_cols % (kCudaWarpSize * 2) == 0) {
      const int64_t total_rows = elem_cnt / num_cols;
      TriuWarpProcessRowGpu<<<BlocksNum4ThreadsNum(total_rows * kCudaWarpSize),
                              kCudaThreadsNumPerBlock, 0,
                              ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
          total_rows, num_rows, num_cols, diagonal, x->dptr<T>(), y->mut_dptr<T>());
    } else {
      TriuGpu<<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
          elem_cnt, num_rows, num_cols, diagonal, x->dptr<T>(), y->mut_dptr<T>());
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_TRIU_KERNEL(dtype)                                                        \
  REGISTER_USER_KERNEL("triu")                                                                  \
      .SetCreateFn<GpuTriuKernel<dtype>>()                                                      \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                          \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value))        \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, true));                       \
        return Maybe<void>::Ok();                                                               \
      });

REGISTER_CUDA_TRIU_KERNEL(half)
REGISTER_CUDA_TRIU_KERNEL(float)
REGISTER_CUDA_TRIU_KERNEL(double)
REGISTER_CUDA_TRIU_KERNEL(uint8_t)
REGISTER_CUDA_TRIU_KERNEL(int8_t)
REGISTER_CUDA_TRIU_KERNEL(int32_t)
REGISTER_CUDA_TRIU_KERNEL(int64_t)
REGISTER_CUDA_TRIU_KERNEL(bool)

}  // namespace oneflow
