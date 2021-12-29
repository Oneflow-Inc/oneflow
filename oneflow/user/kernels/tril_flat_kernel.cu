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

__device__ __forceinline__ int64_t TriangularNumber(int64_t n) {
  return n > 0 ? n * (n + 1) / 2 : 0;
}

template<typename T>
__global__ void TrilFlatGpu(const int64_t elem_cnt, const int64_t num_rows, const int64_t num_cols,
                            const int64_t diagonal, const T* x, T* y) {
  // num_cols = 4, diagonal = 2
  // # - triangular area because of diagonal shift
  // x - triangular area limited by num_cols
  // # x * - triangular area reguardless diagonal and column limit
  // * - area should be return
  // #000
  // ##00
  // ***0
  // ****
  // ****x
  // ****xx
  // ...
  const int64_t matrix_size = num_rows * num_cols;
  CUDA_1D_KERNEL_LOOP_T(int64_t, k, elem_cnt) {
    const int64_t offset_in_matrix = k % matrix_size;
    const int64_t row = offset_in_matrix / num_cols;
    const int64_t col = offset_in_matrix - num_cols * row;
    if (row + diagonal >= col) {
      const int64_t row_offset = TriangularNumber(row + diagonal);
      const int64_t diagonal_area = TriangularNumber(diagonal);
      const int64_t area_outside = TriangularNumber(row + diagonal - num_cols);
      const int64_t y_addr = row_offset - diagonal_area - area_outside + col;
      y[y_addr] = x[k];
    }
  }
}

template<typename T>
__global__ void TrilFlatGradGpu(const int64_t elem_cnt, const int64_t num_rows,
                                const int64_t num_cols, const int64_t diagonal, const T* dy,
                                T* dx) {
  const int64_t matrix_size = num_rows * num_cols;
  CUDA_1D_KERNEL_LOOP_T(int64_t, k, elem_cnt) {
    const int64_t offset_in_matrix = k % matrix_size;
    const int64_t row = offset_in_matrix / num_cols;
    const int64_t col = offset_in_matrix - num_cols * row;
    if (row + diagonal >= col) {
      const int64_t row_offset = TriangularNumber(row + diagonal);
      const int64_t diagonal_area = TriangularNumber(diagonal);
      const int64_t area_outside = TriangularNumber(row + diagonal - num_cols);
      const int64_t dy_addr = row_offset - diagonal_area - area_outside + col;
      dx[k] = dy[dy_addr];
    } else {
      dx[k] = 0;
    }
  }
}

}  // namespace

template<typename T>
class GpuTrilFlatKernel final : public user_op::OpKernel {
 public:
  GpuTrilFlatKernel() = default;
  ~GpuTrilFlatKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("in", 0);
    const auto shape = x->shape();
    const auto diagonal = ctx->Attr<int64_t>("diagonal");
    const int64_t num_rows = shape.At(shape.NumAxes() - 2);
    const int64_t num_cols = shape.At(shape.NumAxes() - 1);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int32_t elem_cnt = shape.elem_cnt();

    TrilFlatGpu<<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                  ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
        elem_cnt, num_rows, num_cols, diagonal, x->dptr<T>(), y->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
class GpuTrilFlatBackwardKernel final : public user_op::OpKernel {
 public:
  GpuTrilFlatBackwardKernel() = default;
  ~GpuTrilFlatBackwardKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const auto shape = dx->shape();
    const auto diagonal = ctx->Attr<int64_t>("diagonal");
    const int64_t num_rows = shape.At(shape.NumAxes() - 2);
    const int64_t num_cols = shape.At(shape.NumAxes() - 1);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const int32_t elem_cnt = shape.elem_cnt();

    TrilFlatGradGpu<<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                      ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
        elem_cnt, num_rows, num_cols, diagonal, dy->dptr<T>(), dx->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

// #define REGISTER_CUDA_TRIL_FLAT_KERNEL(dtype)                                       \
//   REGISTER_USER_KERNEL("tril_flat")                                                 \
//       .SetCreateFn<GpuTrilFlatKernel<dtype>>()                                      \
//       .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)              \
//                        && (user_op::HobDataType("out", 0)                           \
//                            == GetDataType<dtype>::value)) return Maybe<void>::Ok(); \
//   });                                                                               \
//   REGISTER_USER_KERNEL("tril_flat_grad")                                            \
//       .SetCreateFn<GpuTrilFlatGradKernel<dtype>>()                                  \
//       .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)              \
//                        && (user_op::HobDataType("dy", 0)                            \
//                            == GetDataType<dtype>::value)) return Maybe<void>::Ok(); \
// });
#define REGISTER_CUDA_TRIL_FLAT_KERNEL(dtype)                                              \
  REGISTER_USER_KERNEL("tril_flat")                                                       \
      .SetCreateFn<GpuTrilFlatKernel<dtype>>()                                            \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                     \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value)); \
  REGISTER_USER_KERNEL("tril_flat_grad")                                                  \
      .SetCreateFn<GpuTrilFlatBackwardKernel<dtype>>()                                    \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                     \
                       && (user_op::HobDataType("dy", 0) == GetDataType<dtype>::value));

REGISTER_CUDA_TRIL_FLAT_KERNEL(float)
REGISTER_CUDA_TRIL_FLAT_KERNEL(double)
REGISTER_CUDA_TRIL_FLAT_KERNEL(uint8_t)
REGISTER_CUDA_TRIL_FLAT_KERNEL(int8_t)
REGISTER_CUDA_TRIL_FLAT_KERNEL(int32_t)
REGISTER_CUDA_TRIL_FLAT_KERNEL(int64_t)
REGISTER_CUDA_TRIL_FLAT_KERNEL(half)

}  // namespace oneflow
