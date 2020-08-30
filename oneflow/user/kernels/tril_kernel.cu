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
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/util/cuda_half_util.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void TrilCalGpu(const int64_t elem_cnt, const int64_t row, const int64_t col,
                           const int64_t diagonal, const T* x, T* y) {
  T zero = GetZeroVal<T>();
  int64_t matrix_cnt = row * col;
  CUDA_1D_KERNEL_LOOP_T(int64_t, k, elem_cnt) {
    int64_t index_in_matrix = k - matrix_cnt * (k / matrix_cnt);
    int64_t i = index_in_matrix / col;
    int64_t j = index_in_matrix - col * (index_in_matrix / col);
    y[k] = j > i + diagonal ? zero : x[k];
  }
}

__global__ void NaiveHalfTrilCalGpu(const int64_t elem_cnt, const int64_t row, const int64_t col,
                                    const int64_t diagonal, const half* x, half* y) {
  half zero = hzero();
  int64_t matrix_cnt = row * col;
  CUDA_1D_KERNEL_LOOP_T(int64_t, k, elem_cnt) {
    int64_t index_in_matrix = k - matrix_cnt * (k / matrix_cnt);
    int64_t i = index_in_matrix / col;
    int64_t j = index_in_matrix - col * (index_in_matrix / col);
    y[k] = j > i + diagonal ? zero : x[k];
  }
}

}  // namespace

template<typename T>
class GpuTrilKernel final : public user_op::OpKernel {
 public:
  GpuTrilKernel() = default;
  ~GpuTrilKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("in", 0);
    const auto shape = x->shape();
    const int64_t diagonal = ctx->Attr<int64_t>("diagonal");
    const int64_t row = shape.At(shape.NumAxes() - 2);
    const int64_t col = shape.At(shape.NumAxes() - 1);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int32_t elem_cnt = shape.elem_cnt();
    RUN_CUDA_KERNEL((TrilCalGpu<T>), ctx->device_ctx(), elem_cnt, elem_cnt, row, col, diagonal,
                    x->dptr<T>(), y->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<>
class GpuTrilKernel<float16> final : public user_op::OpKernel {
 public:
  GpuTrilKernel() = default;
  ~GpuTrilKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("in", 0);
    const auto shape = x->shape();
    const int64_t diagonal = ctx->Attr<int64_t>("diagonal");
    const int64_t row = shape.At(shape.NumAxes() - 2);
    const int64_t col = shape.At(shape.NumAxes() - 1);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int32_t elem_cnt = shape.elem_cnt();
    RUN_CUDA_KERNEL(NaiveHalfTrilCalGpu, ctx->device_ctx(), elem_cnt, elem_cnt, row, col, diagonal,
                    reinterpret_cast<const half*>(x->dptr<float16>()),
                    reinterpret_cast<half*>(y->mut_dptr<float16>()));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GPU_TRIL_KERNEL(dtype)                                             \
  REGISTER_USER_KERNEL("tril").SetCreateFn<GpuTrilKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceTag() == "gpu")                                            \
      & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_GPU_TRIL_KERNEL(float)
REGISTER_GPU_TRIL_KERNEL(double)
REGISTER_GPU_TRIL_KERNEL(int8_t)
REGISTER_GPU_TRIL_KERNEL(int32_t)
REGISTER_GPU_TRIL_KERNEL(int64_t)
REGISTER_GPU_TRIL_KERNEL(float16)

}  // namespace oneflow
