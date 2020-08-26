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

namespace oneflow {

namespace {

template<typename T>
__global__ void TrilCalGpu(const int64_t elem_cnt, const int64_t row, const int64_t col,
                           const int64_t diagonal, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP_T(int64_t, k, elem_cnt) {
    int64_t i = (k % (row * col)) / col;
    int64_t j = (k % (row * col)) % col;
    y[k] = j > i + diagonal ? 0 : x[k];
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

#define REGISTER_GPU_TRIL_KERNEL(dtype)                   \
  REGISTER_USER_KERNEL("tril")                            \
      .SetCreateFn<GpuTrilKernel<dtype>>()                \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu") \
                       & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_GPU_TRIL_KERNEL(float)
REGISTER_GPU_TRIL_KERNEL(double)
REGISTER_GPU_TRIL_KERNEL(int8_t)
REGISTER_GPU_TRIL_KERNEL(int32_t)
REGISTER_GPU_TRIL_KERNEL(int64_t)

template<typename T>
class GpuTrilGradKernel final : public user_op::OpKernel {
 public:
  GpuTrilGradKernel() = default;
  ~GpuTrilGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const auto shape = dy->shape();
    const int64_t diagonal = ctx->Attr<int64_t>("diagonal");
    const int64_t row = shape.At(shape.NumAxes() - 2);
    const int64_t col = shape.At(shape.NumAxes() - 1);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const int32_t elem_cnt = shape.elem_cnt();
    RUN_CUDA_KERNEL((TrilCalGpu<T>), ctx->device_ctx(), elem_cnt, elem_cnt, row, col, diagonal,
                    dy->dptr<T>(), dx->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};


#define REGISTER_GPU_TRIL_GRAD_KERNEL(dtype)              \
  REGISTER_USER_KERNEL("tril_grad")                       \
      .SetCreateFn<GpuTrilGradKernel<dtype>>()            \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu") \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_GPU_TRIL_GRAD_KERNEL(float)
REGISTER_GPU_TRIL_GRAD_KERNEL(double)
REGISTER_GPU_TRIL_GRAD_KERNEL(int8_t)
REGISTER_GPU_TRIL_GRAD_KERNEL(int32_t)
REGISTER_GPU_TRIL_GRAD_KERNEL(int64_t)

}  // namespace oneflow
