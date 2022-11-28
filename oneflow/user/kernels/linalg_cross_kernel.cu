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
#include "oneflow/core/framework/op_kernel.h"
#include "oneflow/core/device/cuda_util.h"

namespace {

template<typename T>
__global__ void LinalgCrossForward(const int64_t n, const T* input, const T* other, T* out) {
  CUDA_1D_KERNEL_LOOP_T(int64_t, i, n) {
    const int64_t index = i * 3;
    out[index] = input[index + 1] * other[index + 2] - input[index + 2] * other[index + 1];
    out[index + 1] = input[index + 2] * other[index] - input[index] * other[index + 2];
    out[index + 2] = input[index] * other[index + 1] - input[index + 1] * other[index];
  }
}

}  // namespace

namespace oneflow {

template<typename T>
class CudaLinalgCrossKernel final : public user_op::OpKernel {
 public:
  CudaLinalgCrossKernel() = default;
  ~CudaLinalgCrossKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto* input_tensor = ctx->Tensor4ArgNameAndIndex("input", 0);
    const auto* other_tensor = ctx->Tensor4ArgNameAndIndex("other", 0);
    auto* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);

    const int64_t n = input_tensor->shape_view().elem_cnt() / 3;

    if (n == 0) { return; }
    RUN_CUDA_KERNEL((LinalgCrossForward<T>), ctx->stream(), n, n, input_tensor->dptr<T>(),
                    other_tensor->dptr<T>(), out_tensor->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_LINALG_CROSS_KERNEL(dtype)                       \
  REGISTER_USER_KERNEL("linalg_cross")                                 \
      .SetCreateFn<CudaLinalgCrossKernel<dtype>>()                     \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("input", 0) == GetDataType<dtype>::value));

REGISTER_CUDA_LINALG_CROSS_KERNEL(float)
REGISTER_CUDA_LINALG_CROSS_KERNEL(double)

}  // namespace oneflow
