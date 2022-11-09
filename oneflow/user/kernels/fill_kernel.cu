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
#include "oneflow/core/common/nd_index_offset_helper.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

namespace {
template<typename T>
__global__ void FillTensorGpuForward(const int n, const T* value, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = value[0]; }
}
};  // namespace

template<typename T>
class FillTensorGpuKernel final : public user_op::OpKernel {
 public:
  FillTensorGpuKernel() = default;
  ~FillTensorGpuKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const user_op::Tensor* value = ctx->Tensor4ArgNameAndIndex("value", 0);
    const int32_t elem_cnt = in->shape_view().elem_cnt();
    RUN_CUDA_KERNEL((FillTensorGpuForward<T>), ctx->stream(), elem_cnt, elem_cnt, value->dptr<T>(),
                    out->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FILL_CUDA_KERNEL(dtype)                               \
  REGISTER_USER_KERNEL("fill_tensor_")                                 \
      .SetCreateFn<FillTensorGpuKernel<dtype>>()                       \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_FILL_CUDA_KERNEL(float)
REGISTER_FILL_CUDA_KERNEL(half)
REGISTER_FILL_CUDA_KERNEL(double)
REGISTER_FILL_CUDA_KERNEL(int8_t)
REGISTER_FILL_CUDA_KERNEL(int32_t)
REGISTER_FILL_CUDA_KERNEL(int64_t)

}  // namespace oneflow
