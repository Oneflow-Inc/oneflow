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

template<typename T>
class CpuLeakyReluYZHKernel final : public user_op::OpKernel {
 public:
  CpuLeakyReluYZHKernel() = default;
  ~CpuLeakyReluYZHKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const int32_t elem_cnt = x->shape().elem_cnt();
    const T* x_ptr = x->dptr<T>();
    T* y_ptr = y->mut_dptr<T>();
    const auto alpha = ctx->Attr<float>("alpha");
    FOR_RANGE(int32_t, i, 0, elem_cnt) { y_ptr[i] = x_ptr[i] > 0 ? x_ptr[i] : alpha * x_ptr[i]; }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CPU_LEAKY_RELU_YZH_KERNEL(dtype)                     \
  REGISTER_USER_KERNEL("leaky_relu_yzh")                              \
      .SetCreateFn<CpuLeakyReluYZHKernel<dtype>>()                    \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU) \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

REGISTER_CPU_LEAKY_RELU_YZH_KERNEL(float)
REGISTER_CPU_LEAKY_RELU_YZH_KERNEL(double)
REGISTER_CPU_LEAKY_RELU_YZH_KERNEL(uint8_t)
REGISTER_CPU_LEAKY_RELU_YZH_KERNEL(int8_t)
REGISTER_CPU_LEAKY_RELU_YZH_KERNEL(int32_t)
REGISTER_CPU_LEAKY_RELU_YZH_KERNEL(int64_t)

}  // namespace oneflow