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
class CpuYZHReluKernel final : public user_op::OpKernel {
 public:
  CpuYZHReluKernel() = default;
  ~CpuYZHReluKernel() = default;
 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const int32_t elem_cnt = x->shape().elem_cnt();
    const T* x_ptr = x->dptr<T>();
    T* y_ptr = y->mut_dptr<T>();
    FOR_RANGE(int32_t, i, 0, elem_cnt) { y_ptr[i] = x_ptr[i] > 0 ? x_ptr[i] : 0; }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CPU_YZH_RELU_KERNEL(dtype)             \
  REGISTER_USER_KERNEL("yzh_relu")                      \
      .SetCreateFn<CpuYZHReluKernel<dtype>>()           \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU) \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

REGISTER_CPU_YZH_RELU_KERNEL(float)
REGISTER_CPU_YZH_RELU_KERNEL(double)
REGISTER_CPU_YZH_RELU_KERNEL(uint8_t)
REGISTER_CPU_YZH_RELU_KERNEL(int8_t)
REGISTER_CPU_YZH_RELU_KERNEL(int32_t)
REGISTER_CPU_YZH_RELU_KERNEL(int64_t)

}  // namespace oneflow