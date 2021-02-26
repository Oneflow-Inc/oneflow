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
class MultiCountNotFiniteCpuKernel final : public user_op::OpKernel {
 public:
  MultiCountNotFiniteCpuKernel() = default;
  ~MultiCountNotFiniteCpuKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    int64_t* y_ptr = y->mut_dptr<int64_t>();
    int64_t count = 0;
    FOR_RANGE(int32_t, i, 0, ctx->inputs().size()) {
      user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", i);
      const T* x_ptr = x->dptr<T>();
      FOR_RANGE(int32_t, j, 0, x->shape().elem_cnt()) {
        if (!std::isfinite(x_ptr[j])) { count++; }
      }
    }
    y_ptr[0] = count;
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_COUNT_NOT_FINITE_CPU_KERNEL(dtype)       \
  REGISTER_USER_KERNEL("count_not_finite")                \
      .SetCreateFn<MultiCountNotFiniteCpuKernel<dtype>>() \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu") \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_COUNT_NOT_FINITE_CPU_KERNEL(float)
REGISTER_COUNT_NOT_FINITE_CPU_KERNEL(double)

#define REGISTER_MULTI_COUNT_NOT_FINITE_CPU_KERNEL(dtype) \
  REGISTER_USER_KERNEL("multi_count_not_finite")          \
      .SetCreateFn<MultiCountNotFiniteCpuKernel<dtype>>() \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu") \
                       & (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_MULTI_COUNT_NOT_FINITE_CPU_KERNEL(float)
REGISTER_MULTI_COUNT_NOT_FINITE_CPU_KERNEL(double)

}  // namespace oneflow
