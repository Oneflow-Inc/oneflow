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

template<typename T, typename U>
class FusedCastScaleCpuKernel final : public user_op::OpKernel {
 public:
  FusedCastScaleCpuKernel() = default;
  ~FusedCastScaleCpuKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* scalar = ctx->Tensor4ArgNameAndIndex("scalar", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const int64_t n = x->shape().elem_cnt();
    const auto scalar_val = *(scalar->dptr<T>());
    const U* x_ptr = x->dptr<U>();
    T* y_ptr = y->mut_dptr<T>();
    FOR_RANGE(int64_t, i, 0, n) { y_ptr[i] = static_cast<T>(x_ptr[i]) * scalar_val; }
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_CAST_SCALE_CPU_KERNEL(x_type, y_type)                          \
  REGISTER_USER_KERNEL("fused_cast_scale")                                            \
      .SetCreateFn<FusedCastScaleCpuKernel<y_type, x_type>>()                         \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")                             \
                       & (user_op::HobDataType("y", 0) == GetDataType<y_type>::value) \
                       & (user_op::HobDataType("x", 0) == GetDataType<x_type>::value));

REGISTER_FUSED_CAST_SCALE_CPU_KERNEL(float, double);
REGISTER_FUSED_CAST_SCALE_CPU_KERNEL(double, float);
#undef REGISTER_FUSED_CAST_SCALE_CPU_KERNEL

}  // namespace oneflow
