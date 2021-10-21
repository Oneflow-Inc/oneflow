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
#ifndef ONEFLOW_USER_KERNELS_BIAS_ADD_KERNEL_H_
#define ONEFLOW_USER_KERNELS_BIAS_ADD_KERNEL_H_

#include "oneflow/core/framework/framework.h"

namespace oneflow {

template<DeviceType device_type, typename T, typename Index>
struct BiasAddCalculation {
  static void Invoke(DeviceCtx* ctx, Index outer_size, Index bias_size, Index inner_size,
                     const T* x, const T* bias, T* y);
};

template<DeviceType device_type, typename T>
class BiasAddUserKernel final : public user_op::OpKernel {
 public:
  BiasAddUserKernel() = default;
  ~BiasAddUserKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto* a_tensor = ctx->Tensor4ArgNameAndIndex("a", 0);
    const auto* b_tensor = ctx->Tensor4ArgNameAndIndex("b", 0);
    auto* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int32_t bias_add_axis = ctx->Attr<int32_t>("axis");
    const int64_t outer_size = a_tensor->shape().Count(0, bias_add_axis);
    const int64_t bias_size = a_tensor->shape().At(bias_add_axis);
    const int64_t inner_size = a_tensor->shape().Count(bias_add_axis + 1);
    const auto n = a_tensor->shape().elem_cnt();
    if (IsKernelSafeInt32(n)) {
      BiasAddCalculation<device_type, T, int32_t>::Invoke(
          ctx->device_ctx(), outer_size, bias_size, inner_size, a_tensor->dptr<T>(),
          b_tensor->dptr<T>(), out_tensor->mut_dptr<T>());
    } else {
      BiasAddCalculation<device_type, T, int64_t>::Invoke(
          ctx->device_ctx(), outer_size, bias_size, inner_size, a_tensor->dptr<T>(),
          b_tensor->dptr<T>(), out_tensor->mut_dptr<T>());
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_BIAS_ADD_USER_KERNEL(op_device_type, dtype)                                    \
  REGISTER_USER_KERNEL("bias_add")                                                              \
      .SetCreateFn<BiasAddUserKernel<DeviceType::k##op_device_type, dtype>>()                   \
      .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::k##op_device_type)               \
                       & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value))         \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "a", 0, true));                        \
        return Maybe<void>::Ok();                                                               \
      });

}  // namespace oneflow
#endif  // ONEFLOW_USER_KERNELS_BIAS_ADD_KERNEL_H_
