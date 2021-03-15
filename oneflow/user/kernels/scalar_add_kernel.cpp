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
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class ScalarAddUserKernel final : public user_op::OpKernel {
 public:
  ScalarAddUserKernel() = default;
  ~ScalarAddUserKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    T scalar_operand = static_cast<T>(0);
    if (ctx->Attr<bool>("has_int_operand")) {
      scalar_operand = static_cast<T>(ctx->Attr<int64_t>("int_operand"));
    } else if (ctx->Attr<bool>("has_float_operand")) {
      scalar_operand = static_cast<T>(ctx->Attr<double>("float_operand"));
    } else {
      UNIMPLEMENTED();
    }
    const T* in_ptr = in->dptr<T>();
    T* out_ptr = out->mut_dptr<T>();

    NewKernelUtil<device_type>::AddByScalar(ctx->device_ctx(), out->shape().elem_cnt(), in_ptr,
                                            scalar_operand, out_ptr);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_KERNEL(kernel_device_type, dtype)                                              \
  REGISTER_USER_KERNEL("scalar_add")                                                            \
      .SetCreateFn<ScalarAddUserKernel<DeviceType::k##kernel_device_type, dtype>>()             \
      .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::k##kernel_device_type)           \
                       & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value))         \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, true));                       \
        return Maybe<void>::Ok();                                                               \
      });

REGISTER_KERNEL(CPU, int8_t)
REGISTER_KERNEL(CPU, int32_t)
REGISTER_KERNEL(CPU, int64_t)
REGISTER_KERNEL(CPU, float)
REGISTER_KERNEL(CPU, double)
#ifdef WITH_CUDA
REGISTER_KERNEL(GPU, int8_t)
REGISTER_KERNEL(GPU, int32_t)
REGISTER_KERNEL(GPU, int64_t)
REGISTER_KERNEL(GPU, float)
REGISTER_KERNEL(GPU, double)
REGISTER_KERNEL(GPU, float16)
#endif

}  // namespace oneflow
