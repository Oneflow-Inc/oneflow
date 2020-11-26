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

namespace {

template<bool assign_if, typename C>
class AssignIfCPUKernel final : public user_op::OpKernel {
 public:
  AssignIfCPUKernel() = default;
  ~AssignIfCPUKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* condition = ctx->Tensor4ArgNameAndIndex("condition", 0);
    if ((assign_if == (*condition->dptr<C>() == 0))) { return; }
    const user_op::Tensor* value = ctx->Tensor4ArgNameAndIndex("value", 0);
    user_op::Tensor* ref = ctx->Tensor4ArgNameAndIndex("ref", 0);
    if (value->dptr() == ref->dptr()) { return; }
    CHECK_EQ(value->shape(), ref->shape());
    CHECK_EQ(value->data_type(), ref->data_type());
    const size_t tensor_bytes_size = ref->shape().elem_cnt() * GetSizeOfDataType(ref->data_type());
    AutoMemcpy(ctx->device_ctx(), ref->mut_dptr(), value->dptr(), tensor_bytes_size,
               ref->mem_case(), value->mem_case());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

}  // namespace

#define REGISTER_ASSIGN_WITH_CONDITION_CPU_KERNEL(op_type_name, assign_if, condition_type) \
  REGISTER_USER_KERNEL(op_type_name)                                                       \
      .SetCreateFn<AssignIfCPUKernel<assign_if, condition_type>>()                         \
      .SetIsMatchedHob(                                                                    \
          (user_op::HobDeviceTag() == DeviceType::kCPU)                                    \
          & (user_op::HobDataType("condition", 0) == GetDataType<condition_type>::value));

#define REGISTER_ASSIGN_IF_CPU_KERNEL(condition_cpp_type, condition_data_type)      \
  REGISTER_ASSIGN_WITH_CONDITION_CPU_KERNEL("assign_if", true, condition_cpp_type); \
  REGISTER_ASSIGN_WITH_CONDITION_CPU_KERNEL("assign_if_not", false, condition_cpp_type)

OF_PP_FOR_EACH_TUPLE(REGISTER_ASSIGN_IF_CPU_KERNEL, INT_DATA_TYPE_SEQ)

}  // namespace oneflow
