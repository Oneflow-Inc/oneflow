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

template<bool assign_if, typename C, typename T>
__global__ void AssignGpu(int64_t elem_cnt, const C* condition, const T* value, T* ref) {
  if (assign_if == (*condition == 0)) { return; }
  CUDA_1D_KERNEL_LOOP(i, elem_cnt) { ref[i] = value[i]; }
}

template<bool assign_if, typename C, typename T>
class AssignIfGPUKernel final : public user_op::OpKernel {
 public:
  AssignIfGPUKernel() = default;
  ~AssignIfGPUKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* condition = ctx->Tensor4ArgNameAndIndex("condition", 0);
    CHECK_EQ(condition->shape().NumAxes(), 1);
    CHECK_EQ(condition->shape().At(0), 1);
    const user_op::Tensor* value = ctx->Tensor4ArgNameAndIndex("value", 0);
    user_op::Tensor* ref = ctx->Tensor4ArgNameAndIndex("ref", 0);
    if (value->dptr() == ref->dptr()) { return; }
    CHECK_EQ(value->shape(), ref->shape());
    CHECK_EQ(value->data_type(), ref->data_type());
    const size_t elem_cnt = ref->shape().elem_cnt();
    AssignGpu<assign_if, C, T><<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                                 ctx->device_ctx()->cuda_stream()>>>(
        elem_cnt, condition->dptr<C>(), value->dptr<T>(), ref->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

}  // namespace

#define REGISTER_ASSIGN_WITH_CONDITION_VALUE_GPU_KERNEL(op_type_name, assign_if, condition_type, \
                                                        value_type)                              \
  REGISTER_USER_KERNEL(op_type_name)                                                             \
      .SetCreateFn<AssignIfGPUKernel<assign_if, condition_type, value_type>>()                   \
      .SetIsMatchedHob(                                                                          \
          (user_op::HobDeviceTag() == DeviceType::kGPU)                                          \
          & (user_op::HobDataType("condition", 0) == GetDataType<condition_type>::value)         \
          & (user_op::HobDataType("value", 0) == GetDataType<value_type>::value));

#define REGISTER_ASSIGN_IF_GPU_KERNEL(condition_type, value_type)                         \
  REGISTER_ASSIGN_WITH_CONDITION_VALUE_GPU_KERNEL(                                        \
      "assign_if", true, OF_PP_PAIR_FIRST(condition_type), OF_PP_PAIR_FIRST(value_type)); \
  REGISTER_ASSIGN_WITH_CONDITION_VALUE_GPU_KERNEL(                                        \
      "assign_if_not", false, OF_PP_PAIR_FIRST(condition_type), OF_PP_PAIR_FIRST(value_type))

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_ASSIGN_IF_GPU_KERNEL, INT_DATA_TYPE_SEQ,
                                 POD_DATA_TYPE_SEQ)

}  // namespace oneflow
