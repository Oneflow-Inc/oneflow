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
#include "oneflow/user/kernels/roll_kernel_utils.h"

#include <algorithm>

namespace oneflow {

template<typename T>
class CpuRepeatInterLeaveKernel final : public user_op::OpKernel {
 public:
  CpuRepeatInterLeaveKernel() = default;
  ~CpuRepeatInterLeaveKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const user_op::Tensor* cumsum = ctx->Tensor4ArgNameAndIndex("cumsum", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const T* in_ptr = in->dptr<T>();
    const T* cumsum_ptr = cumsum->dptr<T>();
    T* out_ptr = out->mut_dptr<T>();
    for (T i = 0; i < in->shape_view().At(0); i++) {
      T end = cumsum_ptr[i];
      T size = in_ptr[i];
      T start = end - size;
      for (T j = start; j < end; j++) { out_ptr[j] = i; }
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_REPEAT_INTER_LEAVE_KERNEL(dtype)                     \
  REGISTER_USER_KERNEL("repeat_interleave")                           \
      .SetCreateFn<CpuRepeatInterLeaveKernel<dtype>>()                \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU) \
                       && (user_op::HobDataType("in", 0) == GetDataType<dtype>::value))

REGISTER_REPEAT_INTER_LEAVE_KERNEL(int32_t);
REGISTER_REPEAT_INTER_LEAVE_KERNEL(int64_t);

}  // namespace oneflow
