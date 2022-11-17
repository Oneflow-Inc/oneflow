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
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/user_op_tensor.h"

namespace oneflow {

namespace {

template<typename T>
class CpuLerpKernel final : public user_op::OpKernel {
 public:
  CpuLerpKernel() = default;
  ~CpuLerpKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* start = ctx->Tensor4ArgNameAndIndex("start", 0);
    const user_op::Tensor* end = ctx->Tensor4ArgNameAndIndex("end", 0);
    const user_op::Tensor* weight = ctx->Tensor4ArgNameAndIndex("weight", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    const int64_t start_elem_cnt = start->shape_view().elem_cnt();
    const int64_t end_elem_cnt = end->shape_view().elem_cnt();
    const int64_t weight_elem_cnt = weight->shape_view().elem_cnt();
    CHECK_EQ(start_elem_cnt, end_elem_cnt);
    CHECK_EQ(start_elem_cnt, weight_elem_cnt);

    const T* start_ptr = start->dptr<T>();
    const T* end_ptr = end->dptr<T>();
    const T* weight_ptr = weight->dptr<T>();
    T* out_ptr = out->dptr<T>();

    FOR_RANGE(int64_t, i, 0, start_elem_cnt) {
      out_ptr[i] = start_ptr[i] + weight_ptr[i] * (end_ptr[i] - start_ptr[i]);
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CPU_LERP_KERNEL(dtype)                               \
  REGISTER_USER_KERNEL("lerp")                                        \
      .SetCreateFn<CpuLerpKernel<dtype>>()                            \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU) \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_CPU_LERP_KERNEL(float)
REGISTER_CPU_LERP_KERNEL(double)
REGISTER_CPU_LERP_KERNEL(uint8_t)
REGISTER_CPU_LERP_KERNEL(int8_t)
REGISTER_CPU_LERP_KERNEL(int32_t)
REGISTER_CPU_LERP_KERNEL(int64_t)

}  // namespace

}  // namespace oneflow