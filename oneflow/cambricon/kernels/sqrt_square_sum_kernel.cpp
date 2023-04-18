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
#include "oneflow/cambricon/bang/bang_kernels.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {
namespace user_op {

template<typename T>
class MluSqrtSquareSumKernel final : public user_op::OpKernel {
 public:
  MluSqrtSquareSumKernel() = default;
  ~MluSqrtSquareSumKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);

    auto* stream = ctx->stream()->As<ep::MluStream>();
    BangHandle handle(stream->mlu_stream(), stream->device()->nclusters(),
                      stream->device()->ncores_per_cluster());
    bang_sqrt_square_sum_kernel(handle, x->shape_view().elem_cnt(), x->dptr<T>(), y->mut_dptr<T>());
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_MLU_SQUARE_SUM_KERNEL(T)                             \
  REGISTER_USER_KERNEL("sqrt_square_sum")                             \
      .SetCreateFn<MluSqrtSquareSumKernel<T>>()                       \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU) \
                       && (user_op::HobDataType("y", 0) == GetDataType<T>::value));

REGISTER_MLU_SQUARE_SUM_KERNEL(float)

#undef REGISTER_MLU_SQUARE_SUM_KERNEL

}  // namespace user_op
}  // namespace oneflow
