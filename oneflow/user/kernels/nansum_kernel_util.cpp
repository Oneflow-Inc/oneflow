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

template<typename T>
struct NanSumKernel {
  static void Forward(ep::Stream* stream, const int64_t num_elements, const T* input, T* output) {
    FOR_RANGE(int32_t, i, 0, num_elements) { output[i] = isnan(input[i]) ? T{0.} : input[i]; }
  }
};

template<typename T>
class CpuNanSumKernel final : public user_op::OpKernel {
 public:
  CpuNanSumKernel() = default;
  ~CpuNanSumKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("input", 0);
    const int64_t elements = in->shape_view().elem_cnt();
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("output", 0);
    NanSumKernel<T>::Forward(ctx->stream(), elements, in->dptr<T>(), out->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CPU_NANSUM_KERNEL(dtype)                             \
  REGISTER_USER_KERNEL("replace_nansum")                              \
      .SetCreateFn<CpuNanSumKernel<dtype>>()                          \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU) \
                       && (user_op::HobDataType("input", 0) == GetDataType<dtype>::value));

REGISTER_CPU_NANSUM_KERNEL(float);
REGISTER_CPU_NANSUM_KERNEL(double)

}  // namespace oneflow
