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
class CpuMedianKernel final : public user_op::OpKernel {
 public:
  CpuMedianKernel() = default;
  ~CpuMedianKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("input", 0);
    const int64_t size = in->shape_view().elem_cnt();
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("output", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    T* out_ptr = out->mut_dptr<T>();
    Memcpy<DeviceType::kCPU>(ctx->stream(), tmp_buffer->mut_dptr<void>(), in->dptr<void>(),
                             size * sizeof(T));
    T* first = tmp_buffer->mut_dptr<T>();
    T* last = first + size;
    T* median = first + (size - 1) / 2;
    std::nth_element(first, median, last);
    *out_ptr = *median;
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CPU_MEDIAN_KERNEL(dtype)                                                  \
  REGISTER_USER_KERNEL("median")                                                           \
      .SetCreateFn<CpuMedianKernel<dtype>>()                                               \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)                      \
                       && (user_op::HobDataType("input", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t {                        \
        return ctx->InputShape("input", 0).elem_cnt() * sizeof(dtype);                     \
      });

REGISTER_CPU_MEDIAN_KERNEL(float)
REGISTER_CPU_MEDIAN_KERNEL(double)
REGISTER_CPU_MEDIAN_KERNEL(int8_t)
REGISTER_CPU_MEDIAN_KERNEL(uint8_t)
REGISTER_CPU_MEDIAN_KERNEL(int32_t)
REGISTER_CPU_MEDIAN_KERNEL(int64_t)

}  // namespace oneflow
