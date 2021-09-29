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
#include "oneflow/core/common/data_type.h"

namespace oneflow {

namespace user_op {

template<DeviceType device_type, typename T>
class CpuScalarFloorDivKernel final : public OpKernel {
 public:
  CpuScalarFloorDivKernel() = default;
  ~CpuScalarFloorDivKernel() = default;

 private:
  void Compute(KernelComputeContext* ctx) const override {
    const Tensor* in_tensor = ctx->Tensor4ArgNameAndIndex("in", 0);
    Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    const T* in_ptr = in_tensor->dptr<T>();
    T* out_ptr = out_tensor->mut_dptr<T>();
    T scalar_operand = static_cast<T>(0);
    if (ctx->Attr<bool>("has_int_operand")) {
      scalar_operand = static_cast<T>(ctx->Attr<int64_t>("int_operand"));
    } else if (ctx->Attr<bool>("has_float_operand")) {
      scalar_operand = static_cast<T>(ctx->Attr<double>("float_operand"));
    } else {
      UNIMPLEMENTED();
    }
    const int64_t elem_cnt = in_tensor->shape().elem_cnt();
    FOR_RANGE(int64_t, i, 0, elem_cnt) { out_ptr[i] = std::floor(in_ptr[i] / scalar_operand); }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CPU_SCALAR_FLOORDIV_KERNEL(device, dtype)   \
  REGISTER_USER_KERNEL("scalar_floordiv")                    \
      .SetCreateFn<CpuScalarFloorDivKernel<device, dtype>>() \
      .SetIsMatchedHob((HobDeviceTag() == device)            \
                       & (HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_CPU_SCALAR_FLOORDIV_KERNEL(DeviceType::kCPU, uint8_t);
REGISTER_CPU_SCALAR_FLOORDIV_KERNEL(DeviceType::kCPU, int8_t);
REGISTER_CPU_SCALAR_FLOORDIV_KERNEL(DeviceType::kCPU, int32_t);
REGISTER_CPU_SCALAR_FLOORDIV_KERNEL(DeviceType::kCPU, int64_t);
REGISTER_CPU_SCALAR_FLOORDIV_KERNEL(DeviceType::kCPU, float);
REGISTER_CPU_SCALAR_FLOORDIV_KERNEL(DeviceType::kCPU, double);

}  // namespace user_op

}  // namespace oneflow
