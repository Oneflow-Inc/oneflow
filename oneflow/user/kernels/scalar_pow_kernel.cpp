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
class CpuScalarPowKernel final : public OpKernel {
 public:
  CpuScalarPowKernel() = default;
  ~CpuScalarPowKernel() = default;

 private:
  void Compute(KernelComputeContext* ctx) const override {
    const Tensor* in_tensor = ctx->Tensor4ArgNameAndIndex("in", 0);
    Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    const T* in_ptr = in_tensor->dptr<T>();
    T* out_ptr = out_tensor->mut_dptr<T>();
    const T exponent = static_cast<T>(ctx->Attr<double>("exponent"));

    const int64_t elem_cnt = in_tensor->shape().elem_cnt();
    FOR_RANGE(int64_t, i, 0, elem_cnt) { out_ptr[i] = std::pow(in_ptr[i], exponent); }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CPU_SCALAR_POW_KERNEL(device, dtype)   \
  REGISTER_USER_KERNEL("scalar_pow")                    \
      .SetCreateFn<CpuScalarPowKernel<device, dtype>>() \
      .SetIsMatchedHob((HobDeviceTag() == device)       \
                       & (HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_CPU_SCALAR_POW_KERNEL(DeviceType::kCPU, float);
REGISTER_CPU_SCALAR_POW_KERNEL(DeviceType::kCPU, double);

template<DeviceType device_type, typename T>
class CpuScalarPowGradKernel final : public OpKernel {
 public:
  CpuScalarPowGradKernel() = default;
  ~CpuScalarPowGradKernel() = default;

 private:
  void Compute(KernelComputeContext* ctx) const override {
    const Tensor* x_tensor = ctx->Tensor4ArgNameAndIndex("x", 0);
    const Tensor* dy_tensor = ctx->Tensor4ArgNameAndIndex("dy", 0);
    Tensor* dx_tensor = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const T* x_ptr = x_tensor->dptr<T>();
    const T* dy_ptr = dy_tensor->dptr<T>();
    T* dx_ptr = dx_tensor->mut_dptr<T>();
    const T exponent = static_cast<T>(ctx->Attr<double>("exponent"));

    const int32_t elem_cnt = x_tensor->shape().elem_cnt();
    FOR_RANGE(int32_t, i, 0, elem_cnt) {
      dx_ptr[i] = exponent * (std::pow(x_ptr[i], exponent - static_cast<T>(1))) * dy_ptr[i];
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CPU_SCALAR_POW_GRAD_KERNEL(device, dtype)  \
  REGISTER_USER_KERNEL("scalar_pow_grad")                   \
      .SetCreateFn<CpuScalarPowGradKernel<device, dtype>>() \
      .SetIsMatchedHob((HobDeviceTag() == device)           \
                       & (HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_CPU_SCALAR_POW_GRAD_KERNEL(DeviceType::kCPU, float);
REGISTER_CPU_SCALAR_POW_GRAD_KERNEL(DeviceType::kCPU, double);

}  // namespace user_op

}  // namespace oneflow
