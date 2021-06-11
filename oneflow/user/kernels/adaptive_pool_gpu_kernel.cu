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
#include "oneflow/core/kernel/util/cuda_half_util.h"
#include "oneflow/core/cuda/elementwise.cuh"

namespace oneflow {

namespace user_op {


template<DeviceType device_type, typename T>
class GpuAdaptiveAvgPool2dKernel final : public OpKernel {
 public:
  GpuAdaptiveAvgPool2dKernel() = default;
  ~GpuAdaptiveAvgPool2dKernel() = default;

 private:
  void Compute(KernelComputeContext* ctx) const override {
    const Tensor* x_tensor = ctx->Tensor4ArgNameAndIndex("x", 0);
    Tensor* y_tensor = ctx->Tensor4ArgNameAndIndex("y", 0);
    
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GPU_ELU_KERNEL(device, dtype)                                            \
  REGISTER_USER_KERNEL("adaptive_avg_pool2d").SetCreateFn<GpuAdaptiveAvgPool2dKernel<device, dtype>>().SetIsMatchedHob( \
      (HobDeviceTag() == device) & (HobDataType("y", 0) == GetDataType<dtype>::value));

REGISTER_GPU_ELU_KERNEL(DeviceType::kGPU, half);
REGISTER_GPU_ELU_KERNEL(DeviceType::kGPU, float);
REGISTER_GPU_ELU_KERNEL(DeviceType::kGPU, double);

template<DeviceType device_type, typename T>
class GpuEluGradKernel final : public OpKernel {
 public:
  GpuEluGradKernel() = default;
  ~GpuEluGradKernel() = default;

 private:
  void Compute(KernelComputeContext* ctx) const override {
    const Tensor* x_tensor = ctx->Tensor4ArgNameAndIndex("x", 0);
    const Tensor* dy_tensor = ctx->Tensor4ArgNameAndIndex("dy", 0);
    Tensor* dx_tensor = ctx->Tensor4ArgNameAndIndex("dx", 0);
    
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GPU_ELU_BACKWARD_KERNEL(device, dtype) \
  REGISTER_USER_KERNEL("elu_grad")                      \
      .SetCreateFn<GpuEluGradKernel<device, dtype>>()   \
      .SetIsMatchedHob((HobDeviceTag() == device)       \
                       & (HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_GPU_ELU_BACKWARD_KERNEL(DeviceType::kGPU, half);
REGISTER_GPU_ELU_BACKWARD_KERNEL(DeviceType::kGPU, float);
REGISTER_GPU_ELU_BACKWARD_KERNEL(DeviceType::kGPU, double);

}  // namespace user_op

}  // namespace oneflow
