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
#include "oneflow/user/kernels/slice_util.h"

namespace oneflow {

namespace {

template<DeviceType device_type, typename T>
class SliceKernel final : public user_op::OpKernel {
 public:
  SliceKernel() = default;
  ~SliceKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x_tensor = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y_tensor = ctx->Tensor4ArgNameAndIndex("y", 0);
    SliceParams params = ConstructSliceParams(ctx, x_tensor, y_tensor);
    SliceKernelUtil<device_type, T>::Forward(ctx->device_ctx(), params, x_tensor->dptr<T>(),
                                             y_tensor->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<DeviceType device_type, typename T>
class SliceGradKernel final : public user_op::OpKernel {
 public:
  SliceGradKernel() = default;
  ~SliceGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy_tensor = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx_tensor = ctx->Tensor4ArgNameAndIndex("dx", 0);
    SliceParams params = ConstructSliceParams(ctx, dx_tensor, dy_tensor);
    SliceKernelUtil<device_type, T>::Backward(ctx->device_ctx(), params, dy_tensor->dptr<T>(),
                                              dx_tensor->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace

#define REGISTER_SLICE_KERNELS(device, dtype)                                              \
  REGISTER_USER_KERNEL("slice").SetCreateFn<SliceKernel<device, dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == device)                                                 \
      & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));                      \
  REGISTER_USER_KERNEL("slice_grad")                                                       \
      .SetCreateFn<SliceGradKernel<device, dtype>>()                                       \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                                \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

#define REGISTER_SLICE_KERNELS_FOR_DEVICES(dtype) \
  REGISTER_SLICE_KERNELS(DeviceType::kCPU, dtype) \
  REGISTER_SLICE_KERNELS(DeviceType::kGPU, dtype)

REGISTER_SLICE_KERNELS_FOR_DEVICES(float)
REGISTER_SLICE_KERNELS_FOR_DEVICES(double)
REGISTER_SLICE_KERNELS_FOR_DEVICES(int32_t)
REGISTER_SLICE_KERNELS_FOR_DEVICES(int64_t)
REGISTER_SLICE_KERNELS_FOR_DEVICES(int8_t)
REGISTER_SLICE_KERNELS_FOR_DEVICES(uint8_t)

}  // namespace oneflow
