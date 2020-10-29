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
namespace oneflow {

namespace {

template<typename T>
__global__ void RangeForwardGpuKernel(const int start, const int delta, const int range_shape,
                                      T *out) {
  // Use Loop to set the value
  CUDA_1D_KERNEL_LOOP(i, range_shape) { out[i] = start + i * delta; }
}
}  // namespace

template<typename T>
class RangeGpuKernel final : public user_op::OpKernel {
 public:
  RangeGpuKernel() = default;
  ~RangeGpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext *ctx) const override {
    const int64_t start = ctx->Attr<int64_t>("start");
    const int64_t delta = ctx->Attr<int64_t>("delta");
    const int64_t range_shape = ctx->Attr<int64_t>("range_shape");
    // Get out tensor
    user_op::Tensor *out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    // Run cuda range forward kernel
    // The thread num is set as range_shape
    RUN_CUDA_KERNEL(RangeForwardGpuKernel, ctx->device_ctx(), range_shape, start, delta,
                    range_shape, out_tensor->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GPU_RANGE_KERNEL(device, dtype)                                              \
  REGISTER_USER_KERNEL("range").SetCreateFn<RangeGpuKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceTag() == device)                                              \
      & (user_op::HobAttr<DataType>("dtype") == GetDataType<dtype>::value));

REGISTER_GPU_RANGE_KERNEL(DeviceType::kGPU, int32_t)
REGISTER_GPU_RANGE_KERNEL(DeviceType::kGPU, int64_t)
REGISTER_GPU_RANGE_KERNEL(DeviceType::kGPU, float)

}  // namespace oneflow
