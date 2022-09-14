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
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/ep/include/primitive/permute.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/ep/include/primitive/matmul.h"
#include "oneflow/core/ep/include/primitive/memset.h"

namespace oneflow {

namespace {

template<typename T>
class DeformableConv2dCudaKernel final : public user_op::OpKernel {
 public:
  DeformableConv2dCudaKernel() = default;
  ~DeformableConv2dCudaKernel() = default;

 private:
  using user_op::OpKernel::Compute;

  void Compute(user_op::KernelComputeContext* ctx) const override {}
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_DEFORM_CONV2D_GPU_KERNEL(dtype)                       \
  REGISTER_USER_KERNEL("deform_conv2d")                                \
      .SetCreateFn<DeformableConv2dCudaKernel<dtype>>()                \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("input", 0) == GetDataType<dtype>::value));

REGISTER_DEFORM_CONV2D_GPU_KERNEL(float)
REGISTER_DEFORM_CONV2D_GPU_KERNEL(double)

}  // namespace
}  // namespace oneflow