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
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

namespace user_op {

template<DeviceType device_type, typename T>
class TransposeKernel final : public OpKernel {
 public:
  TransposeKernel() = default;
  ~TransposeKernel() = default;

 private:
  void Compute(KernelComputeContext* ctx) const override {
    const Tensor* tensor_in = ctx->Tensor4ArgNameAndIndex("input", 0);
    Tensor* tensor_out = ctx->Tensor4ArgNameAndIndex("output", 0);
    const auto& perm = ctx->Attr<std::vector<int32_t>>("perm");
    NewKernelUtil<device_type>::Transpose(ctx->device_ctx(), tensor_in->shape().NumAxes(),
                                          tensor_in->shape(), tensor_out->shape(),
                                          StdVec2PbRf(perm), tensor_in->shape().elem_cnt(),
                                          tensor_in->dptr<T>(), tensor_out->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_TRANSPOSE_KERNEL(device, dtype)                                         \
  REGISTER_USER_KERNEL("transpose")                                                      \
      .SetCreateFn<TransposeKernel<device, dtype>>()                                     \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                               \
                       & (user_op::HobDataType("input", 0) == GetDataType<dtype>::value) \
                       & (user_op::HobDataType("output", 0) == GetDataType<dtype>::value));

REGISTER_TRANSPOSE_KERNEL(DeviceType::kCPU, int8_t)
REGISTER_TRANSPOSE_KERNEL(DeviceType::kCPU, int32_t)
REGISTER_TRANSPOSE_KERNEL(DeviceType::kCPU, int64_t)
REGISTER_TRANSPOSE_KERNEL(DeviceType::kCPU, float)
REGISTER_TRANSPOSE_KERNEL(DeviceType::kCPU, double)

#ifdef WITH_CUDA
REGISTER_TRANSPOSE_KERNEL(DeviceType::kGPU, int8_t)
REGISTER_TRANSPOSE_KERNEL(DeviceType::kGPU, int32_t)
REGISTER_TRANSPOSE_KERNEL(DeviceType::kGPU, int64_t)
REGISTER_TRANSPOSE_KERNEL(DeviceType::kGPU, float)
REGISTER_TRANSPOSE_KERNEL(DeviceType::kGPU, double)
REGISTER_TRANSPOSE_KERNEL(DeviceType::kGPU, float16)
#endif

}  // namespace user_op
}  // namespace oneflow
