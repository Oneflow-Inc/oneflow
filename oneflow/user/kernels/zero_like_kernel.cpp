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

namespace oneflow {

template<DeviceType device_type>
class ZeroLikeKernel final : public user_op::OpKernel {
 public:
  ZeroLikeKernel() = default;
  ~ZeroLikeKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    Memset<device_type>(ctx->device_ctx(), out->mut_dptr(), 0,
                        out->shape().elem_cnt() * GetSizeOfDataType(out->data_type()));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_ZERO_LIKE_KERNEL(device_type_v)    \
  REGISTER_USER_KERNEL("zero_like")                 \
      .SetCreateFn<ZeroLikeKernel<device_type_v>>() \
      .SetIsMatchedHob(user_op::HobDeviceTag() == device_type_v);

REGISTER_ZERO_LIKE_KERNEL(DeviceType::kCPU)
#ifdef WITH_CUDA
REGISTER_ZERO_LIKE_KERNEL(DeviceType::kGPU)
#endif

}  // namespace oneflow
