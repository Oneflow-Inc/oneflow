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
#include "oneflow/core/common/scalar.h"
#include "oneflow/user/kernels/fused_clip_grad.h"

namespace oneflow {

namespace {

template<DeviceType device_type, typename T>
class FusedClipGradKernel final : public user_op::OpKernel {
 public:
  FusedClipGradKernel() = default;
  ~FusedClipGradKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const int32_t input_size = ctx->input_size("grad");
    const float max_norm = ctx->Attr<float>("max_norm");
    const float norm_type = ctx->Attr<float>("norm_type");


    for (int32_t i = 0; i < input_size; ++i) {
      const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("grad", i);
    
      if (norm_type == 0) {

      } else if (norm_type == INFINITY) {

      } else if (norm_type == -INFINITY) {

      } else {

      }
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

#define REGISTER_FUSED_CLIP_GRAD_KERNEL(device, dtype)                             \
  REGISTER_USER_KERNEL("fused_clip_grad")                                                 \
      .SetCreateFn<FusedClipGradKernel<device, dtype>>()                   \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                               \
                       && (user_op::HobDataType("grad", 0) == GetDataType<dtype>::value));

#ifdef WITH_CUDA
REGISTER_FUSED_CLIP_GRAD_KERNEL(DeviceType::kCUDA, float);
#endif

}  // namespace

}  // namespace oneflow