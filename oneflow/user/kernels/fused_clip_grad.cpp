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

template<DeviceType device_type>
class FusedClipGradKernel final : public user_op::OpKernel {
 public:
  FusedClipGradKernel() = default;
  ~FusedClipGradKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* grad = ctx->Tensor4ArgNameAndIndex("grad", 0);
    const float max_norm = ctx->Attr<float>("max_norm");
    const float norm_type = ctx->Attr<float>("norm_type");

    if (norm_type == 0) {

    } else if (norm_type == INFINITY) {

    } else if (norm_type == -INFINITY) {

    } else {

    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

}  // namespace

}  // namespace oneflow