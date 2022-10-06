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
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/common/singleton.h"

namespace oneflow {

namespace {

template<DeviceType device_type>
class IdentityEvalKernel final : public user_op::OpKernel {
 public:
  IdentityEvalKernel() = default;
  ~IdentityEvalKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    UNIMPLEMENTED();
 }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return true; }
};

#define REGISTER_IDENTITY_KERNEL(op, device)                                                    \
  REGISTER_USER_KERNEL(op)                                                                      \
      .SetCreateFn<IdentityEvalKernel<device>>()                                                \
      .SetIsMatchedHob(user_op::HobDeviceType() == device)                                      \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, false));                      \
        return Maybe<void>::Ok();                                                               \
      });

REGISTER_IDENTITY_KERNEL("identity_eval", DeviceType::kCPU)  // NOLINT
#ifdef WITH_CUDA
REGISTER_IDENTITY_KERNEL("identity_eval", DeviceType::kCUDA)  // NOLINT
#endif

}  // namespace

}  // namespace oneflow