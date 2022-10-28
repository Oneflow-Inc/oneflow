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
#include "oneflow/user/kernels/l1_l2_regularize_gradient_kernel_util.h"

namespace oneflow {

namespace {

template<DeviceType device_type, typename T>
class L1L2RegularizeGradientKernel final : public user_op::OpKernel {
 public:
  L1L2RegularizeGradientKernel() = default;
  ~L1L2RegularizeGradientKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* model = ctx->Tensor4ArgNameAndIndex("model", 0);
    const user_op::Tensor* model_diff = ctx->Tensor4ArgNameAndIndex("model_diff", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const auto l1 = ctx->Attr<float>("l1");
    const auto l2 = ctx->Attr<float>("l2");
    L1L2RegularizeGradientKernelUtil<device_type, T>::RegularizeGradient(
        ctx->stream(), out->shape_view().elem_cnt(), model->dptr<T>(), model_diff->dptr<T>(),
        out->mut_dptr<T>(), l1, l2);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_L1_L2_REGULARIZE_GRADIENT_KERNEL(device, dtype)                                \
  REGISTER_USER_KERNEL("l1_l2_regularize_gradient")                                             \
      .SetCreateFn<L1L2RegularizeGradientKernel<device, dtype>>()                               \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                                     \
                       && (user_op::HobDataType("out", 0) == GetDataType<dtype>::value))        \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "model_diff", 0, true));               \
        return Maybe<void>::Ok();                                                               \
      });

REGISTER_L1_L2_REGULARIZE_GRADIENT_KERNEL(DeviceType::kCPU, float)
REGISTER_L1_L2_REGULARIZE_GRADIENT_KERNEL(DeviceType::kCPU, double)
#ifdef WITH_CUDA
REGISTER_L1_L2_REGULARIZE_GRADIENT_KERNEL(DeviceType::kCUDA, float)
REGISTER_L1_L2_REGULARIZE_GRADIENT_KERNEL(DeviceType::kCUDA, double)
#endif

}  // namespace

}  // namespace oneflow
