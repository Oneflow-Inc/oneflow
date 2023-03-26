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
#include "oneflow/cambricon/bang/bang_kernels.h"
#include "oneflow/cambricon/common/mlu_util.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {

template<typename T>
class MluL1L2RegularizeGradientKernel final : public user_op::OpKernel {
 public:
  MluL1L2RegularizeGradientKernel() = default;
  ~MluL1L2RegularizeGradientKernel() = default;

 private:
  using user_op::OpKernel::Compute;

  // formula: out = model_diff + l1 * (model >= 0 ? 1 : -1) + l2 * model
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* model = ctx->Tensor4ArgNameAndIndex("model", 0);
    const user_op::Tensor* model_diff = ctx->Tensor4ArgNameAndIndex("model_diff", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const auto l1 = ctx->Attr<float>("l1");
    const auto l2 = ctx->Attr<float>("l2");

    auto* stream = ctx->stream()->As<ep::MluStream>();
    BangHandle handle(stream->mlu_stream(), stream->device()->nclusters(),
                      stream->device()->ncores_per_cluster());

    if constexpr (std::is_same<T, float16>::value) {
      bang_regularize_gradient_half_kernel(handle, out->shape_view().elem_cnt(),
                                           model->dptr<float16>(), model_diff->dptr<float16>(),
                                           out->mut_dptr<float16>(), l1, l2);
    } else {
      bang_regularize_gradient_kernel(handle, out->shape_view().elem_cnt(), model->dptr<T>(),
                                      model_diff->dptr<T>(), out->mut_dptr<T>(), l1, l2);
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_L1_L2_REGULARIZE_GRADIENT_MLU_KERNEL(dtype)                                    \
  REGISTER_USER_KERNEL("l1_l2_regularize_gradient")                                             \
      .SetCreateFn<MluL1L2RegularizeGradientKernel<dtype>>()                                    \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU)                           \
                       && (user_op::HobDataType("model", 0) == GetDataType<dtype>::value))      \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "model_diff", 0, true));               \
        return Maybe<void>::Ok();                                                               \
      });

REGISTER_L1_L2_REGULARIZE_GRADIENT_MLU_KERNEL(float)
REGISTER_L1_L2_REGULARIZE_GRADIENT_MLU_KERNEL(float16)

#undef REGISTER_L1_L2_REGULARIZE_GRADIENT_MLU_KERNEL

}  // namespace oneflow
