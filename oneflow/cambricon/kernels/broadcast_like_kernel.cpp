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
#include "oneflow/cambricon/common/mlu_util.h"
#include "oneflow/cambricon/ep/mlu_stream.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/cambricon/cnnl/cnnl_tensor_descriptor.h"

namespace oneflow {

class MluBroadcastLikeKernel final : public user_op::OpKernel {
 public:
  MluBroadcastLikeKernel() = default;
  ~MluBroadcastLikeKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* like_tensor = ctx->Tensor4ArgNameAndIndex("like", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const auto& axis = ctx->Attr<std::vector<int32_t>>("broadcast_axes");

    const auto y_shape = y->shape_view();

    // handle 0-size tensor
    if (std::any_of(y_shape.begin(), y_shape.end(), [](int64_t dim) { return dim == 0; })) {
      return;
    }

    const Shape& reduced_shape =
        CreateReducedShapeOrOnesShape(like_tensor->shape_view(), {axis.begin(), axis.end()});

    CnnlTensorDescriptor in_desc(x), y_desc(y);
    std::vector<int> reduced_dims_input =
        std::vector<int>(reduced_shape.begin(), reduced_shape.end());
    in_desc.set_reshape(x, reduced_dims_input);
    OF_CNNL_CHECK(cnnlExpand(ctx->stream()->As<ep::MluStream>()->cnnl_handle(), in_desc.desc(),
                             x->dptr(), y_desc.desc(), y->mut_dptr()));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("broadcast_like")
    .SetCreateFn<MluBroadcastLikeKernel>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kMLU));

}  // namespace oneflow
