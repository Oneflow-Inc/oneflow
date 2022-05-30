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
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

/*static*/ Maybe<void> PadOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& x_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  const auto& padding_before = ctx->Attr<std::vector<int64_t>>("padding_before");
  const auto& padding_after = ctx->Attr<std::vector<int64_t>>("padding_after");
  FOR_RANGE(int64_t, i, 0, x_tensor.shape().NumAxes()) {
    if (padding_before[i] == 0 && padding_after[i] == 0) {
      ctx->NewBuilder().Split(user_op::OpArg("x", 0), i).Split(user_op::OpArg("y", 0), i).Build();
    }
  }
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> PadOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& x_shape = ctx->InputShape("x", 0);
  const auto& padding_before = ctx->Attr<std::vector<int64_t>>("padding_before");
  const auto& padding_after = ctx->Attr<std::vector<int64_t>>("padding_after");
  CHECK_EQ_OR_RETURN(padding_before.size(), x_shape.NumAxes());
  CHECK_EQ_OR_RETURN(padding_after.size(), x_shape.NumAxes());
  DimVector y_dim_vec(x_shape.NumAxes());
  FOR_RANGE(int64_t, i, 0, x_shape.NumAxes()) {
    y_dim_vec[i] = x_shape.At(i) + padding_before[i] + padding_after[i];
  }
  *ctx->OutputShape("y", 0) = Shape(y_dim_vec);
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> PadOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return PadOp::InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> PadOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("y", 0) = ctx->InputDType("x", 0);
  return Maybe<void>::Ok();
}

REGISTER_USER_OP_GRAD("pad").SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                                                       user_op::AddOpFn AddOp) -> Maybe<void> {
  if (op.NeedGenGradTensor4OpInput("x", 0)) {
    user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
    std::vector<int64_t> padding_before = op.attr<std::vector<int64_t>>("padding_before");
    std::vector<int64_t> padding_after = op.attr<std::vector<int64_t>>("padding_after");
    for (int i = 0; i < padding_before.size(); i++) {
      padding_before[i] = -padding_before[i];
      padding_after[i] = -padding_after[i];
    }
    user_op::UserOpConfWrapper grad_op =
        builder.Op("pad")
            .Input("x", op.GetGradTensorWithOpOutput("y", 0))
            .Output("y")
            .Attr("floating_constant_value", static_cast<double>(0.0))
            .Attr("integral_constant_value", static_cast<int64_t>(0))
            .Attr("padding_before", padding_before)
            .Attr("padding_after", padding_after)
            .Build();
    op.BindGradTensorWithOpInput(grad_op.output("y", 0), "x", 0);
    AddOp(grad_op);
  }
  return Maybe<void>::Ok();
});

}  // namespace oneflow
