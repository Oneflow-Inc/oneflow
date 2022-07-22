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
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

/*static*/ Maybe<void> TfPreluOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& x_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  const user_op::TensorDesc& alpha_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("alpha", 0);
  ctx->NewBuilder()
      .Split(user_op::OpArg("x", 0), 0)
      .Broadcast(user_op::OpArg("alpha", 0))
      .Split(user_op::OpArg("y", 0), 0)
      .Build();
  FOR_RANGE(int64_t, i, 1, x_tensor.shape().NumAxes()) {
    if (x_tensor.shape().At(i) == alpha_tensor.shape().At(i - 1)) {
      ctx->NewBuilder()
          .Split(user_op::OpArg("x", 0), i)
          .Split(user_op::OpArg("alpha", 0), i - 1)
          .Split(user_op::OpArg("y", 0), i)
          .Build();
    }
  }
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> TfPreluOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& x_desc = ctx->InputTensorDesc("x", 0);
  user_op::TensorDesc* y_desc = ctx->OutputTensorDesc("y", 0);
  const Shape& alpha_shape = ctx->InputShape("alpha", 0);
  CHECK_EQ_OR_RETURN(x_desc.shape().NumAxes(), alpha_shape.NumAxes() + 1);
  FOR_RANGE(int64_t, i, 1, x_desc.shape().NumAxes()) {
    CHECK_OR_RETURN((alpha_shape.At(i - 1) == x_desc.shape().At(i))
                    || (alpha_shape.At(i - 1) == 1));
  }
  *y_desc->mut_shape() = x_desc.shape();
  *y_desc->mut_is_dynamic() = x_desc.is_dynamic();
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> TfPreluOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> TfPreluOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("y", 0) = ctx->InputDType("x", 0);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> TfPreluGradOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& x_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  const user_op::TensorDesc& alpha_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("alpha", 0);
  ctx->NewBuilder()
      .Split(user_op::OpArg("dy", 0), 0)
      .Split(user_op::OpArg("x", 0), 0)
      .Broadcast(user_op::OpArg("alpha", 0))
      .Split(user_op::OpArg("dx", 0), 0)
      .PartialSum(user_op::OpArg("alpha_diff", 0))
      .Build();
  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("dy", 0))
      .Broadcast(user_op::OpArg("x", 0))
      .Broadcast(user_op::OpArg("alpha", 0))
      .PartialSum(user_op::OpArg("dx", 0))
      .PartialSum(user_op::OpArg("alpha_diff", 0))
      .Build();
  FOR_RANGE(int64_t, i, 1, x_tensor.shape().NumAxes()) {
    if (x_tensor.shape().At(i) == alpha_tensor.shape().At(i - 1)) {
      ctx->NewBuilder()
          .Split(user_op::OpArg("dy", 0), i)
          .Split(user_op::OpArg("x", 0), i)
          .Split(user_op::OpArg("alpha", 0), i - 1)
          .Split(user_op::OpArg("dx", 0), i)
          .Split(user_op::OpArg("alpha_diff", 0), i - 1)
          .Build();
    }
  }
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> TfPreluGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& x_desc = ctx->InputTensorDesc("x", 0);
  const user_op::TensorDesc& dy_desc = ctx->InputTensorDesc("dy", 0);
  user_op::TensorDesc* dx_desc = ctx->OutputTensorDesc("dx", 0);
  const user_op::TensorDesc& alpha_desc = ctx->InputTensorDesc("alpha", 0);
  CHECK_EQ_OR_RETURN(x_desc.shape().NumAxes(), alpha_desc.shape().NumAxes() + 1);
  FOR_RANGE(int64_t, i, 1, x_desc.shape().NumAxes()) {
    CHECK_OR_RETURN((alpha_desc.shape().At(i - 1) == x_desc.shape().At(i))
                    || (alpha_desc.shape().At(i - 1) == 1));
  }
  CHECK_EQ_OR_RETURN(dy_desc.shape(), x_desc.shape());
  CHECK_EQ_OR_RETURN(dy_desc.data_type(), x_desc.data_type());
  *dx_desc->mut_shape() = x_desc.shape();
  *dx_desc->mut_is_dynamic() = x_desc.is_dynamic();
  *ctx->MutOutputShape("alpha_diff", 0) = alpha_desc.shape();
  *ctx->OutputIsDynamic("alpha_diff", 0) = alpha_desc.is_dynamic();
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> TfPreluGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> TfPreluGradOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("dx", 0) = ctx->InputDType("x", 0);
  *ctx->OutputDType("alpha_diff", 0) = ctx->InputDType("alpha", 0);
  return Maybe<void>::Ok();
}

REGISTER_USER_OP_GRAD("tf_prelu")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                               user_op::AddOpFn AddOp) -> Maybe<void> {
      if (op.NeedGenGradTensor4OpInput("x", 0) || op.NeedGenGradTensor4OpInput("alpha", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op = builder.Op("tf_prelu_grad")
                                                 .Input("x", op.input("x", 0))
                                                 .Input("dy", op.GetGradTensorWithOpOutput("y", 0))
                                                 .Input("alpha", op.input("alpha", 0))
                                                 .Output("dx")
                                                 .Output("alpha_diff")
                                                 .Build();
        AddOp(grad_op);

        if (op.NeedGenGradTensor4OpInput("x", 0)) {
          op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "x", 0);
        }
        if (op.NeedGenGradTensor4OpInput("alpha", 0)) {
          auto alpha_identity_op =
              user_op::UserOpConfWrapperBuilder(op.op_name() + "_alpha_identity")
                  .Op("identity")
                  .Input("in", grad_op.output("alpha_diff", 0))
                  .Output("out")
                  .Build();
          AddOp(alpha_identity_op);
          op.BindGradTensorWithOpInput(alpha_identity_op.output("out", 0), "alpha", 0);
        }
      }
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
