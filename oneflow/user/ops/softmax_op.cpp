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

/*static*/ Maybe<void> SoftmaxOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& in_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
  FOR_RANGE(int64_t, axis, 0, in_tensor.shape().NumAxes() - 1) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("in", 0), axis)
        .Split(user_op::OpArg("out", 0), axis)
        .Build();
  }
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SoftmaxOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  *ctx->MutOutputShape("out", 0) = ctx->InputShape("in", 0);
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SoftmaxOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> SoftmaxOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("out", 0) = ctx->InputDType("in", 0);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> SoftmaxGradOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& y_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("y", 0);
  FOR_RANGE(int64_t, axis, 0, y_tensor.shape().NumAxes() - 1) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("y", 0), axis)
        .Split(user_op::OpArg("dy", 0), axis)
        .Split(user_op::OpArg("dx", 0), axis)
        .Build();
  }
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SoftmaxGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& y_shape = ctx->InputShape("y", 0);
  const Shape& dy_shape = ctx->InputShape("dy", 0);
  Shape* dx_shape = ctx->MutOutputShape("dx", 0);
  CHECK_OR_RETURN(dy_shape == y_shape) << Error::RuntimeError() << "The size of dy " << dy_shape
                                       << " must match the size of y " << y_shape;
  *dx_shape = dy_shape;
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SoftmaxGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> SoftmaxGradOp::InferDataType(user_op::InferContext* ctx) {
  CHECK_EQ_OR_RETURN(ctx->InputDType("dy", 0), ctx->InputDType("y", 0))
      << Error::TypeError() << "dy and y are expected to have the same dtype, but found "
      << DataType_Name(ctx->InputDType("dy", 0)) << " and "
      << DataType_Name(ctx->InputDType("y", 0));
  *ctx->OutputDType("dx", 0) = ctx->InputDType("y", 0);
  return Maybe<void>::Ok();
}

namespace {

REGISTER_USER_OP_GRAD("softmax").SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                                                           user_op::AddOpFn AddOp) -> Maybe<void> {
  if (op.NeedGenGradTensor4OpInput("in", 0)) {
    user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
    user_op::UserOpConfWrapper softmax_grad_op =
        builder.Op("softmax_grad")
            .Input("y", op.output("out", 0))
            .Input("dy", op.GetGradTensorWithOpOutput("out", 0))
            .Output("dx")
            .Build();
    op.BindGradTensorWithOpInput(softmax_grad_op.output("dx", 0), "in", 0);
    AddOp(softmax_grad_op);
  }
  return Maybe<void>::Ok();
});

}  // namespace

}  // namespace oneflow
