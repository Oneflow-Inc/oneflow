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

/*static*/ Maybe<void> ReluOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& in_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  FOR_RANGE(int64_t, i, 0, in_tensor.shape().NumAxes()) {
    ctx->NewBuilder().Split(user_op::OpArg("x", 0), i).Split(user_op::OpArg("y", 0), i).Build();
  }
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> ReluOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& in_shape = ctx->InputShape("x", 0);
  Shape* out_shape = ctx->MutOutputShape("y", 0);
  *out_shape = in_shape;
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> ReluOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> ReluOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("y", 0) = ctx->InputDType("x", 0);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> ReluGradOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& y_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("y", 0);
  FOR_RANGE(int64_t, i, 0, y_tensor.shape().NumAxes()) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("y", 0), i)
        .Split(user_op::OpArg("dy", 0), i)
        .Split(user_op::OpArg("dx", 0), i)
        .Build();
  }
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> ReluGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& y_shape = ctx->InputShape("y", 0);
  const Shape& dy_shape = ctx->InputShape("dy", 0);
  Shape* dx_shape = ctx->MutOutputShape("dx", 0);
  CHECK_OR_RETURN(dy_shape == y_shape)
      << Error::RuntimeError() << "Tensors y and dy must have the same shape";
  *dx_shape = dy_shape;
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> ReluGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> ReluGradOp::InferDataType(user_op::InferContext* ctx) {
  const DataType& data_type = ctx->InputDType("y", 0);
  CHECK_EQ_OR_RETURN(ctx->InputDType("dy", 0), data_type)
      << Error::TypeError() << "Tensors dy and y must have the same type";
  *ctx->OutputDType("dx", 0) = data_type;
  return Maybe<void>::Ok();
}

namespace {

REGISTER_USER_OP_GRAD("relu").SetBackwardOpConfGenFn(
    [](user_op::BackwardOpConfContext* ctx) -> Maybe<void> {
      const auto relu_grad_op_name = ctx->FwOp().op_name() + "_grad";
      ctx->DefineOp(relu_grad_op_name, [&ctx](user_op::BackwardOpBuilder& builder) {
        return builder.OpTypeName("relu_grad")
            .InputBind("y", ctx->FwOp().output("y", 0))
            .InputBind("dy", ctx->FwOp().output_grad("y", 0))
            .Output("dx")
            .Build();
      });
      ctx->FwOp().InputGradBind(user_op::OpArg("x", 0),
                                [&ctx, &relu_grad_op_name]() -> const std::string& {
                                  return ctx->GetOp(relu_grad_op_name).output("dx", 0);
                                });
      return Maybe<void>::Ok();
    });

}  // namespace

}  // namespace oneflow
