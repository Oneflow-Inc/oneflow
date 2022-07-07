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

/* static */ Maybe<void> SoftplusOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  *ctx->OutputShape("out", 0) = ctx->InputShape("in", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> SoftplusOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> SoftplusOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& in_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
  FOR_RANGE(int64_t, i, 0, in_tensor.shape().NumAxes()) {
    ctx->NewBuilder().Split(user_op::OpArg("in", 0), i).Split(user_op::OpArg("out", 0), i).Build();
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> SoftplusOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("out", 0) = ctx->InputDType("in", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> SoftplusGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& x_shape = ctx->InputShape("x", 0);
  const Shape& dy_shape = ctx->InputShape("dy", 0);
  Shape* dx_shape = ctx->OutputShape("dx", 0);
  CHECK_OR_RETURN(dy_shape == x_shape) << Error::RuntimeError() << "The size of dy " << dy_shape
                                       << " must match the size of x " << x_shape;
  *dx_shape = dy_shape;
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> SoftplusGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> SoftplusGradOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& x_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  FOR_RANGE(int64_t, i, 0, x_tensor.shape().NumAxes()) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("x", 0), i)
        .Split(user_op::OpArg("dy", 0), i)
        .Split(user_op::OpArg("dx", 0), i)
        .Build();
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> SoftplusGradOp::InferDataType(user_op::InferContext* ctx) {
  CHECK_EQ_OR_RETURN(ctx->InputDType("dy", 0), ctx->InputDType("x", 0))
      << Error::TypeError() << "dy and x are expected to have the same dtype, but found "
      << DataType_Name(ctx->InputDType("dy", 0)) << " and "
      << DataType_Name(ctx->InputDType("x", 0));
  *ctx->OutputDType("dx", 0) = ctx->InputDType("x", 0);
  return Maybe<void>::Ok();
}

REGISTER_USER_OP_GRAD("softplus")
    .SetBackwardOpConfGenFn([](user_op::BackwardOpConfContext* ctx) -> Maybe<void> {
      const auto softplus_grad_op_name = ctx->FwOp().op_name() + "_grad";
      ctx->DefineOp(softplus_grad_op_name, [&ctx](user_op::BackwardOpBuilder& builder) {
        return builder.OpTypeName("softplus_grad")
            .InputBind("x", ctx->FwOp().input("in", 0))
            .InputBind("dy", ctx->FwOp().output_grad("out", 0))
            .Attr<double>("beta", ctx->FwOp().attr<double>("beta"))
            .Attr<double>("threshold", ctx->FwOp().attr<double>("threshold"))
            .Output("dx")
            .Build();
      });
      ctx->FwOp().InputGradBind(user_op::OpArg("in", 0),
                                [&ctx, &softplus_grad_op_name]() -> const std::string& {
                                  return ctx->GetOp(softplus_grad_op_name).output("dx", 0);
                                });
      return Maybe<void>::Ok();
    });
}  // namespace oneflow
