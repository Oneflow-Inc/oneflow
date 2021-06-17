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
#include "oneflow/user/ops/nn_util.h"

namespace oneflow {

namespace {

Maybe<void> InferFWTensorDesc(user_op::InferContext* ctx) {
  std::vector<int64_t> output_size = ctx->Attr<std::vector<int64_t>>("output_size");
  const Shape& x_shape = ctx->InputShape("x", 0);
  int h = 0;
  int w = 0;
  if (output_size.size() >= 1) {
    h = output_size[0];  // h
    w = output_size[0];
    if (output_size.size() == 2) {
      w = output_size[1];  // w
    }
  }
  DimVector out_shape(4);
  out_shape[0] = x_shape.dim_vec()[0];
  out_shape[1] = x_shape.dim_vec()[1];
  out_shape[2] = h;
  out_shape[3] = w;

  *ctx->OutputShape("y", 0) = Shape(out_shape);
  return Maybe<void>::Ok();
}

Maybe<void> InferBWTensorDesc(user_op::InferContext* ctx) {
  *ctx->OutputShape("dx", 0) = ctx->InputShape("x", 0);
  *ctx->OutputIsDynamic4ArgNameAndIndex("dx", 0) = ctx->InputIsDynamic4ArgNameAndIndex("x", 0);
  return Maybe<void>::Ok();
}

Maybe<void> FwGetSbpFn(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  // only for nchw
  FOR_RANGE(int64_t, i, 0, std::min(2, (int)tensor.shape().NumAxes())) {
    ctx->NewBuilder().Split(user_op::OpArg("x", 0), i).Split(user_op::OpArg("y", 0), i).Build();
  }
  return Maybe<void>::Ok();
}

Maybe<void> BwGetSbpFn(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  FOR_RANGE(int64_t, i, 0, std::min(2, (int)tensor.shape().NumAxes())) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("x", 0), i)
        .Split(user_op::OpArg("y", 0), i)
        .Split(user_op::OpArg("dy", 0), i)
        .Split(user_op::OpArg("dx", 0), i)
        .Build();
  }
  return Maybe<void>::Ok();
}

Maybe<void> InferFWDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("y", 0) = *ctx->Dtype4ArgNameAndIndex("x", 0);
  return Maybe<void>::Ok();
}

Maybe<void> InferBWDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("dx", 0) = *ctx->Dtype4ArgNameAndIndex("x", 0);
  return Maybe<void>::Ok();
}

REGISTER_USER_OP("adaptive_avg_pool2d")
    .Input("x")
    .Attr<std::vector<int64_t>>("output_size")
    .Output("y")
    .SetTensorDescInferFn(InferFWTensorDesc)
    .SetGetSbpFn(FwGetSbpFn)
    .SetDataTypeInferFn(InferFWDataType);

REGISTER_USER_OP("adaptive_avg_pool2d_grad")
    .Input("x")
    .Input("dy")
    .Attr<std::vector<int64_t>>("output_size")
    .Output("dx")
    .SetTensorDescInferFn(InferBWTensorDesc)
    .SetGetSbpFn(BwGetSbpFn)
    .SetDataTypeInferFn(InferBWDataType);

REGISTER_USER_OP_GRAD("adaptive_avg_pool2d")
    .SetBackwardOpConfGenFn([](user_op::BackwardOpConfContext* ctx) {
      const auto adaptive_avg_pool2d_grad_op_name = ctx->FwOp().op_name() + "_grad";
      ctx->DefineOp(adaptive_avg_pool2d_grad_op_name, [&ctx](user_op::BackwardOpBuilder& builder) {
        return builder.OpTypeName("adaptive_avg_pool2d_grad")
            .InputBind("x", ctx->FwOp().input("x", 0))
            .InputBind("dy", ctx->FwOp().output_grad("y", 0))
            .Output("dx")
            .Build();
      });
      ctx->FwOp().InputGradBind(
          user_op::OpArg("x", 0),
          [&ctx, &adaptive_avg_pool2d_grad_op_name]() -> const std::string& {
            return ctx->GetOp(adaptive_avg_pool2d_grad_op_name).output("dx", 0);
          });
    });

}  // namespace

}  // namespace oneflow
