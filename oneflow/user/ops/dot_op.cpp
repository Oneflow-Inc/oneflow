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

namespace oneflow {

namespace {

REGISTER_USER_OP("dot")
    .Input("x")
    .Input("y")
    .Output("out")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& x = ctx->InputTensorDesc("x", 0);
      const user_op::TensorDesc& y = ctx->InputTensorDesc("y", 0);
      CHECK_OR_RETURN(x.shape() == y.shape()) << "Input tensor shape is different";
      CHECK_OR_RETURN(x.shape().NumAxes() == 1) << "Input tensor is not 1D";
      *ctx->OutputShape("out", 0) = Shape({1});
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder()
          .Split(user_op::OpArg("x", 0), 0)
          .Split(user_op::OpArg("y", 0), 0)
          .PartialSum(user_op::OpArg("out", 0))
          .Build();

      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& x = ctx->InputTensorDesc("x", 0);
      const user_op::TensorDesc& y = ctx->InputTensorDesc("y", 0);
      CHECK_OR_RETURN(x.data_type() == y.data_type()) << "The input tensor type is different";
      *ctx->OutputDType("out", 0) = ctx->InputDType("x", 0);
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("dot_grad")
    .Input("x")
    .Input("y")
    .Input("dout")
    .Output("dx")
    .Output("dy")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& x = ctx->InputTensorDesc("x", 0);
      const user_op::TensorDesc& y = ctx->InputTensorDesc("y", 0);
      const user_op::TensorDesc& dout = ctx->InputTensorDesc("dout", 0);
      CHECK_OR_RETURN(dout.shape() == Shape({1}));
      CHECK_OR_RETURN(x.shape() == y.shape()) << "Input tensor shape is different";
      CHECK_OR_RETURN(x.shape().NumAxes() == 1) << "Input tensor is not 1D";
      *ctx->OutputShape("dx", 0) = ctx->InputShape("x", 0);
      *ctx->OutputShape("dy", 0) = ctx->InputShape("y", 0);
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder()
          .Split(user_op::OpArg("x", 0), 0)
          .Split(user_op::OpArg("y", 0), 0)
          .Broadcast(user_op::OpArg("dout", 0))
          .Split(user_op::OpArg("dx", 0), 0)
          .Split(user_op::OpArg("dy", 0), 0)
          .Build();
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& x = ctx->InputTensorDesc("x", 0);
      const user_op::TensorDesc& y = ctx->InputTensorDesc("y", 0);
      const user_op::TensorDesc& dout = ctx->InputTensorDesc("dout", 0);
      CHECK_OR_RETURN((x.data_type() == y.data_type()) & (dout.data_type() == y.data_type()))
          << "The input tensor type is different";
      *ctx->OutputDType("dx", 0) = ctx->InputDType("x", 0);
      *ctx->OutputDType("dy", 0) = ctx->InputDType("y", 0);
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP_GRAD("dot").SetBackwardOpConfGenFn([](user_op::BackwardOpConfContext* ctx) {
  const auto dot_grad_op_name = ctx->FwOp().op_name() + "_grad";
  ctx->DefineOp(dot_grad_op_name, [&ctx](user_op::BackwardOpBuilder& builder) {
    return builder.OpTypeName("dot_grad")
        .InputBind("x", ctx->FwOp().input("x", 0))
        .InputBind("y", ctx->FwOp().input("y", 0))
        .InputBind("dout", ctx->FwOp().output_grad("out", 0))
        .Output("dx")
        .Output("dy")
        .Build();
  });
  ctx->FwOp().InputGradBind(user_op::OpArg("x", 0),
                            [&ctx, &dot_grad_op_name]() -> const std::string& {
                              return ctx->GetOp(dot_grad_op_name).output("dx", 0);
                            });
  ctx->FwOp().InputGradBind(user_op::OpArg("y", 0),
                            [&ctx, &dot_grad_op_name]() -> const std::string& {
                              return ctx->GetOp(dot_grad_op_name).output("dy", 0);
                            });
});

}  // namespace

}  // namespace oneflow
