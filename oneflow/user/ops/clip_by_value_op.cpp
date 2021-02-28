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

Maybe<void> InferClipTensorDesc(user_op::InferContext* ctx) {
  *ctx->Shape4ArgNameAndIndex("y", 0) = *ctx->Shape4ArgNameAndIndex("x", 0);
  *ctx->Dtype4ArgNameAndIndex("y", 0) = *ctx->Dtype4ArgNameAndIndex("x", 0);
  return Maybe<void>::Ok();
}

Maybe<void> GetClipSbpSignature(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& x_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  FOR_RANGE(int64_t, i, 0, x_tensor.shape().NumAxes()) {
    ctx->NewBuilder().Split(user_op::OpArg("x", 0), i).Split(user_op::OpArg("y", 0), i).Build();
  }
  return Maybe<void>::Ok();
}

Maybe<void> InferClipGradTensorDesc(user_op::InferContext* ctx) {
  *ctx->Shape4ArgNameAndIndex("dx", 0) = *ctx->Shape4ArgNameAndIndex("x", 0);
  *ctx->Dtype4ArgNameAndIndex("dx", 0) = *ctx->Dtype4ArgNameAndIndex("x", 0);
  return Maybe<void>::Ok();
}

Maybe<void> GetClipGradSbpSignature(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& x_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  FOR_RANGE(int64_t, i, 0, x_tensor.shape().NumAxes()) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("dy", 0), i)
        .Split(user_op::OpArg("x", 0), i)
        .Split(user_op::OpArg("dx", 0), i)
        .Build();
  }
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("x", 0))
      .PartialSum(user_op::OpArg("dy", 0))
      .PartialSum(user_op::OpArg("dx", 0))
      .Build();
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_USER_OP("clip_by_scalar")
    .Input("x")
    .Attr<double>("floating_min")
    .Attr<int64_t>("integral_min")
    .Attr<double>("floating_max")
    .Attr<int64_t>("integral_max")
    .Output("y")
    .SetTensorDescInferFn(InferClipTensorDesc)
    .SetGetSbpFn(GetClipSbpSignature);

REGISTER_USER_OP("clip_by_scalar_min")
    .Input("x")
    .Attr<double>("floating_min")
    .Attr<int64_t>("integral_min")
    .Output("y")
    .SetTensorDescInferFn(InferClipTensorDesc)
    .SetGetSbpFn(GetClipSbpSignature);

REGISTER_USER_OP("clip_by_scalar_max")
    .Input("x")
    .Attr<double>("floating_max")
    .Attr<int64_t>("integral_max")
    .Output("y")
    .SetTensorDescInferFn(InferClipTensorDesc)
    .SetGetSbpFn(GetClipSbpSignature);

REGISTER_USER_OP("clip_by_scalar_grad")
    .Input("dy")
    .Input("x")
    .Attr<double>("floating_min")
    .Attr<int64_t>("integral_min")
    .Attr<double>("floating_max")
    .Attr<int64_t>("integral_max")
    .Output("dx")
    .SetTensorDescInferFn(InferClipGradTensorDesc)
    .SetGetSbpFn(GetClipGradSbpSignature);

REGISTER_USER_OP("clip_by_scalar_min_grad")
    .Input("dy")
    .Input("x")
    .Attr<double>("floating_min")
    .Attr<int64_t>("integral_min")
    .Output("dx")
    .SetTensorDescInferFn(InferClipGradTensorDesc)
    .SetGetSbpFn(GetClipGradSbpSignature);

REGISTER_USER_OP("clip_by_scalar_max_grad")
    .Input("dy")
    .Input("x")
    .Attr<double>("floating_max")
    .Attr<int64_t>("integral_max")
    .Output("dx")
    .SetTensorDescInferFn(InferClipGradTensorDesc)
    .SetGetSbpFn(GetClipGradSbpSignature);

REGISTER_USER_OP_GRAD("clip_by_scalar")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op =
            builder.Op("clip_by_scalar_grad")
                .Attr("floating_min", op.attr<double>("floating_min"))
                .Attr("integral_min", op.attr<int64_t>("integral_min"))
                .Attr("floating_max", op.attr<double>("floating_max"))
                .Attr("integral_max", op.attr<int64_t>("integral_max"))
                .Input("dy", op.GetGradTensorWithOpOutput("y", 0))
                .Input("x", op.input("x", 0))
                .Output("dx")
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "x", 0);
        AddOp(grad_op);
      }
    });

REGISTER_USER_OP_GRAD("clip_by_scalar_min")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op =
            builder.Op("clip_by_scalar_min_grad")
                .Attr("floating_min", op.attr<double>("floating_min"))
                .Attr("integral_min", op.attr<int64_t>("integral_min"))
                .Input("dy", op.GetGradTensorWithOpOutput("y", 0))
                .Input("x", op.input("x", 0))
                .Output("dx")
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "x", 0);
        AddOp(grad_op);
      }
    });

REGISTER_USER_OP_GRAD("clip_by_scalar_max")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op, user_op::AddOpFn AddOp) {
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
        user_op::UserOpConfWrapper grad_op =
            builder.Op("clip_by_scalar_max_grad")
                .Attr("floating_max", op.attr<double>("floating_max"))
                .Attr("integral_max", op.attr<int64_t>("integral_max"))
                .Input("dy", op.GetGradTensorWithOpOutput("y", 0))
                .Input("x", op.input("x", 0))
                .Output("dx")
                .Build();
        op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "x", 0);
        AddOp(grad_op);
      }
    });

}  // namespace oneflow
