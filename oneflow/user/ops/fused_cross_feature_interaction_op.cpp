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
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

/* static */ Maybe<void> FusedCrossFeatureInteractionOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  const Shape& x_shape = ctx->InputShape("x", 0);
  const Shape& weight_shape = ctx->InputShape("weight", 0);
  CHECK_EQ_OR_RETURN(x_shape.At(1), weight_shape.At(1)) << "Matmul K dims should be equal. ";
  *ctx->MutOutputShape("matmul_result", 0) = Shape({x_shape.At(0), weight_shape.At(0)});
  const Shape& x0_shape = ctx->InputShape("x0", 0);
  const Shape& bias_shape = ctx->InputShape("bias", 0);
  CHECK_EQ_OR_RETURN(bias_shape.At(0), x0_shape.At(1)) << "Bias dim should be equal to X0 dim1. ";
  *ctx->MutOutputShape("out", 0) = x0_shape;
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FusedCrossFeatureInteractionOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> FusedCrossFeatureInteractionOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("x", 0), 0)
      .Broadcast(user_op::OpArg("weight", 0))
      .Split(user_op::OpArg("x0", 0), 0)
      .Broadcast(user_op::OpArg("bias", 0))
      .Split(user_op::OpArg("matmul_result", 0), 0)
      .Split(user_op::OpArg("out", 0), 0)
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FusedCrossFeatureInteractionOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->MutOutputDType("out", 0) = ctx->InputDType("x", 0);
  *ctx->MutOutputDType("matmul_result", 0) = ctx->InputDType("x", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FusedCrossFeatureInteractionV1GradOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  const Shape& x0_shape = ctx->InputShape("x0", 0);
  const Shape& weight_shape = ctx->InputShape("weight", 0);
  *ctx->MutOutputShape("dx0", 0) = x0_shape;
  *ctx->MutOutputShape("dw", 0) = weight_shape;
  *ctx->MutOutputShape("dx", 0) = x0_shape;
  *ctx->MutOutputShape("dbias", 0) = Shape({x0_shape.At(1)});
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FusedCrossFeatureInteractionV1GradOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> FusedCrossFeatureInteractionV1GradOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("dy", 0), 0)
      .Broadcast(user_op::OpArg("weight", 0))
      .Split(user_op::OpArg("x", 0), 0)
      .Split(user_op::OpArg("x0", 0), 0)
      .Split(user_op::OpArg("matmul_result", 0), 0)
      .Split(user_op::OpArg("dx0", 0), 0)
      .PartialSum(user_op::OpArg("dw", 0))
      .Split(user_op::OpArg("dx", 0), 0)
      .PartialSum(user_op::OpArg("dbias", 0))
      .Build();

  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FusedCrossFeatureInteractionV1GradOp::InferDataType(
    user_op::InferContext* ctx) {
  *ctx->MutOutputDType("dx0", 0) = ctx->InputDType("x", 0);
  *ctx->MutOutputDType("dw", 0) = ctx->InputDType("x", 0);
  *ctx->MutOutputDType("dx", 0) = ctx->InputDType("x", 0);
  *ctx->MutOutputDType("dbias", 0) = ctx->InputDType("x", 0);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FusedCrossFeatureInteractionV2GradOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  const Shape& x0_shape = ctx->InputShape("x0", 0);
  const Shape& weight_shape = ctx->InputShape("weight", 0);
  *ctx->MutOutputShape("dx0", 0) = x0_shape;
  *ctx->MutOutputShape("dw", 0) = weight_shape;
  *ctx->MutOutputShape("dx", 0) = x0_shape;
  *ctx->MutOutputShape("dbias", 0) = Shape({x0_shape.At(1)});
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FusedCrossFeatureInteractionV2GradOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> FusedCrossFeatureInteractionV2GradOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("dy", 0), 0)
      .Broadcast(user_op::OpArg("weight", 0))
      .Broadcast(user_op::OpArg("bias", 0))
      .Split(user_op::OpArg("x", 0), 0)
      .Split(user_op::OpArg("x0", 0), 0)
      .Split(user_op::OpArg("matmul_result", 0), 0)
      .Split(user_op::OpArg("dx0", 0), 0)
      .PartialSum(user_op::OpArg("dw", 0))
      .Split(user_op::OpArg("dx", 0), 0)
      .PartialSum(user_op::OpArg("dbias", 0))
      .Build();

  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FusedCrossFeatureInteractionV2GradOp::InferDataType(
    user_op::InferContext* ctx) {
  *ctx->MutOutputDType("dx0", 0) = ctx->InputDType("x", 0);
  *ctx->MutOutputDType("dw", 0) = ctx->InputDType("x", 0);
  *ctx->MutOutputDType("dx", 0) = ctx->InputDType("x", 0);
  *ctx->MutOutputDType("dbias", 0) = ctx->InputDType("x", 0);
  return Maybe<void>::Ok();
}

REGISTER_USER_OP_GRAD("fused_cross_feature_interaction")
    .SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                               const user_op::AddOpFn& AddOp) -> Maybe<void> {
      user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
      if (op.attr<std::string>("interaction_mode") == "vector") {
        builder.Op("fused_cross_feature_interaction_v1_grad")
            .Input("dy", op.GetGradTensorWithOpOutput("out", 0))
            .Input("weight", op.input("weight", 0))
            .Input("x", op.input("x", 0))
            .Input("x0", op.input("x0", 0))
            .Input("matmul_result", op.output("matmul_result", 0));
      } else if (op.attr<std::string>("interaction_mode") == "matrix") {
        builder.Op("fused_cross_feature_interaction_v2_grad")
            .Input("dy", op.GetGradTensorWithOpOutput("out", 0))
            .Input("weight", op.input("weight", 0))
            .Input("bias", op.input("bias", 0))
            .Input("x", op.input("x", 0))
            .Input("x0", op.input("x0", 0))
            .Input("matmul_result", op.output("matmul_result", 0));
      } else {
        UNIMPLEMENTED();
      }
      builder.Output("dx").Output("dw").Output("dx0").Output("dbias");
      auto grad_op = builder.Build();
      AddOp(grad_op);
      if (op.NeedGenGradTensor4OpInput("x", 0)) {
        op.BindGradTensorWithOpInput(grad_op.output("dx", 0), "x", 0);
      }
      if (op.NeedGenGradTensor4OpInput("weight", 0)) {
        op.BindGradTensorWithOpInput(grad_op.output("dw", 0), "weight", 0);
      }
      if (op.NeedGenGradTensor4OpInput("x0", 0)) {
        op.BindGradTensorWithOpInput(grad_op.output("dx0", 0), "x0", 0);
      }
      if (op.NeedGenGradTensor4OpInput("bias", 0)) {
        op.BindGradTensorWithOpInput(grad_op.output("dbias", 0), "bias", 0);
      }
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
