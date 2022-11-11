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
#include <glog/logging.h>
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

/*static*/ auto FusedMSAAttentionOp::InferDataType(user_op::InferContext* ctx) -> Maybe<void> {
  DataType query_type = ctx->InputDType("qmk", 0);
  DataType mask_bias_type = ctx->InputDType("mask", 0);
  CHECK_EQ_OR_RETURN(mask_bias_type, query_type);

  const std::string mode = ctx->Attr<std::string>("mode");
  if (ctx->has_input("bias", 0)) {
    CHECK_OR_RETURN(mode == "row" || mode == "triangle_start" || mode == "triangle_end");
    DataType bias_type = ctx->InputDType("bias", 0);
    CHECK_EQ_OR_RETURN(bias_type, query_type);
  } else {
    CHECK_OR_RETURN(mode == "global_col" || mode == "col" || mode == "template");
  }

  ctx->SetOutputDType("out", 0, query_type);
  return Maybe<void>::Ok();
}

/*static*/ auto FusedMSAAttentionOp::InferLogicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  const float scale = ctx->Attr<float>("scale");
  CHECK_LE_OR_RETURN(scale, 1.);

  const Shape& qmk_shape = ctx->InputShape("qmk", 0);
  const std::string mode = ctx->Attr<std::string>("mode");
  if (mode == "global_col") {
    CHECK_EQ_OR_RETURN(qmk_shape.NumAxes(), 3);
    int64_t S1 = qmk_shape.At(0), N = qmk_shape.At(2);
    const Shape& mask_shape = ctx->InputShape("mask", 0);
    CHECK_EQ_OR_RETURN(mask_shape.At(0), S1);
    CHECK_EQ_OR_RETURN(mask_shape.At(1), 1);
    CHECK_EQ_OR_RETURN(mask_shape.At(2), N);
  } else if (mode == "col") {
    CHECK_EQ_OR_RETURN(qmk_shape.NumAxes(), 4);
    int64_t N = qmk_shape.At(0), S = qmk_shape.At(3);
    const Shape& mask_shape = ctx->InputShape("mask", 0);
    CHECK_EQ_OR_RETURN(mask_shape.At(0), N);
    CHECK_EQ_OR_RETURN(mask_shape.At(1), 1);
    CHECK_EQ_OR_RETURN(mask_shape.At(2), 1);
    CHECK_EQ_OR_RETURN(mask_shape.At(3), S);
  } else if (mode == "row" || mode == "triangle_start" || mode == "triangle_end") {
    CHECK_EQ_OR_RETURN(qmk_shape.NumAxes(), 4);
    const int64_t batch_size = qmk_shape.At(0);
    const int64_t num_heads = qmk_shape.At(1);
    const int64_t query_lens = qmk_shape.At(2);
    const int64_t key_lens = qmk_shape.At(3);
    CHECK_GT_OR_RETURN(query_lens, 0);
    CHECK_EQ_OR_RETURN(query_lens, key_lens);

    const Shape& mask_bias_shape = ctx->InputShape("mask", 0);
    CHECK_EQ_OR_RETURN(mask_bias_shape.At(0), batch_size);
    CHECK_EQ_OR_RETURN(mask_bias_shape.At(1), 1);
    CHECK_EQ_OR_RETURN(mask_bias_shape.At(2), 1);
    CHECK_EQ_OR_RETURN(mask_bias_shape.At(3), query_lens);

    CHECK_OR_RETURN(ctx->has_input("bias", 0));
    const Shape& pair_bias_shape = ctx->InputShape("bias", 0);
    CHECK_EQ_OR_RETURN(pair_bias_shape.At(0), 1);
    CHECK_EQ_OR_RETURN(pair_bias_shape.At(1), num_heads);
    CHECK_EQ_OR_RETURN(pair_bias_shape.At(2), query_lens);
    CHECK_EQ_OR_RETURN(pair_bias_shape.At(3), key_lens);
  } else if (mode == "template") {
    CHECK_EQ_OR_RETURN(qmk_shape.NumAxes(), 4);
    int64_t Nt = qmk_shape.At(3);
    const Shape& mask_shape = ctx->InputShape("mask", 0);
    CHECK_EQ_OR_RETURN(mask_shape.At(0), 1);
    CHECK_EQ_OR_RETURN(mask_shape.At(1), 1);
    CHECK_EQ_OR_RETURN(mask_shape.At(2), 1);
    CHECK_EQ_OR_RETURN(mask_shape.At(3), Nt);
  } else {
    LOG(ERROR) << "mode \"" << mode << "\" unimplemented.";
  }
  ctx->SetOutputShape("out", 0, qmk_shape);
  return Maybe<void>::Ok();
}

/*static*/ auto FusedMSAAttentionOp::InferPhysicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  return InferLogicalTensorDesc(ctx);
}

/*static*/ auto FusedMSAAttentionOp::GetSbp(user_op::SbpContext* ctx) -> Maybe<void> {
  ctx->NewBuilder()
      .Split(user_op::OpArg("qmk", 0), 0)
      .Split(user_op::OpArg("mask", 0), 0)
      .Broadcast(user_op::OpArg("bias", 0))
      .Split(user_op::OpArg("out", 0), 0)
      .Build();
  return Maybe<void>::Ok();
}

/*static*/ auto FusedMSAAttentionGradOp::InferDataType(user_op::InferContext* ctx) -> Maybe<void> {
  DataType y_type = ctx->InputDType("y", 0);
  DataType dy_type = ctx->InputDType("dy", 0);
  CHECK_EQ_OR_RETURN(y_type, dy_type);
  const std::string mode = ctx->Attr<std::string>("mode");
  ctx->SetOutputDType("dx", 0, y_type);
  return Maybe<void>::Ok();
}

/*static*/ auto FusedMSAAttentionGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  const Shape& y_shape = ctx->InputShape("y", 0);
  const Shape& dy_shape = ctx->InputShape("dy", 0);
  CHECK_EQ_OR_RETURN(y_shape, dy_shape);
  ctx->SetOutputShape("dx", 0, y_shape);
  return Maybe<void>::Ok();
}

/*static*/ auto FusedMSAAttentionGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  return InferLogicalTensorDesc(ctx);
}

/*static*/ auto FusedMSAAttentionGradOp::GetSbp(user_op::SbpContext* ctx) -> Maybe<void> {
  ctx->NewBuilder()
      .Split(user_op::OpArg("y", 0), 0)
      .Split(user_op::OpArg("dy", 0), 0)
      .Split(user_op::OpArg("dx", 0), 0)
      .Build();
  return Maybe<void>::Ok();
}

/*static*/ auto FusedMSASigmoidMulOp::InferDataType(user_op::InferContext* ctx) -> Maybe<void> {
  DataType g_type = ctx->InputDType("g", 0);
  DataType x_type = ctx->InputDType("x", 0);
  CHECK_EQ_OR_RETURN(g_type, x_type);
  const bool inplace = ctx->Attr<bool>("inplace");
  if (inplace == false) { ctx->SetOutputDType("out", 0, g_type); }
  return Maybe<void>::Ok();
}

/*static*/ auto FusedMSASigmoidMulOp::InferLogicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  const Shape& g_shape = ctx->InputShape("g", 0);
  const Shape& x_shape = ctx->InputShape("x", 0);
  CHECK_EQ_OR_RETURN(g_shape, x_shape);
  const bool inplace = ctx->Attr<bool>("inplace");
  if (inplace == false) { ctx->SetOutputShape("out", 0, g_shape); }
  return Maybe<void>::Ok();
}

/*static*/ auto FusedMSASigmoidMulOp::InferPhysicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  return InferLogicalTensorDesc(ctx);
}

/*static*/ auto FusedMSASigmoidMulOp::GetSbp(user_op::SbpContext* ctx) -> Maybe<void> {
  const bool inplace = ctx->Attr<bool>("inplace");
  if (inplace) {
    ctx->NewBuilder().Split(user_op::OpArg("x", 0), 0).Split(user_op::OpArg("g", 0), 0).Build();
  }
  ctx->NewBuilder()
      .Split(user_op::OpArg("x", 0), 0)
      .Split(user_op::OpArg("g", 0), 0)
      .Split(user_op::OpArg("out", 0), 0)
      .Build();
  return Maybe<void>::Ok();
}

/*static*/ auto FusedMSASigmoidMulGradOp::InferDataType(user_op::InferContext* ctx) -> Maybe<void> {
  DataType dout_type = ctx->InputDType("dout", 0);
  DataType x_type = ctx->InputDType("x", 0);
  DataType g_type = ctx->InputDType("g", 0);
  CHECK_EQ_OR_RETURN(g_type, dout_type);
  CHECK_EQ_OR_RETURN(x_type, dout_type);
  const bool inplace = ctx->Attr<bool>("inplace");
  CHECK_EQ_OR_RETURN(inplace, false);
  ctx->SetOutputDType("dg", 0, g_type);
  ctx->SetOutputDType("dx", 0, x_type);
  return Maybe<void>::Ok();
}

/*static*/ auto FusedMSASigmoidMulGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  const Shape& dout_shape = ctx->InputShape("dout", 0);
  const Shape& g_shape = ctx->InputShape("g", 0);
  const Shape& x_shape = ctx->InputShape("x", 0);
  CHECK_EQ_OR_RETURN(g_shape, dout_shape);
  CHECK_EQ_OR_RETURN(x_shape, dout_shape);
  ctx->SetOutputShape("dg", 0, g_shape);
  ctx->SetOutputShape("dx", 0, x_shape);
  return Maybe<void>::Ok();
}

/*static*/ auto FusedMSASigmoidMulGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  return InferLogicalTensorDesc(ctx);
}

/*static*/ auto FusedMSASigmoidMulGradOp::GetSbp(user_op::SbpContext* ctx) -> Maybe<void> {
  ctx->NewBuilder()
      .Split(user_op::OpArg("x", 0), 0)
      .Split(user_op::OpArg("g", 0), 0)
      .Split(user_op::OpArg("out", 0), 0)
      .Split(user_op::OpArg("dg", 0), 0)
      .Split(user_op::OpArg("dx", 0), 0)
      .Build();
  return Maybe<void>::Ok();
}

/*static*/ auto FusedMSADropoutAddOp::InferDataType(user_op::InferContext* ctx) -> Maybe<void> {
  DataType x_type = ctx->InputDType("x", 0);
  DataType mask_type = ctx->InputDType("mask", 0);
  CHECK_EQ_OR_RETURN(mask_type, x_type);
  DataType res_type = ctx->InputDType("residual", 0);
  CHECK_EQ_OR_RETURN(res_type, x_type);
  const bool inplace = ctx->Attr<bool>("inplace");
  if (inplace == false) { ctx->SetOutputDType("out", 0, x_type); }
  return Maybe<void>::Ok();
}

/*static*/ auto FusedMSADropoutAddOp::InferLogicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  const Shape& x_shape = ctx->InputShape("x", 0);
  const Shape& mask_shape = ctx->InputShape("mask", 0);
  CHECK_EQ_OR_RETURN(mask_shape, x_shape);
  const Shape& res_shape = ctx->InputShape("residual", 0);
  CHECK_EQ_OR_RETURN(res_shape, x_shape);
  const bool inplace = ctx->Attr<bool>("inplace");
  if (inplace == false) { ctx->SetOutputShape("out", 0, x_shape); }
  return Maybe<void>::Ok();
}

/*static*/ auto FusedMSADropoutAddOp::InferPhysicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  return InferLogicalTensorDesc(ctx);
}

/*static*/ auto FusedMSADropoutAddOp::GetSbp(user_op::SbpContext* ctx) -> Maybe<void> {
  const bool inplace = ctx->Attr<bool>("inplace");
  if (inplace) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("x", 0), 0)
        .Split(user_op::OpArg("mask", 0), 0)
        .Split(user_op::OpArg("residual", 0), 0)
        .Build();
  } else {
    ctx->NewBuilder()
        .Split(user_op::OpArg("x", 0), 0)
        .Split(user_op::OpArg("mask", 0), 0)
        .Split(user_op::OpArg("residual", 0), 0)
        .Split(user_op::OpArg("out", 0), 0)
        .Build();
  }
  return Maybe<void>::Ok();
}

/*static*/ auto FusedMSADropoutAddGradOp::InferDataType(user_op::InferContext* ctx) -> Maybe<void> {
  DataType dout_type = ctx->InputDType("dout", 0);
  DataType mask_type = ctx->InputDType("mask", 0);
  CHECK_EQ_OR_RETURN(mask_type, dout_type);
  const bool inplace = ctx->Attr<bool>("inplace");
  CHECK_EQ_OR_RETURN(inplace, false);
  ctx->SetOutputDType("dx", 0, dout_type);
  return Maybe<void>::Ok();
}

/*static*/ auto FusedMSADropoutAddGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  const Shape& dout_shape = ctx->InputShape("dout", 0);
  const Shape& mask_shape = ctx->InputShape("mask", 0);
  CHECK_EQ_OR_RETURN(mask_shape, dout_shape);
  ctx->SetOutputShape("dx", 0, dout_shape);
  return Maybe<void>::Ok();
}

/*static*/ auto FusedMSADropoutAddGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  return InferLogicalTensorDesc(ctx);
}

/*static*/ auto FusedMSADropoutAddGradOp::GetSbp(user_op::SbpContext* ctx) -> Maybe<void> {
  ctx->NewBuilder()
      .Split(user_op::OpArg("dout", 0), 0)
      .Split(user_op::OpArg("mask", 0), 0)
      .Split(user_op::OpArg("dx", 0), 0)
      .Build();

  return Maybe<void>::Ok();
}
}  // namespace oneflow
