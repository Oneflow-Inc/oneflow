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

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/op_generated.h"
#include "oneflow/core/framework/user_op_conf.h"

namespace oneflow {

/*static*/ auto FusedMSASoftmaxOp::InferDataType(user_op::InferContext* ctx) -> Maybe<void> {
  DataType query_type = ctx->InputDType("qmk", 0);
  DataType mask_bias_type = ctx->InputDType("mask", 0);
  CHECK_EQ_OR_RETURN(mask_bias_type, query_type);

  const std::string mode = ctx->Attr<std::string>("mode");
  if (ctx->has_input("bias", 0)) {
    CHECK_OR_RETURN(mode == "row" || mode == "triangular_start" || mode == "triangular_end");
    DataType bias_type = ctx->InputDType("bias", 0);
    CHECK_EQ_OR_RETURN(bias_type, query_type);
  } else {
    CHECK_OR_RETURN(mode == "global_col" || mode == "col" || mode == "template");
  }
  ctx->SetOutputDType("out", 0, query_type);
  return Maybe<void>::Ok();
}

/*static*/ auto FusedMSASoftmaxOp::InferLogicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  const float scale = ctx->Attr<float>("scale");
  CHECK_LE_OR_RETURN(scale, 1.);

  const Shape& qmk_shape = ctx->InputShape("qmk", 0);
  const std::string mode = ctx->Attr<std::string>("mode");
  if (mode == "global_col") {
    int64_t naxes = qmk_shape.NumAxes();
    CHECK_OR_RETURN(naxes == 3 || (naxes == 4 && qmk_shape.At(0) == 1));
    int64_t start = naxes == 3 ? 0 : 1;
    int64_t S1 = qmk_shape.At(0 + start), N = qmk_shape.At(2 + start);
    const Shape& mask_shape = ctx->InputShape("mask", 0);
    CHECK_EQ_OR_RETURN(mask_shape.At(start + 0), S1);
    CHECK_EQ_OR_RETURN(mask_shape.At(start + 1), 1);
    CHECK_EQ_OR_RETURN(mask_shape.At(start + 2), N);
  } else if (mode == "col") {
    int64_t naxes = qmk_shape.NumAxes();
    CHECK_OR_RETURN(naxes == 4 || (naxes == 5 && qmk_shape.at(0) == 1));
    int64_t start = naxes == 4 ? 0 : 1;
    int64_t N = qmk_shape.At(start + 0), S = qmk_shape.At(start + 3);
    const Shape& mask_shape = ctx->InputShape("mask", 0);
    CHECK_EQ_OR_RETURN(mask_shape.At(start + 0), N);
    CHECK_EQ_OR_RETURN(mask_shape.At(start + 1), 1);
    CHECK_EQ_OR_RETURN(mask_shape.At(start + 2), 1);
    CHECK_EQ_OR_RETURN(mask_shape.At(start + 3), S);
  } else if (mode == "row" || mode == "triangular_start" || mode == "triangular_end") {
    int64_t naxes = qmk_shape.NumAxes();
    CHECK_OR_RETURN(naxes == 4 || (naxes == 5 && qmk_shape.at(0) == 1));
    int64_t start = naxes == 4 ? 0 : 1;
    const int64_t batch_size = qmk_shape.At(start + 0);
    const int64_t num_heads = qmk_shape.At(start + 1);
    const int64_t query_lens = qmk_shape.At(start + 2);
    const int64_t key_lens = qmk_shape.At(start + 3);
    CHECK_GT_OR_RETURN(query_lens, 0);
    CHECK_EQ_OR_RETURN(query_lens, key_lens);

    const Shape& mask_bias_shape = ctx->InputShape("mask", 0);
    CHECK_EQ_OR_RETURN(mask_bias_shape.At(start + 0), batch_size);
    CHECK_EQ_OR_RETURN(mask_bias_shape.At(start + 1), 1);
    CHECK_EQ_OR_RETURN(mask_bias_shape.At(start + 2), 1);
    CHECK_EQ_OR_RETURN(mask_bias_shape.At(start + 3), query_lens);

    CHECK_OR_RETURN(ctx->has_input("bias", 0));
    const Shape& pair_bias_shape = ctx->InputShape("bias", 0);
    CHECK_EQ_OR_RETURN(pair_bias_shape.At(start + 0), 1);
    CHECK_EQ_OR_RETURN(pair_bias_shape.At(start + 1), num_heads);
    CHECK_EQ_OR_RETURN(pair_bias_shape.At(start + 2), query_lens);
    CHECK_EQ_OR_RETURN(pair_bias_shape.At(start + 3), key_lens);
  } else if (mode == "template") {
    int64_t naxes = qmk_shape.NumAxes();
    CHECK_OR_RETURN(naxes == 5 || (naxes == 6 && qmk_shape.At(0) == 1));  // *, S, S, h, 1, n_templ
    int64_t start = naxes == 5 ? 0 : 1;
    CHECK_OR_RETURN(qmk_shape.at(start + 0) == qmk_shape.at(start + 1));
    int64_t Nt = qmk_shape.At(start + 4);
    const Shape& mask_shape = ctx->InputShape("mask", 0);
    CHECK_EQ_OR_RETURN(mask_shape.elem_cnt(), Nt);
    CHECK_EQ_OR_RETURN(mask_shape.At(start + 4), Nt);
  } else {
    LOG(ERROR) << "mode \"" << mode << "\" unimplemented.";
  }
  ctx->SetOutputShape("out", 0, qmk_shape);
  return Maybe<void>::Ok();
}

/*static*/ auto FusedMSASoftmaxOp::InferPhysicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  return InferLogicalTensorDesc(ctx);
}

/*static*/ auto FusedMSASoftmaxOp::GetSbp(user_op::SbpContext* ctx) -> Maybe<void> {
  if (ctx->Attr<bool>("inplace") == false)
    ctx->NewBuilder()
        .Split(user_op::OpArg("qmk", 0), 0)
        .Split(user_op::OpArg("mask", 0), 0)
        .Broadcast(user_op::OpArg("bias", 0))
        .Split(user_op::OpArg("out", 0), 0)
        .Build();
  else
    ctx->NewBuilder()
        .Split(user_op::OpArg("qmk", 0), 0)
        .Split(user_op::OpArg("mask", 0), 0)
        .Broadcast(user_op::OpArg("bias", 0))
        .Build();
  return Maybe<void>::Ok();
}

/*static*/ auto FusedMSASoftmaxGradOp::InferDataType(user_op::InferContext* ctx) -> Maybe<void> {
  DataType y_type = ctx->InputDType("y", 0);
  DataType dy_type = ctx->InputDType("dy", 0);
  CHECK_EQ_OR_RETURN(y_type, dy_type);
  const std::string mode = ctx->Attr<std::string>("mode");
  ctx->SetOutputDType("dx", 0, y_type);
  return Maybe<void>::Ok();
}

/*static*/ auto FusedMSASoftmaxGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  const Shape& y_shape = ctx->InputShape("y", 0);
  const Shape& dy_shape = ctx->InputShape("dy", 0);
  CHECK_EQ_OR_RETURN(y_shape, dy_shape);
  ctx->SetOutputShape("dx", 0, y_shape);
  return Maybe<void>::Ok();
}

/*static*/ auto FusedMSASoftmaxGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  return InferLogicalTensorDesc(ctx);
}

/*static*/ auto FusedMSASoftmaxGradOp::GetSbp(user_op::SbpContext* ctx) -> Maybe<void> {
  ctx->NewBuilder()
      .Split(user_op::OpArg("y", 0), 0)
      .Split(user_op::OpArg("dy", 0), 0)
      .Split(user_op::OpArg("dx", 0), 0)
      .Build();
  return Maybe<void>::Ok();
}

/*static*/ auto FusedMSABiasaddSigmoidMulOp::InferDataType(user_op::InferContext* ctx)
    -> Maybe<void> {
  DataType g_type = ctx->InputDType("g", 0);
  DataType x_type = ctx->InputDType("x", 0);
  DataType b_type = ctx->InputDType("b", 0);
  CHECK_EQ_OR_RETURN(g_type, x_type);
  CHECK_EQ_OR_RETURN(b_type, x_type);
  ctx->SetOutputDType("out", 0, g_type);
  return Maybe<void>::Ok();
}

/*static*/ auto FusedMSABiasaddSigmoidMulOp::InferLogicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  const Shape& b_shape = ctx->InputShape("b", 0);
  const Shape& g_shape = ctx->InputShape("g", 0);
  const Shape& x_shape = ctx->InputShape("x", 0);
  CHECK_EQ_OR_RETURN(g_shape, x_shape);
  const int32_t n = x_shape.NumAxes();
  CHECK_EQ_OR_RETURN(b_shape.At(0), x_shape.At(n - 1));
  ctx->SetOutputShape("out", 0, g_shape);
  return Maybe<void>::Ok();
}

/*static*/ auto FusedMSABiasaddSigmoidMulOp::InferPhysicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  return InferLogicalTensorDesc(ctx);
}

/*static*/ auto FusedMSABiasaddSigmoidMulOp::GetSbp(user_op::SbpContext* ctx) -> Maybe<void> {
  const bool inplace = ctx->Attr<bool>("inplace");
  if (inplace) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("x", 0), 0)
        .Split(user_op::OpArg("g", 0), 0)
        .Split(user_op::OpArg("b", 0), 0)
        .Build();
  } else {
    ctx->NewBuilder()
        .Split(user_op::OpArg("x", 0), 0)
        .Split(user_op::OpArg("g", 0), 0)
        .Split(user_op::OpArg("b", 0), 0)
        .Split(user_op::OpArg("out", 0), 0)
        .Build();
  }
  return Maybe<void>::Ok();
}

/*static*/ auto FusedMSABiasaddSigmoidMulGradOp::InferDataType(user_op::InferContext* ctx)
    -> Maybe<void> {
  DataType dout_type = ctx->InputDType("dout", 0);
  DataType x_type = ctx->InputDType("x", 0);
  DataType g_type = ctx->InputDType("g", 0);
  DataType b_type = ctx->InputDType("b", 0);
  CHECK_EQ_OR_RETURN(g_type, dout_type);
  CHECK_EQ_OR_RETURN(x_type, dout_type);
  CHECK_EQ_OR_RETURN(b_type, dout_type);
  const bool inplace = ctx->Attr<bool>("inplace");
  CHECK_EQ_OR_RETURN(inplace, false);
  ctx->SetOutputDType("dg", 0, g_type);
  return Maybe<void>::Ok();
}

/*static*/ auto FusedMSABiasaddSigmoidMulGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  const Shape& dout_shape = ctx->InputShape("dout", 0);
  const Shape& g_shape = ctx->InputShape("g", 0);
  const Shape& x_shape = ctx->InputShape("x", 0);
  const Shape& b_shape = ctx->InputShape("b", 0);
  CHECK_EQ_OR_RETURN(g_shape, dout_shape);
  CHECK_EQ_OR_RETURN(x_shape, dout_shape);
  const int32_t n = x_shape.NumAxes();
  CHECK_EQ_OR_RETURN(b_shape.At(0), x_shape.At(n - 1));
  ctx->SetOutputShape("dg", 0, g_shape);
  return Maybe<void>::Ok();
}

/*static*/ auto FusedMSABiasaddSigmoidMulGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  return InferLogicalTensorDesc(ctx);
}

/*static*/ auto FusedMSABiasaddSigmoidMulGradOp::GetSbp(user_op::SbpContext* ctx) -> Maybe<void> {
  ctx->NewBuilder()
      .Split(user_op::OpArg("x", 0), 0)
      .Split(user_op::OpArg("g", 0), 0)
      .Split(user_op::OpArg("b", 0), 0)
      .Split(user_op::OpArg("out", 0), 0)
      .Split(user_op::OpArg("dg", 0), 0)
      .Build();
  return Maybe<void>::Ok();
}

/*static*/ auto FusedMSABiasaddDropoutResidualOp::InferDataType(user_op::InferContext* ctx)
    -> Maybe<void> {
  DataType x_type = ctx->InputDType("x", 0);
  DataType bias_type = ctx->InputDType("bias", 0);
  DataType mask_type = ctx->InputDType("mask", 0);
  CHECK_EQ_OR_RETURN(mask_type, x_type);
  CHECK_EQ_OR_RETURN(bias_type, x_type);
  DataType res_type = ctx->InputDType("residual", 0);
  CHECK_EQ_OR_RETURN(res_type, x_type);
  ctx->SetOutputDType("out", 0, x_type);
  return Maybe<void>::Ok();
}

/*static*/ auto FusedMSABiasaddDropoutResidualOp::InferLogicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  const Shape& x_shape = ctx->InputShape("x", 0);
  const Shape& bias_shape = ctx->InputShape("bias", 0);
  const Shape& mask_shape = ctx->InputShape("mask", 0);
  const int32_t n = x_shape.NumAxes();
  // CHECK_EQ_OR_RETURN(mask_shape, x_shape);
  CHECK_EQ_OR_RETURN(bias_shape.At(0), x_shape.At(n - 1));
  const Shape& res_shape = ctx->InputShape("residual", 0);
  CHECK_EQ_OR_RETURN(res_shape, x_shape);
  ctx->SetOutputShape("out", 0, x_shape);
  return Maybe<void>::Ok();
}

/*static*/ auto FusedMSABiasaddDropoutResidualOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) -> Maybe<void> {
  return InferLogicalTensorDesc(ctx);
}

/*static*/ auto FusedMSABiasaddDropoutResidualOp::GetSbp(user_op::SbpContext* ctx) -> Maybe<void> {
  const bool inplace = ctx->Attr<bool>("inplace");
  if (inplace) {
    ctx->NewBuilder()
        .Broadcast(user_op::OpArg("bias", 0))
        .Split(user_op::OpArg("x", 0), 0)
        .Split(user_op::OpArg("mask", 0), 0)
        .Split(user_op::OpArg("residual", 0), 0)
        .Build();
  } else {
    ctx->NewBuilder()
        .Broadcast(user_op::OpArg("bias", 0))
        .Split(user_op::OpArg("x", 0), 0)
        .Split(user_op::OpArg("mask", 0), 0)
        .Split(user_op::OpArg("residual", 0), 0)
        .Split(user_op::OpArg("out", 0), 0)
        .Build();
  }
  return Maybe<void>::Ok();
}

/*static*/ auto FusedMSABiasaddDropoutResidualGradOp::InferDataType(user_op::InferContext* ctx)
    -> Maybe<void> {
  DataType dout_type = ctx->InputDType("dout", 0);
  DataType mask_type = ctx->InputDType("mask", 0);
  CHECK_EQ_OR_RETURN(mask_type, dout_type);
  const bool inplace = ctx->Attr<bool>("inplace");
  CHECK_EQ_OR_RETURN(inplace, false);
  ctx->SetOutputDType("dx", 0, dout_type);
  return Maybe<void>::Ok();
}

/*static*/ auto FusedMSABiasaddDropoutResidualGradOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) -> Maybe<void> {
  const Shape& dout_shape = ctx->InputShape("dout", 0);
  const Shape& mask_shape = ctx->InputShape("mask", 0);
  CHECK_EQ_OR_RETURN(mask_shape, dout_shape);
  auto db_shape = Shape();
  auto axes = dout_shape.NumAxes();
  db_shape.push_back(dout_shape.At(axes - 1));
  ctx->SetOutputShape("dx", 0, dout_shape);
  return Maybe<void>::Ok();
}

/*static*/ auto FusedMSABiasaddDropoutResidualGradOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) -> Maybe<void> {
  return InferLogicalTensorDesc(ctx);
}

/*static*/ auto FusedMSABiasaddDropoutResidualGradOp::GetSbp(user_op::SbpContext* ctx)
    -> Maybe<void> {
  ctx->NewBuilder()
      .Split(user_op::OpArg("dout", 0), 0)
      .Split(user_op::OpArg("mask", 0), 0)
      .Split(user_op::OpArg("dx", 0), 0)
      .Build();

  return Maybe<void>::Ok();
}

/*static*/ auto FusedMSATmuOp::InferDataType(user_op::InferContext* ctx) -> Maybe<void> {
  DataType x1_type = ctx->InputDType("x1", 0);
  DataType b1_type = ctx->InputDType("b1", 0);
  DataType x2_type = ctx->InputDType("x2", 0);
  DataType b2_type = ctx->InputDType("b2", 0);
  DataType r_type = ctx->InputDType("residual", 0);
  DataType mask_type = ctx->InputDType("mask", 0);
  CHECK_EQ_OR_RETURN(b1_type, x1_type);
  CHECK_EQ_OR_RETURN(x2_type, x1_type);
  CHECK_EQ_OR_RETURN(b2_type, x1_type);
  CHECK_EQ_OR_RETURN(r_type, x1_type);
  CHECK_EQ_OR_RETURN(mask_type, x1_type);
  const bool inplace = ctx->Attr<bool>("inplace");
  CHECK_EQ_OR_RETURN(inplace, false);
  ctx->SetOutputDType("dx", 0, x1_type);
  return Maybe<void>::Ok();
}

/*static*/ auto FusedMSATmuOp::InferLogicalTensorDesc(user_op::InferContext* ctx) -> Maybe<void> {
  const Shape& x1_shape = ctx->InputShape("x1", 0);
  const Shape& b1_shape = ctx->InputShape("b1", 0);
  const Shape& x2_shape = ctx->InputShape("x2", 0);
  const Shape& b2_shape = ctx->InputShape("b2", 0);
  const Shape& r_shape = ctx->InputShape("residual", 0);
  const Shape& mask_shape = ctx->InputShape("mask", 0);
  CHECK_EQ_OR_RETURN(x2_shape, x1_shape);
  CHECK_EQ_OR_RETURN(r_shape, x1_shape);
  auto axes = x1_shape.NumAxes();
  CHECK_EQ_OR_RETURN(b1_shape.At(0), x1_shape.At(axes - 1));
  CHECK_EQ_OR_RETURN(b2_shape.At(0), x1_shape.At(axes - 1));
  ctx->SetOutputShape("out", 0, x1_shape);
  return Maybe<void>::Ok();
}

/*static*/ auto FusedMSATmuOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) -> Maybe<void> {
  return InferLogicalTensorDesc(ctx);
}

/*static*/ auto FusedMSATmuOp::GetSbp(user_op::SbpContext* ctx) -> Maybe<void> {
  ctx->NewBuilder()
      .Split(user_op::OpArg("x1", 0), 0)
      .Broadcast(user_op::OpArg("b1", 0))
      .Split(user_op::OpArg("x2", 0), 0)
      .Broadcast(user_op::OpArg("b2", 0))
      .Broadcast(user_op::OpArg("mask", 0))
      .Split(user_op::OpArg("residual", 0), 0)
      .Split(user_op::OpArg("out", 0), 0)
      .Build();

  return Maybe<void>::Ok();
}

/*static*/ auto FusedMSATmuGradOp::InferDataType(user_op::InferContext* ctx) -> Maybe<void> {
  DataType dout_type = ctx->InputDType("dout", 0);
  DataType x1_type = ctx->InputDType("x1", 0);
  DataType b1_type = ctx->InputDType("b1", 0);
  DataType x2_type = ctx->InputDType("x2", 0);
  DataType b2_type = ctx->InputDType("b2", 0);
  DataType mask_type = ctx->InputDType("mask", 0);
  CHECK_EQ_OR_RETURN(x1_type, dout_type);
  CHECK_EQ_OR_RETURN(b1_type, dout_type);
  CHECK_EQ_OR_RETURN(x2_type, dout_type);
  CHECK_EQ_OR_RETURN(b2_type, dout_type);
  CHECK_EQ_OR_RETURN(mask_type, dout_type);
  const bool inplace = ctx->Attr<bool>("inplace");
  CHECK_EQ_OR_RETURN(inplace, false);
  ctx->SetOutputDType("dx1", 0, dout_type);
  ctx->SetOutputDType("dx2", 0, dout_type);
  return Maybe<void>::Ok();
}

/*static*/ auto FusedMSATmuGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  const Shape& dout_shape = ctx->InputShape("dout", 0);
  ctx->SetOutputShape("dx1", 0, dout_shape);
  ctx->SetOutputShape("dx2", 0, dout_shape);
  return Maybe<void>::Ok();
}

/*static*/ auto FusedMSATmuGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  return InferLogicalTensorDesc(ctx);
}

/*static*/ auto FusedMSATmuGradOp::GetSbp(user_op::SbpContext* ctx) -> Maybe<void> {
  ctx->NewBuilder()
      .Split(user_op::OpArg("dout", 0), 0)
      .Split(user_op::OpArg("mask", 0), 0)
      .Split(user_op::OpArg("dx1", 0), 0)
      .Split(user_op::OpArg("dx2", 0), 0)
      .Build();

  return Maybe<void>::Ok();
}
}  // namespace oneflow
