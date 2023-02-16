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

/*static*/ auto FusedScaleMaskBiasSoftmaxOp::InferDataType(user_op::InferContext* ctx)
    -> Maybe<void> {
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

/*static*/ auto FusedScaleMaskBiasSoftmaxOp::InferLogicalTensorDesc(user_op::InferContext* ctx)
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

/*static*/ auto FusedScaleMaskBiasSoftmaxOp::InferPhysicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  return InferLogicalTensorDesc(ctx);
}

/*static*/ auto FusedScaleMaskBiasSoftmaxOp::GetSbp(user_op::SbpContext* ctx) -> Maybe<void> {
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

/*static*/ auto FusedScaleMaskBiasSoftmaxGradOp::InferDataType(user_op::InferContext* ctx)
    -> Maybe<void> {
  DataType y_type = ctx->InputDType("y", 0);
  DataType dy_type = ctx->InputDType("dy", 0);
  CHECK_EQ_OR_RETURN(y_type, dy_type);
  const std::string mode = ctx->Attr<std::string>("mode");
  ctx->SetOutputDType("dx", 0, y_type);
  return Maybe<void>::Ok();
}

/*static*/ auto FusedScaleMaskBiasSoftmaxGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  const Shape& y_shape = ctx->InputShape("y", 0);
  const Shape& dy_shape = ctx->InputShape("dy", 0);
  CHECK_EQ_OR_RETURN(y_shape, dy_shape);
  ctx->SetOutputShape("dx", 0, y_shape);
  return Maybe<void>::Ok();
}

/*static*/ auto FusedScaleMaskBiasSoftmaxGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  return InferLogicalTensorDesc(ctx);
}

/*static*/ auto FusedScaleMaskBiasSoftmaxGradOp::GetSbp(user_op::SbpContext* ctx) -> Maybe<void> {
  ctx->NewBuilder()
      .Split(user_op::OpArg("y", 0), 0)
      .Split(user_op::OpArg("dy", 0), 0)
      .Split(user_op::OpArg("dx", 0), 0)
      .Build();
  return Maybe<void>::Ok();
}

}  // namespace oneflow
