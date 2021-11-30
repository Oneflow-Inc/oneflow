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
#include "oneflow/user/ops/loss_op_util.h"
#include "oneflow/core/common/just.h"

namespace oneflow {

Maybe<void> CheckLossReductionAndInferOutputTenserDesc(
    user_op::InferContext* ctx, const std::string& output_name, bool output_is_dynamic,
    const Shape& output_shape_when_reduction_is_none) {
  const std::string reduction = ctx->Attr<std::string>("reduction");
  CHECK_OR_RETURN(LossReductionTypeIsRight(reduction));
  user_op::TensorDesc* out_desc = ctx->OutputTensorDesc(output_name, 0);
  *out_desc->mut_is_dynamic() = output_is_dynamic;
  if (reduction == "none") {
    *out_desc->mut_shape() = output_shape_when_reduction_is_none;
  } else {
    *out_desc->mut_shape() = Shape();
  }
  return Maybe<void>::Ok();
}

Maybe<void> CheckLossReductionAndCheckInputTenserDesc(
    user_op::InferContext* ctx, const std::string& input_name,
    const Shape& input_shape_when_reduction_is_none) {
  const std::string reduction = ctx->Attr<std::string>("reduction");
  CHECK_OR_RETURN(LossReductionTypeIsRight(reduction));
  const auto& input_desc = ctx->InputTensorDesc(input_name, 0);
  if (reduction == "none") {
    CHECK_EQ_OR_RETURN(input_desc.shape(), input_shape_when_reduction_is_none);
  } else {
    CHECK_EQ_OR_RETURN(input_desc.shape(), Shape());
  }
  return Maybe<void>::Ok();
}

user_op::GetSbpFn GenLossForwardDefaultGetSbpFn(
    const std::function<void(user_op::UserOpSbpSignatureBuilder& builder)>& f) {
  return [=](user_op::SbpContext* ctx) -> Maybe<void> {
    const auto reduction = ctx->Attr<std::string>("reduction");
    auto builder = ctx->NewBuilder()
                       .Split(user_op::OpArg("input", 0), 0)
                       .Split(user_op::OpArg("target", 0), 0)
                       .Broadcast(user_op::OpArg("weight", 0));
    f(builder);
    if (reduction != "none") {
      builder.Broadcast(user_op::OpArg("out", 0));
    } else {
      builder.Split(user_op::OpArg("out", 0), 0);
    }
    builder.Build();
    return Maybe<void>::Ok();
  };
}

user_op::GetSbpFn GenLossBackwardDefaultGetSbpFn(
    const std::function<void(user_op::UserOpSbpSignatureBuilder& builder)>& f) {
  return [=](user_op::SbpContext* ctx) -> Maybe<void> {
    const auto& dy_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("dy", 0).shape();
    auto builder = ctx->NewBuilder()
                       .Split(user_op::OpArg("input", 0), 0)
                       .Split(user_op::OpArg("target", 0), 0)
                       .Broadcast(user_op::OpArg("weight", 0))
                       .Split(user_op::OpArg("dx", 0), 0);
    f(builder);
    if (dy_shape.NumAxes() == 0) {
      builder.Broadcast(user_op::OpArg("dy", 0));
    } else {
      builder.Split(user_op::OpArg("dy", 0), 0);
    }
    builder.Build();
    return Maybe<void>::Ok();
  };
}

}  // namespace oneflow
