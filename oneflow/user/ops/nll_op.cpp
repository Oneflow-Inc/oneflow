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
Maybe<void> InferTensorDescFn(user_op::InferContext* ctx) {
  const auto& input_desc = ctx->InputTensorDesc("input", 0);
  const auto& target_desc = ctx->InputTensorDesc("target", 0);
  // TODO weight
  CHECK_EQ_OR_RETURN(input_desc.is_dynamic(), target_desc.is_dynamic());
  CHECK_GE_OR_RETURN(input_desc.shape().NumAxes(), 2);
  const int64_t num_out_axes = input_desc.shape().NumAxes() - 1;
  CHECK_EQ_OR_RETURN(target_desc.shape().NumAxes(), num_out_axes);
  FOR_RANGE(int64_t, i, 0, num_out_axes) {
    CHECK_EQ_OR_RETURN(input_desc.shape().At(i), target_desc.shape().At(i));
  }
  user_op::TensorDesc* out_desc = ctx->OutputTensorDesc("out", 0);
  *out_desc->mut_is_dynamic() = input_desc.is_dynamic();
  *out_desc->mut_shape() = target_desc.shape();
  return Maybe<void>::Ok();
}

Maybe<void> InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& target_desc = ctx->InputTensorDesc("target", 0);
  CHECK_OR_RETURN(IsIndexDataType(target_desc.data_type()));
  *ctx->OutputDType("out", 0) = ctx->InputDType("input", 0);
  const bool has_weight = ctx->Attr<bool>("has_weight");
  if (has_weight) { *ctx->OutputDType("total_weight", 0) = ctx->InputDType("weight", 0); }
  return Maybe<void>::Ok();
}

Maybe<void> InferGradTensorDescFn(user_op::InferContext* ctx) {
  const auto& input_desc = ctx->InputTensorDesc("input", 0);
  const auto& target_desc = ctx->InputTensorDesc("target", 0);
  // TODO weight
  CHECK_EQ_OR_RETURN(input_desc.is_dynamic(), target_desc.is_dynamic());
  CHECK_GE_OR_RETURN(input_desc.shape().NumAxes(), 2);
  const int64_t num_out_axes = input_desc.shape().NumAxes() - 1;
  CHECK_EQ_OR_RETURN(target_desc.shape().NumAxes(), num_out_axes);
  FOR_RANGE(int64_t, i, 0, num_out_axes) {
    CHECK_EQ_OR_RETURN(input_desc.shape().At(i), target_desc.shape().At(i));
  }
  user_op::TensorDesc* dx_desc = ctx->OutputTensorDesc("dx", 0);
  *dx_desc->mut_is_dynamic() = input_desc.is_dynamic();
  *dx_desc->mut_shape() = input_desc.shape();
  return Maybe<void>::Ok();
}

Maybe<void> InferGradDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& target_desc = ctx->InputTensorDesc("target", 0);
  CHECK_OR_RETURN(IsIndexDataType(target_desc.data_type()));
  *ctx->OutputDType("dx", 0) = ctx->InputDType("dy", 0);
  return Maybe<void>::Ok();
}
}  // namespace

REGISTER_USER_OP("nll")
    .Input("input")
    .Input("target")
    .OptionalInput("weight")
    .Output("out")
    .OptionalOutput("total_weight")
    .Attr<int64_t>("ignore_index")
    .Attr<bool>("has_weight")
    .SetTensorDescInferFn(InferTensorDescFn)
    .SetInputArgModifyFn([](user_op::GetInputArgModifier GetInputArgModifierFn,
                            const user_op::UserOpConfWrapper&) -> Maybe<void> {
      user_op::InputArgModifier* target_modifier = GetInputArgModifierFn("target", 0);
      CHECK_OR_RETURN(target_modifier != nullptr);
      target_modifier->set_requires_grad(false);
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn(InferDataType)
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder()
          .Split(user_op::OpArg("input", 0), 0)
          .Split(user_op::OpArg("target", 0), 0)
          .Broadcast(user_op::OpArg("weight", 0))
          .Split(user_op::OpArg("out", 0), 0)
          .Broadcast(user_op::OpArg("total_weight", 0))
          .Build();
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("nll_grad")
    .Input("input")
    .Input("target")
    .OptionalInput("weight")
    .Input("dy")
    .Output("dx")
    .Attr<int64_t>("ignore_index")
    .Attr<bool>("has_weight")
    .SetTensorDescInferFn(InferGradTensorDescFn)
    .SetDataTypeInferFn(InferGradDataType)
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder()
          .Split(user_op::OpArg("input", 0), 0)
          .Split(user_op::OpArg("target", 0), 0)
          .Broadcast(user_op::OpArg("weight", 0))
          .Split(user_op::OpArg("dy", 0), 0)
          .Split(user_op::OpArg("dx", 0), 0)
          .Build();
      return Maybe<void>::Ok();
    });
}  // namespace oneflow
