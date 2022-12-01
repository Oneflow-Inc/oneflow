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

/* static */ Maybe<void> NLLOp::InferDataType(user_op::InferContext* ctx) {
  CHECK_OR_RETURN(IsIndexDataType(ctx->InputDType("target", 0)))
      << ctx->op_name() << ": expected target being integer type";

  DataType input_dtype = ctx->InputDType("input", 0);
  if (ctx->has_input("weight", 0)) {
    DataType weight_dtype = ctx->InputDType("weight", 0);
    CHECK_EQ_OR_RETURN(weight_dtype, input_dtype) << ctx->op_name() << ": expected weight dtype "
                                                  << input_dtype << ", but got " << weight_dtype;
  }

  ctx->SetOutputDType("output", 0, input_dtype);
  ctx->SetOutputDType("out_weight", 0, input_dtype);

  return Maybe<void>::Ok();
}

/* static */ Maybe<void> NLLOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const auto& input_desc = ctx->InputTensorDesc("input", 0);
  const auto& target_desc = ctx->InputTensorDesc("target", 0);

  const bool is_dynamic = input_desc.is_dynamic();
  CHECK_EQ_OR_RETURN(target_desc.is_dynamic(), is_dynamic)
      << ctx->op_name() << ": expected the same dynamic with input and target";
  const int64_t K = input_desc.shape().NumAxes();
  CHECK_GE_OR_RETURN(K, 2) << ctx->op_name() << ": expected 2 or more dimensions for input";
  CHECK_EQ_OR_RETURN(target_desc.shape().NumAxes(), K - 1)
      << ctx->op_name() << ": expected 1 less diemensions than input for target";
  const int64_t N = target_desc.shape().elem_cnt();
  const int64_t C = input_desc.shape().At(input_desc.shape().NumAxes() - 1);
  CHECK_EQ_OR_RETURN(input_desc.shape().elem_cnt(), N * C)
      << ctx->op_name() << ": expected input size " << input_desc.shape().ToString()
      << " to match target size " << target_desc.shape().ToString();

  if (ctx->has_input("weight", 0)) {
    const auto& weight_desc = ctx->InputTensorDesc("weight", 0);
    CHECK_EQ_OR_RETURN(weight_desc.is_dynamic(), is_dynamic)
        << ctx->op_name() << ": expected the same dynamic with input and weight";
    CHECK_EQ_OR_RETURN(weight_desc.shape().elem_cnt(), C)
        << ctx->op_name() << ": expected weight size " << C << ", got "
        << weight_desc.shape().ToString();
  }

  user_op::TensorDesc* output_desc = ctx->MutOutputTensorDesc("output", 0);
  output_desc->set_is_dynamic(is_dynamic);
  output_desc->set_shape(Shape({N}));

  user_op::TensorDesc* out_weight_desc = ctx->MutOutputTensorDesc("out_weight", 0);
  out_weight_desc->set_is_dynamic(is_dynamic);
  out_weight_desc->set_shape(Shape({N}));

  return Maybe<void>::Ok();
}

/* static */ Maybe<void> NLLOp::GetSbp(user_op::SbpContext* ctx) {
  // split batch dim
  auto builder1 = ctx->NewBuilder()
                      .Split(user_op::OpArg("input", 0), 0)
                      .Split(user_op::OpArg("target", 0), 0)
                      .Split(user_op::OpArg("output", 0), 0)
                      .Split(user_op::OpArg("out_weight", 0), 0);
  if (ctx->user_op_conf().has_input("weight", 0)) {
    builder1.Broadcast(user_op::OpArg("weight", 0));
  }
  builder1.Build();

  // split class dim
  const auto& shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("input", 0).shape();
  auto builder2 = ctx->NewBuilder()
                      .Split(user_op::OpArg("input", 0), shape.NumAxes() - 1)
                      .Broadcast(user_op::OpArg("target", 0))
                      .PartialSum(user_op::OpArg("output", 0))
                      .PartialSum(user_op::OpArg("out_weight", 0));
  if (ctx->user_op_conf().has_input("weight", 0)) {
    builder2.Split(user_op::OpArg("weight", 0), 0);
  }
  builder2.Build();

  return Maybe<void>::Ok();
}

/* static */ Maybe<void> NLLOp::ModifyInputArg(const GetInputArgModifier& GetInputArgModifierFn,
                                               const user_op::UserOpConfWrapper& conf) {
  user_op::InputArgModifier* target_modifier = GetInputArgModifierFn("target", 0);
  CHECK_OR_RETURN(target_modifier != nullptr);
  target_modifier->set_requires_grad(false);
  if (conf.has_input("weight", 0)) {
    auto* weight_modifier = GetInputArgModifierFn("weight", 0);
    if (weight_modifier) { weight_modifier->set_requires_grad(false); }
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> NLLGradOp::InferDataType(user_op::InferContext* ctx) {
  CHECK_OR_RETURN(IsIndexDataType(ctx->InputDType("target", 0)))
      << ctx->op_name() << ": expected target being integer type";

  DataType input_dtype = ctx->InputDType("input", 0);
  CHECK_EQ_OR_RETURN(ctx->InputDType("out_grad", 0), input_dtype)
      << ctx->op_name() << ": expected out_grad dtype " << input_dtype << ", got "
      << ctx->InputDType("out_grad", 0);

  if (ctx->has_input("weight", 0)) {
    CHECK_EQ_OR_RETURN(ctx->InputDType("weight", 0), input_dtype)
        << ctx->op_name() << ": expected weight dtype " << input_dtype << ", got "
        << ctx->InputDType("weight", 0);
  }

  ctx->SetOutputDType("in_grad", 0, input_dtype);

  return Maybe<void>::Ok();
}

/* static */ Maybe<void> NLLGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const auto& input_desc = ctx->InputTensorDesc("input", 0);
  const auto& target_desc = ctx->InputTensorDesc("target", 0);
  const auto& out_grad_desc = ctx->InputTensorDesc("out_grad", 0);

  bool is_dynamic = input_desc.is_dynamic();
  CHECK_EQ_OR_RETURN(target_desc.is_dynamic(), is_dynamic)
      << ctx->op_name() << ": expected target dynamic " << is_dynamic;
  CHECK_EQ_OR_RETURN(out_grad_desc.is_dynamic(), is_dynamic)
      << ctx->op_name() << ": expected out_grad dynamic " << is_dynamic;

  const int64_t N = target_desc.shape().elem_cnt();
  CHECK_EQ_OR_RETURN(out_grad_desc.shape().elem_cnt(), N)
      << ctx->op_name() << ": expected out_grad size " << N << ", got "
      << out_grad_desc.shape().ToString();

  const int64_t C = input_desc.shape().At(input_desc.shape().NumAxes() - 1);
  CHECK_EQ_OR_RETURN(input_desc.shape().elem_cnt(), N * C)
      << ctx->op_name() << ": expected input size " << N << ", got "
      << input_desc.shape().ToString();

  if (ctx->has_input("weight", 0)) {
    const auto& weight_desc = ctx->InputTensorDesc("weight", 0);
    CHECK_EQ_OR_RETURN(weight_desc.shape().elem_cnt(), C)
        << ctx->op_name() << ": expected weight size " << C << ", got "
        << weight_desc.shape().ToString();
  }

  user_op::TensorDesc* in_grad_desc = ctx->MutOutputTensorDesc("in_grad", 0);
  in_grad_desc->set_is_dynamic(is_dynamic);
  in_grad_desc->set_shape(input_desc.shape());

  return Maybe<void>::Ok();
}

/* static */ Maybe<void> NLLGradOp::GetSbp(user_op::SbpContext* ctx) {
  // split batch dim
  auto builder1 = ctx->NewBuilder()
                      .Split(user_op::OpArg("input", 0), 0)
                      .Split(user_op::OpArg("target", 0), 0)
                      .Split(user_op::OpArg("out_grad", 0), 0)
                      .Split(user_op::OpArg("in_grad", 0), 0);
  if (ctx->user_op_conf().has_input("weight", 0)) {
    builder1.Broadcast(user_op::OpArg("weight", 0));
  }
  builder1.Build();

  // split class dim
  const auto& shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("input", 0).shape();
  auto builder2 = ctx->NewBuilder()
                      .Split(user_op::OpArg("input", 0), shape.NumAxes() - 1)
                      .Broadcast(user_op::OpArg("target", 0))
                      .Broadcast(user_op::OpArg("out_grad", 0))
                      .Split(user_op::OpArg("in_grad", 0), shape.NumAxes() - 1);
  if (ctx->user_op_conf().has_input("weight", 0)) {
    builder2.Split(user_op::OpArg("weight", 0), 0);
  }
  builder2.Build();

  return Maybe<void>::Ok();
}

}  // namespace oneflow
