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

/*static*/ Maybe<void> DistributedPartialFcSampleOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("weight", 0), 0)
      .Broadcast(user_op::OpArg("label", 0))
      .Broadcast(user_op::OpArg("mapped_label", 0))
      .Split(user_op::OpArg("sampled_label", 0), 0)
      .Split(user_op::OpArg("sampled_weight", 0), 0)
      .Build();
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> DistributedPartialFcSampleOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  const int64_t num_sample = ctx->Attr<int64_t>("num_sample");
  const user_op::TensorDesc& weight = ctx->InputTensorDesc("weight", 0);
  const user_op::TensorDesc& label = ctx->InputTensorDesc("label", 0);
  user_op::TensorDesc* mapped_label = ctx->MutOutputTensorDesc("mapped_label", 0);
  user_op::TensorDesc* sampled_weight = ctx->MutOutputTensorDesc("sampled_weight", 0);
  user_op::TensorDesc* sampled_label = ctx->MutOutputTensorDesc("sampled_label", 0);
  mapped_label->set_shape(label.shape());
  mapped_label->set_is_dynamic(label.is_dynamic());
  Shape sampled_weight_shape = weight.shape();
  sampled_weight_shape.Set(0, num_sample);
  sampled_weight->set_shape(sampled_weight_shape);
  sampled_weight->set_is_dynamic(weight.is_dynamic());
  Shape sampled_label_shape = label.shape();
  sampled_label_shape.Set(0, num_sample);
  sampled_label->set_shape(sampled_label_shape);
  sampled_label->set_is_dynamic(label.is_dynamic());
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> DistributedPartialFcSampleOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  const int64_t num_sample = ctx->Attr<int64_t>("num_sample");
  const int64_t parallel_num = ctx->parallel_ctx().parallel_num();
  CHECK_EQ_OR_RETURN(num_sample % parallel_num, 0);
  const int64_t num_sample_per_rank = num_sample / parallel_num;
  const user_op::TensorDesc& weight = ctx->InputTensorDesc("weight", 0);
  const user_op::TensorDesc& label = ctx->InputTensorDesc("label", 0);
  user_op::TensorDesc* mapped_label = ctx->MutOutputTensorDesc("mapped_label", 0);
  user_op::TensorDesc* sampled_weight = ctx->MutOutputTensorDesc("sampled_weight", 0);
  user_op::TensorDesc* sampled_label = ctx->MutOutputTensorDesc("sampled_label", 0);
  mapped_label->set_shape(label.shape());
  mapped_label->set_is_dynamic(label.is_dynamic());
  Shape sampled_weight_shape = weight.shape();
  sampled_weight_shape.Set(0, num_sample_per_rank);
  sampled_weight->set_shape(sampled_weight_shape);
  sampled_weight->set_is_dynamic(weight.is_dynamic());
  Shape sampled_label_shape = label.shape();
  sampled_label_shape.Set(0, num_sample_per_rank);
  sampled_label->set_shape(sampled_label_shape);
  sampled_label->set_is_dynamic(label.is_dynamic());
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> DistributedPartialFcSampleOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("mapped_label", 0, ctx->InputDType("label", 0));
  ctx->SetOutputDType("sampled_weight", 0, ctx->InputDType("weight", 0));
  ctx->SetOutputDType("sampled_label", 0, ctx->InputDType("label", 0));
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> DistributedPartialFcSampleOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper&) {
  user_op::InputArgModifier* label_modifier = GetInputArgModifierFn("label", 0);
  CHECK_NOTNULL_OR_RETURN(label_modifier);
  label_modifier->set_requires_grad(false);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> DistributedPartialFcSampleDisableBoxingOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("sampled_weight_diff", 0), 0)
      .Split(user_op::OpArg("sampled_label", 0), 0)
      .Broadcast(user_op::OpArg("boxing_disabled_sampled_weight_diff", 0))
      .Broadcast(user_op::OpArg("boxing_disabled_sampled_label", 0))
      .Build();
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> DistributedPartialFcSampleDisableBoxingOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  user_op::TensorDesc* boxing_disabled_sampled_weight_diff =
      ctx->MutOutputTensorDesc("boxing_disabled_sampled_weight_diff", 0);
  Shape boxing_disabled_sampled_weight_diff_shape = ctx->InputShape("sampled_weight_diff", 0);
  CHECK_EQ_OR_RETURN(boxing_disabled_sampled_weight_diff_shape.At(0) % ctx->parallel_num(), 0);
  boxing_disabled_sampled_weight_diff_shape.Set(
      0, boxing_disabled_sampled_weight_diff_shape.At(0) / ctx->parallel_num());
  boxing_disabled_sampled_weight_diff->set_shape(boxing_disabled_sampled_weight_diff_shape);
  boxing_disabled_sampled_weight_diff->set_is_dynamic(
      ctx->InputIsDynamic("sampled_weight_diff", 0));
  user_op::TensorDesc* boxing_disabled_sampled_label =
      ctx->MutOutputTensorDesc("boxing_disabled_sampled_label", 0);
  Shape boxing_disabled_sampled_label_shape = ctx->InputShape("sampled_label", 0);
  ;
  CHECK_EQ_OR_RETURN(boxing_disabled_sampled_label_shape.At(0) % ctx->parallel_num(), 0);
  boxing_disabled_sampled_label_shape.Set(
      0, boxing_disabled_sampled_label_shape.At(0) / ctx->parallel_num());
  boxing_disabled_sampled_label->set_shape(boxing_disabled_sampled_label_shape);
  boxing_disabled_sampled_label->set_is_dynamic(ctx->InputIsDynamic("sampled_label", 0));
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> DistributedPartialFcSampleDisableBoxingOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  ctx->SetOutputShape("boxing_disabled_sampled_weight_diff", 0,
                      ctx->InputShape("sampled_weight_diff", 0));
  ctx->SetOutputIsDynamic("boxing_disabled_sampled_weight_diff", 0,
                          ctx->InputIsDynamic("sampled_weight_diff", 0));
  ctx->SetOutputShape("boxing_disabled_sampled_label", 0, ctx->InputShape("sampled_label", 0));
  ctx->SetOutputIsDynamic("boxing_disabled_sampled_label", 0,
                          ctx->InputIsDynamic("sampled_label", 0));
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> DistributedPartialFcSampleDisableBoxingOp::InferDataType(
    user_op::InferContext* ctx) {
  ctx->SetOutputDType("boxing_disabled_sampled_weight_diff", 0,
                      ctx->InputDType("sampled_weight_diff", 0));
  ctx->SetOutputDType("boxing_disabled_sampled_label", 0, ctx->InputDType("sampled_label", 0));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
