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
  user_op::TensorDesc* mapped_label = ctx->OutputTensorDesc("mapped_label", 0);
  user_op::TensorDesc* sampled_weight = ctx->OutputTensorDesc("sampled_weight", 0);
  user_op::TensorDesc* sampled_label = ctx->OutputTensorDesc("sampled_label", 0);
  *mapped_label->mut_shape() = label.shape();
  *mapped_label->mut_is_dynamic() = label.is_dynamic();
  *sampled_weight->mut_shape() = weight.shape();
  sampled_weight->mut_shape()->Set(0, num_sample);
  *sampled_weight->mut_is_dynamic() = weight.is_dynamic();
  *sampled_label->mut_shape() = label.shape();
  sampled_label->mut_shape()->Set(0, num_sample);
  *sampled_label->mut_is_dynamic() = label.is_dynamic();
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
  user_op::TensorDesc* mapped_label = ctx->OutputTensorDesc("mapped_label", 0);
  user_op::TensorDesc* sampled_weight = ctx->OutputTensorDesc("sampled_weight", 0);
  user_op::TensorDesc* sampled_label = ctx->OutputTensorDesc("sampled_label", 0);
  *mapped_label->mut_shape() = label.shape();
  *mapped_label->mut_is_dynamic() = label.is_dynamic();
  *sampled_weight->mut_shape() = weight.shape();
  sampled_weight->mut_shape()->Set(0, num_sample_per_rank);
  *sampled_weight->mut_is_dynamic() = weight.is_dynamic();
  *sampled_label->mut_shape() = label.shape();
  sampled_label->mut_shape()->Set(0, num_sample_per_rank);
  *sampled_label->mut_is_dynamic() = label.is_dynamic();
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> DistributedPartialFcSampleOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->OutputDType("mapped_label", 0) = ctx->InputDType("label", 0);
  *ctx->OutputDType("sampled_weight", 0) = ctx->InputDType("weight", 0);
  *ctx->OutputDType("sampled_label", 0) = ctx->InputDType("label", 0);
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
      ctx->OutputTensorDesc("boxing_disabled_sampled_weight_diff", 0);
  *boxing_disabled_sampled_weight_diff->mut_shape() = ctx->InputShape("sampled_weight_diff", 0);
  CHECK_EQ_OR_RETURN(boxing_disabled_sampled_weight_diff->shape().At(0) % ctx->parallel_num(), 0);
  boxing_disabled_sampled_weight_diff->mut_shape()->Set(
      0, boxing_disabled_sampled_weight_diff->shape().At(0) / ctx->parallel_num());
  *boxing_disabled_sampled_weight_diff->mut_is_dynamic() =
      ctx->InputIsDynamic("sampled_weight_diff", 0);
  user_op::TensorDesc* boxing_disabled_sampled_label =
      ctx->OutputTensorDesc("boxing_disabled_sampled_label", 0);
  *boxing_disabled_sampled_label->mut_shape() = ctx->InputShape("sampled_label", 0);
  CHECK_EQ_OR_RETURN(boxing_disabled_sampled_label->shape().At(0) % ctx->parallel_num(), 0);
  boxing_disabled_sampled_label->mut_shape()->Set(
      0, boxing_disabled_sampled_label->shape().At(0) / ctx->parallel_num());
  *boxing_disabled_sampled_label->mut_is_dynamic() = ctx->InputIsDynamic("sampled_label", 0);
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> DistributedPartialFcSampleDisableBoxingOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  *ctx->MutOutputShape("boxing_disabled_sampled_weight_diff", 0) =
      ctx->InputShape("sampled_weight_diff", 0);
  *ctx->OutputIsDynamic("boxing_disabled_sampled_weight_diff", 0) =
      ctx->InputIsDynamic("sampled_weight_diff", 0);
  *ctx->MutOutputShape("boxing_disabled_sampled_label", 0) = ctx->InputShape("sampled_label", 0);
  *ctx->OutputIsDynamic("boxing_disabled_sampled_label", 0) =
      ctx->InputIsDynamic("sampled_label", 0);
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> DistributedPartialFcSampleDisableBoxingOp::InferDataType(
    user_op::InferContext* ctx) {
  *ctx->OutputDType("boxing_disabled_sampled_weight_diff", 0) =
      ctx->InputDType("sampled_weight_diff", 0);
  *ctx->OutputDType("boxing_disabled_sampled_label", 0) = ctx->InputDType("sampled_label", 0);
  return Maybe<void>::Ok();
}

REGISTER_USER_OP_GRAD("distributed_partial_fc_sample")
    .SetBackwardOpConfGenFn([](user_op::BackwardOpConfContext* ctx) -> Maybe<void> {
      const auto disable_boxing_op_name = ctx->FwOp().op_name() + "_disable_boxing";
      ctx->DefineOp(disable_boxing_op_name, [&ctx](user_op::BackwardOpBuilder& builder) {
        return builder.OpTypeName("distributed_partial_fc_sample_disable_boxing")
            .InputBind("sampled_weight_diff", ctx->FwOp().output_grad("sampled_weight", 0))
            .InputBind("sampled_label", ctx->FwOp().output("sampled_label", 0))
            .Output("boxing_disabled_sampled_weight_diff")
            .Output("boxing_disabled_sampled_label")
            .Build();
      });
      const auto unsorted_segment_sum_like_op_name =
          ctx->FwOp().op_name() + "_grad_unsorted_segment_sum_like";
      ctx->DefineOp(unsorted_segment_sum_like_op_name, [&ctx, &disable_boxing_op_name](
                                                           user_op::BackwardOpBuilder& builder) {
        return builder.OpTypeName("unsorted_segment_sum_like")
            .InputBind(
                "data",
                ctx->GetOp(disable_boxing_op_name).output("boxing_disabled_sampled_weight_diff", 0))
            .InputBind(
                "segment_ids",
                ctx->GetOp(disable_boxing_op_name).output("boxing_disabled_sampled_label", 0))
            .InputBind("like", ctx->FwOp().input("weight", 0))
            .Output("out")
            .Attr("axis", static_cast<int64_t>(0))
            .Build();
      });
      ctx->FwOp().InputGradBind(
          user_op::OpArg("weight", 0),
          [&ctx, &unsorted_segment_sum_like_op_name]() -> const std::string& {
            return ctx->GetOp(unsorted_segment_sum_like_op_name).output("out", 0);
          });
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
