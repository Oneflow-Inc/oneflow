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

namespace {

std::function<Maybe<void>(const std::string&)> MakeSetOutTensorDescFn(user_op::InferContext* ctx,
                                                                      const Shape& shape) {
  return [=](const std::string& bn) -> Maybe<void> {
    if (ctx->has_output(bn, 0)) {
      auto* tensor_desc = ctx->MutOutputTensorDesc(bn, 0);
      CHECK_OR_RETURN(tensor_desc != nullptr) << "output tensordesc of " << bn << " is null.";
      tensor_desc->set_shape(shape);
    }
    return Maybe<void>::Ok();
  };
}

std::function<Maybe<void>(const std::string&)> MakeSetOutDataTypeFn(user_op::InferContext* ctx,
                                                                    DataType data_type) {
  return [=](const std::string& bn) -> Maybe<void> {
    if (ctx->has_output(bn, 0)) {
      auto* tensor_desc = ctx->MutOutputTensorDesc(bn, 0);
      CHECK_OR_RETURN(tensor_desc != nullptr) << "output tensordesc of " << bn << " is null.";
      tensor_desc->set_data_type(data_type);
    }
    return Maybe<void>::Ok();
  };
}

}  // namespace

/* static */ Maybe<void> BatchNormGatherStatsWithCountsOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  const auto& mean = ctx->InputTensorDesc("mean", 0);
  const Shape& mean_shape = mean.shape();
  const Shape param_shape({mean_shape.At(1)});
  const auto SetOutTensorDesc = MakeSetOutTensorDescFn(ctx, param_shape);
  JUST(SetOutTensorDesc("global_mean"));
  JUST(SetOutTensorDesc("global_invstd"));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> BatchNormGatherStatsWithCountsOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> BatchNormGatherStatsWithCountsOp::GetSbp(user_op::SbpContext* ctx) {
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> BatchNormGatherStatsWithCountsOp::InferDataType(
    user_op::InferContext* ctx) {
  const auto& x = ctx->InputTensorDesc("input", 0);
  const auto data_type = x.data_type();
  const DataType out_data_type = data_type == DataType::kFloat16 ? DataType::kFloat : data_type;
  const auto SetOutDataType = MakeSetOutDataTypeFn(ctx, out_data_type);
  JUST(SetOutDataType("global_mean"));
  JUST(SetOutDataType("global_invstd"));
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> BatchNormGatherStatsWithCountsOp::ModifyInputArg(
    const GetInputArgModifier& GetInputArgModifierFn, const user_op::UserOpConfWrapper& conf) {
  if (conf.has_input("running_mean", 0)) {
    CHECK_OR_RETURN(conf.has_input("running_var", 0))
        << "running_mean and running_var should be provided as inputs in the same time.";
    user_op::InputArgModifier* running_mean_modifier = GetInputArgModifierFn("running_mean", 0);
    CHECK_OR_RETURN(running_mean_modifier != nullptr)
        << "input arg modifier of running_mean is null.";
    running_mean_modifier->set_is_mutable(true);
    running_mean_modifier->set_requires_grad(false);
    user_op::InputArgModifier* running_var_modifier = GetInputArgModifierFn("running_var", 0);
    CHECK_OR_RETURN(running_var_modifier != nullptr)
        << "input arg modifier of running_var is null.";
    running_var_modifier->set_is_mutable(true);
    running_var_modifier->set_requires_grad(false);
  }
  return Maybe<void>::Ok();
}

}  // namespace oneflow
