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

/* static */ Maybe<void> BatchNormStatsOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const auto& x = ctx->InputTensorDesc("input", 0);
  const Shape& x_shape = x.shape();
  const auto axis = ctx->Attr<int32_t>("axis");
  CHECK_GE_OR_RETURN(axis, 0) << "channel axis should be larger than 0";
  CHECK_LT_OR_RETURN(axis, x_shape.NumAxes())
      << "channel axis should be less than " << x_shape.NumAxes();
  const Shape param_shape({x_shape.At(axis)});
  const auto SetOutTensorDesc = MakeSetOutTensorDescFn(ctx, param_shape);
  JUST(SetOutTensorDesc("mean"));
  JUST(SetOutTensorDesc("invstd"));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> BatchNormStatsOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> BatchNormStatsOp::GetSbp(user_op::SbpContext* ctx) {
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> BatchNormStatsOp::InferDataType(user_op::InferContext* ctx) {
  const auto& x = ctx->InputTensorDesc("input", 0);
  const auto data_type = x.data_type();
  const DataType out_data_type = data_type == DataType::kFloat16 ? DataType::kFloat : data_type;
  const auto SetOutDataType = MakeSetOutDataTypeFn(ctx, out_data_type);
  JUST(SetOutDataType("mean"));
  JUST(SetOutDataType("invstd"));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
