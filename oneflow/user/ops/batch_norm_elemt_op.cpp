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

/* static */ Maybe<void> BatchNormElemtOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const auto& x = ctx->InputTensorDesc("input", 0);
  const Shape& x_shape = x.shape();
  const auto SetOutTensorDesc = MakeSetOutTensorDescFn(ctx, x_shape);
  JUST(SetOutTensorDesc("output"));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> BatchNormElemtOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> BatchNormElemtOp::GetSbp(user_op::SbpContext* ctx) {
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> BatchNormElemtOp::InferDataType(user_op::InferContext* ctx) {
  const auto& x = ctx->InputTensorDesc("input", 0);
  const auto data_type = x.data_type();
  const DataType out_data_type = data_type == DataType::kFloat16 ? DataType::kFloat : data_type;
  const auto SetOutDataType = MakeSetOutDataTypeFn(ctx, out_data_type);
  JUST(SetOutDataType("output"));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
