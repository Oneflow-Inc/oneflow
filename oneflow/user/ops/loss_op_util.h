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
#ifndef ONEFLOW_USER_OPS_LOSS_OP_UTIL_H_
#define ONEFLOW_USER_OPS_LOSS_OP_UTIL_H_

#include "oneflow/core/framework/framework.h"

namespace oneflow {

inline bool LossReductionTypeIsRight(const std::string& reduction) {
  if ((reduction != "none") && (reduction != "sum") && (reduction != "mean")) { return false; }
  return true;
}

inline Maybe<void> CheckLossReductionAndInferOutputTenserDesc(
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

inline Maybe<void> CheckLossReductionAndCheckInputTenserDesc(
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
}  // namespace oneflow

#endif  // ONEFLOW_USER_OPS_LOSS_OP_UTIL_H_
