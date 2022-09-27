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

Maybe<void> InferTensorDesc(user_op::InferContext* ctx) {
  const Shape& input_shape = ctx->InputShape("input", 0);
  user_op::TensorDesc* output_desc = ctx->MutOutputTensorDesc("output", 0);
  output_desc->set_shape(Shape({input_shape.elem_cnt(), input_shape.NumAxes()}));
  output_desc->set_is_dynamic(true);
  user_op::TensorDesc* output_size_desc = ctx->MutOutputTensorDesc("output_size", 0);
  output_size_desc->set_shape(Shape({1}));
  return Maybe<void>::Ok();
}

}  // namespace

/* static */ Maybe<void> ArgwhereOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return InferTensorDesc(ctx);
}

/*static*/ Maybe<void> ArgwhereOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> ArgwhereOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}

/* static */ Maybe<void> ArgwhereOp::InferDataType(user_op::InferContext* ctx) {
  const DataType dtype = ctx->Attr<DataType>("dtype");
  user_op::TensorDesc* output_desc = ctx->MutOutputTensorDesc("output", 0);
  output_desc->set_data_type(dtype);
  user_op::TensorDesc* output_size_desc = ctx->MutOutputTensorDesc("output_size", 0);
  output_size_desc->set_data_type(dtype);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
