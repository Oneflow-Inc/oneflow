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

/* static */ Maybe<void> MutableCastOnceOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& input_tensor_desc = ctx->InputTensorDesc("in", 0);
  user_op::TensorDesc* output_tensor_desc = ctx->MutOutputTensorDesc("out", 0);
  output_tensor_desc->set_shape(input_tensor_desc.shape());
  output_tensor_desc->set_is_dynamic(input_tensor_desc.is_dynamic());
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> MutableCastOnceOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> MutableCastOnceOp::GetSbp(user_op::SbpContext* ctx) {
  const auto& in_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
  for (int i = 0; i < in_tensor.shape().NumAxes(); ++i) {
    ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
  }
  ctx->NewBuilder().PartialSum(ctx->inputs()).PartialSum(ctx->outputs()).Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> MutableCastOnceOp::InferDataType(user_op::InferContext* ctx) {
  user_op::TensorDesc* output_tensor_desc = ctx->MutOutputTensorDesc("out", 0);
  output_tensor_desc->set_data_type(ctx->Attr<DataType>("dtype"));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
