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
  user_op::TensorDesc* output_desc = ctx->MutOutputTensorDesc("out", 0);
  const int64_t size = ctx->Attr<int64_t>("size");
  output_desc->set_shape(Shape({size}));
  return Maybe<void>::Ok();
}

}  // namespace

/* static */ Maybe<void> BinCountOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return InferTensorDesc(ctx);
}

/*static*/ Maybe<void> BinCountOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> BinCountOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}

/* static */ Maybe<void> BinCountOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& input_desc = ctx->InputTensorDesc("in", 0);
  user_op::TensorDesc* output_desc = ctx->MutOutputTensorDesc("out", 0);
  if (ctx->has_input("weight", 0)) {
    const user_op::TensorDesc& weight_desc = ctx->InputTensorDesc("weight", 0);
    output_desc->set_data_type(weight_desc.data_type());
  } else {
    output_desc->set_data_type(input_desc.data_type());
  }
  return Maybe<void>::Ok();
}

}  // namespace oneflow
