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

/*static*/ Maybe<void> OneRecReaderOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  user_op::TensorDesc* out_tensor = ctx->OutputTensorDesc("out", 0);
  int64_t batch_size = ctx->Attr<int64_t>("batch_size");
  *out_tensor->mut_shape() = Shape({batch_size});
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> OneRecReaderOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->MutOutputDType("out", 0) = DataType::kTensorBuffer;
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> OneRecReaderOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Split(ctx->outputs(), 0).Build();
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> OneRecReaderOp::InferNdSbp(user_op::InferNdSbpFnContext* ctx) {
  SbpParallel default_sbp;
  default_sbp.mutable_split_parallel()->set_axis(0);
  return user_op::InferNdSbp4SrcOp(ctx, default_sbp);
}

}  // namespace oneflow
