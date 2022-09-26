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

/* static */ Maybe<void> SpmmCooOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  int64_t a_row = ctx->Attr<int64_t>("a_num_rows");
  const user_op::TensorDesc& b = ctx->InputTensorDesc("b", 0);
  int64_t b_col = b.shape().At(1);

  *ctx->MutOutputShape("out", 0) = {a_row, b_col};

  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> SpmmCooOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> SpmmCooOp::GetSbp(user_op::SbpContext* ctx) {
  // ctx->NewBuilder()
  //   .Split(user_op::OpArg("a_coo_row", 0), 0)
  //   .Split(user_op::OpArg("a_coo_col", 0), 0)
  //   .Split(user_op::OpArg("a_coo_val", 0), 0)
  //   .Broadcast(user_op::OpArg("b", 0))
  //   .Split(user_op::OpArg("out", 0), 0)
  //   .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> SpmmCooOp::InferDataType(user_op::InferContext* ctx) {
  *ctx->MutOutputDType("out", 0) = ctx->InputDType("b", 0);

  return Maybe<void>::Ok();
}

}  // namespace oneflow
