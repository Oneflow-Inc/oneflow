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

/* static */ Maybe<void> SpmmCsrOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  bool transpose_a = ctx->Attr<bool>("transpose_a");
  bool transpose_b = ctx->Attr<bool>("transpose_b");
  CHECK_EQ_OR_RETURN(transpose_b, false)
      << ctx->op_name() << ": transpose for mat 2 is not implemented yet";

  const user_op::TensorDesc& b = ctx->InputTensorDesc("b", 0);

  int64_t c_row = transpose_a? ctx->Attr<int64_t>("a_num_cols"): ctx->Attr<int64_t>("a_num_rows");
  int64_t c_col = b.shape().At(1);
  ctx->SetOutputShape("out", 0,  Shape({c_row, c_col}));

  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> SpmmCsrOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> SpmmCsrOp::GetSbp(user_op::SbpContext* ctx) {
  // ctx->NewBuilder()
  //   .Split(user_op::OpArg("a_csr_row", 0), 0)
  //   .Split(user_op::OpArg("a_csr_col", 0), 0)
  //   .Split(user_op::OpArg("a_csr_val", 0), 0)
  //   .Broadcast(user_op::OpArg("b", 0))
  //   .Split(user_op::OpArg("out", 0), 0)
  //   .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> SpmmCsrOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("b", 0));

  return Maybe<void>::Ok();
}

}  // namespace oneflow
