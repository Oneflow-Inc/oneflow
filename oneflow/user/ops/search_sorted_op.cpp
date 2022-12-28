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

/* static */ Maybe<void> SearchSortedOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  ctx->SetOutputShape("out", 0, ctx->InputShape("values", 0));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> SearchSortedOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> SearchSortedOp::GetSbp(user_op::SbpContext* ctx) {
  // The current implementation can only do arg_sort in the last dimension and should use
  // Broadcast (by default) instead of Split for that dimension
  const user_op::TensorDesc& in_tensor =
      ctx->LogicalTensorDesc4InputArgNameAndIndex("sorted_sequence", 0);
  FOR_RANGE(int64_t, i, 0, in_tensor.shape().NumAxes() - 1) {
    ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> SearchSortedOp::CheckAttr(const user_op::UserOpDefWrapper& def,
                                                   const user_op::UserOpConfWrapper& conf) {
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> SearchSortedOp::InferDataType(user_op::InferContext* ctx) {
  const bool& out_int32 = ctx->Attr<bool>("out_int32");
  if (out_int32) {
    ctx->SetOutputDType("out", 0, DataType::kInt32);
  } else {
    ctx->SetOutputDType("out", 0, DataType::kInt64);
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> SearchSortedScalarOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  ctx->SetOutputShape("out", 0, Shape({}));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> SearchSortedScalarOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> SearchSortedScalarOp::GetSbp(user_op::SbpContext* ctx) {
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> SearchSortedScalarOp::CheckAttr(const user_op::UserOpDefWrapper& def,
                                                         const user_op::UserOpConfWrapper& conf) {
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> SearchSortedScalarOp::InferDataType(user_op::InferContext* ctx) {
  const bool& out_int32 = ctx->Attr<bool>("out_int32");
  if (out_int32) {
    ctx->SetOutputDType("out", 0, DataType::kInt32);
  } else {
    ctx->SetOutputDType("out", 0, DataType::kInt64);
  }
  return Maybe<void>::Ok();
}

}  // namespace oneflow
