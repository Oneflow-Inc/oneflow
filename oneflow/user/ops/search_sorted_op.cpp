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
  *ctx->OutputShape("out", 0) = ctx->InputShape("values", 0);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> SearchSortedOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> SearchSortedOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& in_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
  FOR_RANGE(int64_t, i, 0, in_tensor.shape().NumAxes() - 1) {
    ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> SearchSortedOp::CheckAttr(const user_op::UserOpDefWrapper& def,
                                              const user_op::UserOpConfWrapper& conf) {
  const std::string& side = conf.attr<std::string>("side");
  CHECK_OR_RETURN(side == "left" || side == "right") << "for searchsorted op, side can only be 'left' or 'right' but got " << side;
  const std::string& right = conf.attr<std::string>("right");
  CHECK_OR_RETURN((right == "True" and side == "right") or (right == "False" and side == "left")) << "for searchsorted op,  \
                  side and right can't be set to opposites, but got side of " << side << " while right was " << right;
  
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> SearchSortedOp::InferDataType(user_op::InferContext* ctx) {
  const bool& out_int32 = ctx->Attr<bool>("out_int32");
  if (out_int32) {
    *ctx->OutputDType("out", 0) = DataType::kInt32;
  } else {
    *ctx->OutputDType("out", 0) = DataType::kInt64;
  }
  return Maybe<void>::Ok();
}

}  // namespace oneflow
