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

/*static*/ Maybe<void> SortOp::GetSbp(user_op::SbpContext* ctx) {
  // The current implementation can only do sort in the last dimension and should use Broadcast
  // (by default) instead of Split for that dimension
  const user_op::TensorDesc& in_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
  FOR_RANGE(int64_t, i, 0, in_tensor.shape().NumAxes() - 1) {
    ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
  }
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SortOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  ctx->SetOutputShape("out", 0, ctx->InputShape("in", 0));
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SortOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}
/*static*/ Maybe<void> SortOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("in", 0));
  return Maybe<void>::Ok();
}
/*static*/ Maybe<void> SortOp::CheckAttr(const user_op::UserOpDefWrapper&,
                                         const user_op::UserOpConfWrapper& op_conf) {
  const std::string& direction = op_conf.attr<std::string>("direction");
  CHECK_OR_RETURN(direction == "ASCENDING" || direction == "DESCENDING")
      << Error::RuntimeError()
      << "The input direction parameter value is expected to be ASCENDING or DESCENDING, "
      << "but found it to be " << direction;
  return Maybe<void>::Ok();
}

}  // namespace oneflow
