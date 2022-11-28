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

/*static*/ auto FlipOp::InferLogicalTensorDesc(user_op::InferContext* ctx) -> Maybe<void> {
  const user_op::TensorDesc& x_desc = ctx->InputTensorDesc("x", 0);
  const int input_dims = x_desc.shape().NumAxes();
  const std::vector<int32_t> dims = ctx->Attr<std::vector<int32_t>>("dims");
  CHECK_OR_RETURN(dims.size() <= input_dims) << "len of dims must less than len of input tensor";
  for (auto x : dims) { CHECK_OR_RETURN(x < input_dims) << "dims parameter is illegal."; }
  user_op::TensorDesc* y_desc = ctx->MutOutputTensorDesc("y", 0);
  y_desc->set_shape(x_desc.shape());
  return Maybe<void>::Ok();
}
/*static*/ auto FlipOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) -> Maybe<void> {
  return FlipOp::InferLogicalTensorDesc(ctx);
}
/*static*/ auto FlipOp::GetSbp(user_op::SbpContext* ctx) -> Maybe<void> {
  const user_op::TensorDesc& x_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  const std::vector<int32_t> dims = ctx->Attr<std::vector<int32_t>>("dims");
  FOR_RANGE(int64_t, i, 0, x_tensor.shape().NumAxes()) {
    bool flag = true;
    for (auto x : dims) {
      if (x == i) {
        flag = false;
        break;
      }
    }
    if (flag) {
      ctx->NewBuilder().Split(user_op::OpArg("x", 0), i).Split(user_op::OpArg("y", 0), i).Build();
    }
  }
  return Maybe<void>::Ok();
}
/*static*/ auto FlipOp::InferDataType(user_op::InferContext* ctx) -> Maybe<void> {
  ctx->SetOutputDType("y", 0, ctx->InputDType("x", 0));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
