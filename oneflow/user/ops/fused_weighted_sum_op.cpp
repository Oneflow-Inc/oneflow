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

/* static */ Maybe<void> FusedWeightedSumOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const auto& in_0 = ctx->InputTensorDesc("in", 0);
  auto* out = ctx->MutOutputTensorDesc("out", 0);
  for (int64_t i = 1; i < ctx->input_size("in"); ++i) {
    const auto& cur_in = ctx->InputTensorDesc("in", i);
    CHECK_EQ_OR_RETURN(in_0.shape(), cur_in.shape())
        << Error::RuntimeError()
        << "inconsistent tensor size, expected all tensor to have the same shape, "
        << "but got " << in_0.shape().DebugStr() << " and " << cur_in.shape().DebugStr();
  }
  out->set_shape(in_0.shape());
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> FusedWeightedSumOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> FusedWeightedSumOp::GetSbp(user_op::SbpContext* ctx) {
  const int64_t num_axes = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0).shape().NumAxes();
  for (int64_t i = 0; i < num_axes; ++i) {
    ctx->NewBuilder().Split(ctx->inputs(), i).Split(user_op::OpArg("out", 0), i).Build();
  }
  ctx->NewBuilder().PartialSum(ctx->inputs()).PartialSum(user_op::OpArg("out", 0)).Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FusedWeightedSumOp::InferDataType(user_op::InferContext* ctx) {
  const auto& in_0 = ctx->InputTensorDesc("in", 0);
  auto* out = ctx->MutOutputTensorDesc("out", 0);
  const DataType data_type = in_0.data_type();
  for (int64_t i = 1; i < ctx->input_size("in"); ++i) {
    const auto& cur_in = ctx->InputTensorDesc("in", i);
    CHECK_EQ_OR_RETURN(cur_in.data_type(), data_type)
        << Error::RuntimeError() << ctx->op_name()
        << " expected all tenser to have same type, but found " << DataType_Name(cur_in.data_type())
        << " and " << DataType_Name(data_type);
  }
  out->set_data_type(data_type);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> FusedWeightedSumOp::CheckAttr(const user_op::UserOpDefWrapper&,
                                                     const user_op::UserOpConfWrapper& op_conf) {
  CHECK_OR_RETURN(op_conf.input_size("in") >= 2)
      << Error::RuntimeError()
      << "The number of input tensors should be greater than or equal to 2";
  return Maybe<void>::Ok();
}

}  // namespace oneflow
