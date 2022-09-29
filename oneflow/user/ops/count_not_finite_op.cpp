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

/* static */ Maybe<void> CountNotFiniteOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  user_op::TensorDesc* y_desc = ctx->MutOutputTensorDesc("y", 0);
  y_desc->set_shape(Shape({1}));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> CountNotFiniteOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> CountNotFiniteOp::GetSbp(user_op::SbpContext* ctx) {
  const user_op::TensorDesc& x = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0);
  FOR_RANGE(int64_t, i, 0, x.shape().NumAxes()) {
    ctx->NewBuilder().Split(user_op::OpArg("x", 0), i).PartialSum(user_op::OpArg("y", 0)).Build();
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> CountNotFiniteOp::InferDataType(user_op::InferContext* ctx) {
  user_op::TensorDesc* y_desc = ctx->MutOutputTensorDesc("y", 0);
  y_desc->set_data_type(DataType::kInt64);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> MultiCountNotFiniteOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  user_op::TensorDesc* y_desc = ctx->MutOutputTensorDesc("y", 0);
  y_desc->set_shape(Shape({1}));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> MultiCountNotFiniteOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> MultiCountNotFiniteOp::GetSbp(user_op::SbpContext* ctx) {
  int64_t min_num_axes = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0).shape().NumAxes();
  for (int64_t i = 1; i < ctx->user_op_conf().input_size("x"); ++i) {
    min_num_axes = std::min(min_num_axes,
                            ctx->LogicalTensorDesc4InputArgNameAndIndex("x", i).shape().NumAxes());
  }
  for (int64_t i = 0; i < min_num_axes; ++i) {
    ctx->NewBuilder().Split(ctx->inputs(), i).PartialSum(user_op::OpArg("y", 0)).Build();
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> MultiCountNotFiniteOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& first_x_desc = ctx->InputTensorDesc("x", 0);
  for (const auto& in_arg_pair : ctx->inputs()) {
    const user_op::TensorDesc& x_desc = ctx->InputTensorDesc(in_arg_pair.first, in_arg_pair.second);
    CHECK_EQ_OR_RETURN(x_desc.data_type(), first_x_desc.data_type())
        << "InferDataType Failed. Expected " << DataType_Name(first_x_desc.data_type())
        << ", but got " << DataType_Name(x_desc.data_type());
  }
  user_op::TensorDesc* y_desc = ctx->MutOutputTensorDesc("y", 0);
  y_desc->set_data_type(DataType::kInt64);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> MultiCountNotFiniteOp::CheckAttr(const user_op::UserOpDefWrapper&,
                                                        const user_op::UserOpConfWrapper& op_conf) {
  CHECK_OR_RETURN(op_conf.input_size("x") >= 1);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
