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

/* static */ Maybe<void> ImageTargetResizeOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& in_desc = ctx->InputTensorDesc("in", 0);
  CHECK_OR_RETURN(in_desc.shape().NumAxes() == 1 && in_desc.shape().At(0) >= 1);
  user_op::TensorDesc* out_desc = ctx->MutOutputTensorDesc("out", 0);
  out_desc->set_shape(in_desc.shape());
  user_op::TensorDesc* size_desc = ctx->MutOutputTensorDesc("size", 0);
  size_desc->set_shape(Shape({in_desc.shape().elem_cnt(), 2}));
  user_op::TensorDesc* scale_desc = ctx->MutOutputTensorDesc("scale", 0);
  scale_desc->set_shape(Shape({in_desc.shape().elem_cnt(), 2}));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> ImageTargetResizeOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> ImageTargetResizeOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Split(ctx->inputs(), 0).Split(ctx->outputs(), 0).Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> ImageTargetResizeOp::CheckAttr(const user_op::UserOpDefWrapper& def,
                                                        const user_op::UserOpConfWrapper& conf) {
  bool check_failed = false;
  std::stringstream err;
  err << "Illegal attr value for " << conf.op_type_name() << " op, op_name: " << conf.op_name();
  const int32_t target_size = conf.attr<int32_t>("target_size");
  const int32_t max_size = conf.attr<int32_t>("max_size");
  if (target_size <= 0) {
    err << ", target_size: " << target_size << " (target_size must be greater than 0)";
    check_failed = true;
  }
  if (max_size < target_size) {
    err << ", max_size: " << max_size << " (max_size must be greater than 0)";
    check_failed = true;
  }
  if (check_failed) { return oneflow::Error::CheckFailedError() << err.str(); }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> ImageTargetResizeOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& in_desc = ctx->InputTensorDesc("in", 0);
  CHECK_OR_RETURN(in_desc.data_type() == DataType::kTensorBuffer);
  user_op::TensorDesc* out_desc = ctx->MutOutputTensorDesc("out", 0);
  out_desc->set_data_type(DataType::kTensorBuffer);
  user_op::TensorDesc* size_desc = ctx->MutOutputTensorDesc("size", 0);
  size_desc->set_data_type(DataType::kInt32);
  user_op::TensorDesc* scale_desc = ctx->MutOutputTensorDesc("scale", 0);
  scale_desc->set_data_type(DataType::kFloat);
  return Maybe<void>::Ok();
}

}  // namespace oneflow
