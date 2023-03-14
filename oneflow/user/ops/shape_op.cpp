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
#include "oneflow/core/framework/sbp_infer_util.h"
#include "oneflow/core/framework/user_op_conf.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

/*static*/ Maybe<void> ShapeOp::GetSbp(user_op::SbpContext* ctx) {
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> ShapeOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const user_op::TensorDesc& in_tensor_desc = ctx->InputTensorDesc("in", 0);
  user_op::TensorDesc* out_tensor_desc = ctx->MutOutputTensorDesc("out", 0);

  const Shape& in_shape = in_tensor_desc.shape();
  out_tensor_desc->set_shape(Shape{static_cast<long>(in_shape.size())});
  out_tensor_desc->set_data_type(DataType::kInt64);

  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> ShapeOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/*static*/ Maybe<void> ShapeOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, DataType::kInt64);
  return Maybe<void>::Ok();
}

}  // namespace oneflow

