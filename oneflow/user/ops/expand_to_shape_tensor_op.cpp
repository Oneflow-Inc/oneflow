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
#include "oneflow/core/job/nd_sbp_util.h"

namespace oneflow {

/* static */ Maybe<void> ExpandToShapeTensorOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const auto rank_of_shape_input = ctx->InputShape("shape", 0)[0].val();
  const auto rank = rank_of_shape_input == 0 ? 1 : rank_of_shape_input;
  Shape shape;
  for (int64_t i = 0; i < rank; i++) {
    shape.push_back(Dim::Unknown());
  }

  ctx->SetOutputShape("out", 0, shape);
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> ExpandToShapeTensorOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> ExpandToShapeTensorOp::GetSbp(user_op::SbpContext* ctx) {
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> ExpandToShapeTensorOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("in", 0));
  return Maybe<void>::Ok();
}

}  // namespace oneflow

