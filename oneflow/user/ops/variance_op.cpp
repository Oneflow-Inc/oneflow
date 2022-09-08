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
#include "oneflow/core/operator/reduce_sbp_util.h"
#include "oneflow/core/ndarray/binary_func.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

Maybe<void> VarOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& input_shape = ctx->InputShape("input", 0);
  const auto& reduce_axes = ctx->Attr<std::vector<int32_t>>("dim");
  CHECK_OR_RETURN(!reduce_axes.empty());
  const AxisVector reduce_axes_vec = {reduce_axes.begin(), reduce_axes.end()};
  const Shape& reduce_shape = CreateReducedShape(input_shape, reduce_axes_vec);
  const bool keepdim = ctx->Attr<bool>("keepdim");
  Shape output_shape;
  if (keepdim) {
    output_shape = reduce_shape;
  } else {
    output_shape = reduce_shape.RemoveOnes(reduce_axes_vec);
  }
  ctx->SetOutputShape("output", 0, output_shape);
  return Maybe<void>::Ok();
}

Maybe<void> VarOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

Maybe<void> VarOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("output", 0, ctx->InputDType("input", 0));
  return Maybe<void>::Ok();
}

Maybe<void> VarOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Broadcast(ctx->inputs()).Broadcast(ctx->outputs()).Build();
  const Shape& input_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("input", 0).shape();
  const int64_t ndim = input_shape.NumAxes();
  const std::vector<int32_t> axis = ctx->Attr<std::vector<int32_t>>("dim");
  const bool keepdim = ctx->Attr<bool>("keepdim");
  if (keepdim) {
    for (int i = 0; i < ndim; i++) {
      if (std::find(axis.begin(), axis.end(), i) == axis.end()) {
        ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
      }
    }
  } else {
    int offset = 0;
    for (int i = 0; i < ndim; i++) {
      if (std::find(axis.begin(), axis.end(), i) == axis.end()) {
        ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i - offset).Build();
      } else {
        offset += 1;
      }
    }
  }
  return Maybe<void>::Ok();
}

}  // namespace oneflow
