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

/* static */ Maybe<void> NormalTensorTensorOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder().Broadcast(ctx->inputs()).Broadcast(ctx->outputs()).Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> NormalTensorTensorOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& mean_shape = ctx->InputShape("mean", 0);
  const Shape& std_shape = ctx->InputShape("std", 0);
  size_t dimsA = mean_shape.NumAxes();
  size_t dimsB = std_shape.NumAxes();
  size_t ndim = dimsA > dimsB ? dimsA : dimsB;
  Shape expandedSizes(ndim);
  // Use ptrdiff_t to ensure signed comparison.
  for (ptrdiff_t i = (ptrdiff_t)ndim - 1; i >= 0; --i) {
    ptrdiff_t offset = ndim - 1 - i;
    ptrdiff_t dimA = dimsA - 1 - offset;
    ptrdiff_t dimB = dimsB - 1 - offset;
    auto sizeA = (dimA >= 0) ? mean_shape.At(dimA) : 1;
    auto sizeB = (dimB >= 0) ? std_shape.At(dimB) : 1;
    CHECK_OR_RETURN(
        sizeA == sizeB || sizeA == 1 || sizeB == 1)
        << "The size of tensor a (" << sizeA << ") must match the size of tensor b (" << sizeB
        << ") at non-singleton dimension " << i;
      // 1s map to the other size (even 0).
      expandedSizes.Set(i, sizeA == 1 ? sizeB : sizeA);
  }

  
  ctx->SetOutputShape("out", 0, expandedSizes);
  ctx->SetOutputIsDynamic("out", 0, ctx->InputIsDynamic("mean", 0));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> NormalTensorTensorOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}


/* static */ Maybe<void> NormalTensorTensorOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("mean", 0));
  return Maybe<void>::Ok();
}

}  // namespace oneflow


