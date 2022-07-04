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

namespace {

Maybe<void> InferTensorDesc4Matmul(user_op::InferContext* ctx) {
  const user_op::TensorDesc& a = ctx->InputTensorDesc("a", 0);
  const user_op::TensorDesc& b = ctx->InputTensorDesc("b", 0);
  int64_t k = a.shape().At(0);
  CHECK_EQ_OR_RETURN(k, b.shape().At(0)) << "Dim K should be equal to vector b's dim0. ";
  int64_t n = b.shape().At(1);
  *ctx->OutputShape("out", 0) = Shape({n});
  return Maybe<void>::Ok();
}

Maybe<void> InferDataType4Matmul(user_op::InferContext* ctx) {
  const DataType& dtype = ctx->InputDType("a", 0);
  CHECK_EQ_OR_RETURN(ctx->InputDType("b", 0), dtype)
      << "Matrix A datatype should be equal to Vector B. ";
  *ctx->OutputDType("out", 0) = dtype;
  return Maybe<void>::Ok();
}

}  // namespace

/* static */ Maybe<void> VectorMatrixProductOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return InferTensorDesc4Matmul(ctx);
}

/*static*/ Maybe<void> VectorMatrixProductOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> VectorMatrixProductOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("a", 0))
      .Split(user_op::OpArg("b", 0), 1)
      .Split(user_op::OpArg("out", 0), 0)
      .Build();
  ctx->NewBuilder()
      .Split(user_op::OpArg("a", 0), 0)
      .Split(user_op::OpArg("b", 0), 0)
      .PartialSum(user_op::OpArg("out", 0))
      .Build();
  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("a", 0))
      .Broadcast(user_op::OpArg("b", 0))
      .PartialSum(user_op::OpArg("out", 0))
      .Build();
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("a", 0))
      .PartialSum(user_op::OpArg("b", 0))
      .PartialSum(user_op::OpArg("out", 0))
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> VectorMatrixProductOp::InferDataType(user_op::InferContext* ctx) {
  return InferDataType4Matmul(ctx);
}

}  // namespace oneflow
