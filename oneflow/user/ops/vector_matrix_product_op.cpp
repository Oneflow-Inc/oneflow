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

Maybe<void> InferTensorDesc4VectorMatrixProduct(user_op::InferContext* ctx) {
  const user_op::TensorDesc& a = ctx->InputTensorDesc("a", 0);
  const user_op::TensorDesc& b = ctx->InputTensorDesc("b", 0);
  int64_t k = a.shape().At(0);
  CHECK_EQ_OR_RETURN(k, b.shape().At(0)) << "Dim K should be equal to vector b's dim0. ";
  int64_t n = b.shape().At(1);
  ctx->SetOutputShape("out", 0, Shape({n}));
  return Maybe<void>::Ok();
}

Maybe<void> InferDataType4VectorMatrixProduct(user_op::InferContext* ctx) {
  DataType dtype = ctx->InputDType("a", 0);
  CHECK_EQ_OR_RETURN(ctx->InputDType("b", 0), dtype)
      << "Matrix A datatype should be equal to Vector B. ";
  ctx->SetOutputDType("out", 0, dtype);
  return Maybe<void>::Ok();
}

Maybe<void> InferTensorDesc4VectorMatrixProductGradA(user_op::InferContext* ctx) {
  /*
  A(k, ) matmul B(k, n) -> (1, k) matmul (k, n) -> (1, n) -> (n)
  GradA = dy (n) matmul B_transpose(n, k) -> (1, n) matmul (n, k)
  */
  const user_op::TensorDesc& b = ctx->InputTensorDesc("b", 0);
  int64_t k = b.shape().At(0);
  ctx->SetOutputShape("dx", 0, Shape({k}));
  return Maybe<void>::Ok();
}

Maybe<void> InferTensorDesc4VectorMatrixProductGradB(user_op::InferContext* ctx) {
  /*
  A(k, ) matmul B(k, n) -> (1, k) matmul (k, n) -> (1, n) -> (n)
  GradB = a (k, 1) matmul dy (1, n)
  */
  const user_op::TensorDesc& dy = ctx->InputTensorDesc("dy", 0);
  const user_op::TensorDesc& a = ctx->InputTensorDesc("a", 0);
  int64_t k = a.shape().At(0);
  int64_t n = dy.shape().At(0);
  ctx->SetOutputShape("dx", 0, Shape({k, n}));
  return Maybe<void>::Ok();
}

Maybe<void> InferDataType4Grad(user_op::InferContext* ctx) {
  DataType dtype = ctx->InputDType("dy", 0);
  ctx->SetOutputDType("dx", 0, dtype);
  return Maybe<void>::Ok();
}

}  // namespace

/* static */ Maybe<void> VectorMatrixProductOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return InferTensorDesc4VectorMatrixProduct(ctx);
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
  return InferDataType4VectorMatrixProduct(ctx);
}

/* static */ Maybe<void> VectorMatrixProductGradAOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferTensorDesc4VectorMatrixProductGradA(ctx);
}

/*static*/ Maybe<void> VectorMatrixProductGradAOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> VectorMatrixProductGradAOp::GetSbp(user_op::SbpContext* ctx) {
  /*
  A(k, ) matmul B(k, n) -> (1, k) matmul (k, n) -> (1, n) -> (n)
  GradA = dy (n) matmul B_transpose(n, k) -> (1, n) matmul (n, k)
  */
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("dy", 0))
      .Split(user_op::OpArg("b", 0), 0)
      .Split(user_op::OpArg("dx", 0), 0)
      .Build();
  ctx->NewBuilder()
      .Split(user_op::OpArg("dy", 0), 0)
      .Split(user_op::OpArg("b", 0), 1)
      .PartialSum(user_op::OpArg("dx", 0))
      .Build();
  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("dy", 0))
      .Broadcast(user_op::OpArg("b", 0))
      .PartialSum(user_op::OpArg("dx", 0))
      .Build();
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("dy", 0))
      .PartialSum(user_op::OpArg("b", 0))
      .PartialSum(user_op::OpArg("dx", 0))
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> VectorMatrixProductGradAOp::InferDataType(user_op::InferContext* ctx) {
  return InferDataType4Grad(ctx);
}

/* static */ Maybe<void> VectorMatrixProductGradBOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferTensorDesc4VectorMatrixProductGradB(ctx);
}

/*static*/ Maybe<void> VectorMatrixProductGradBOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> VectorMatrixProductGradBOp::GetSbp(user_op::SbpContext* ctx) {
  /*
  A(k, ) matmul B(k, n) -> (1, k) matmul (k, n) -> (1, n) -> (n)
  A(k, ) -> (1, k)
  GradB = a_transpose (k, 1) matmul dy (1, n)
  */
  ctx->NewBuilder()
      .Split(user_op::OpArg("a", 0), 0)
      .Broadcast(user_op::OpArg("dy", 0))
      .Split(user_op::OpArg("dx", 0), 0)
      .Build();
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("a", 0))
      .Split(user_op::OpArg("dy", 0), 0)
      .Split(user_op::OpArg("dx", 0), 1)
      .Build();
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("a", 0))
      .PartialSum(user_op::OpArg("dy", 0))
      .PartialSum(user_op::OpArg("dx", 0))
      .Build();
  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("a", 0))
      .Broadcast(user_op::OpArg("dy", 0))
      .PartialSum(user_op::OpArg("dx", 0))
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> VectorMatrixProductGradBOp::InferDataType(user_op::InferContext* ctx) {
  return InferDataType4Grad(ctx);
}

}  // namespace oneflow
