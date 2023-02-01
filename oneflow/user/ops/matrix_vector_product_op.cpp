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

Maybe<void> InferTensorDesc4MatrixVectorProduct(user_op::InferContext* ctx) {
  const user_op::TensorDesc& a = ctx->InputTensorDesc("a", 0);
  const user_op::TensorDesc& b = ctx->InputTensorDesc("b", 0);
  int64_t m = a.shape().At(0);
  int64_t k = a.shape().At(1);
  CHECK_EQ_OR_RETURN(k, b.shape().At(0)) << "Dim K should be equal to vector b's dim0. ";
  ctx->SetOutputShape("out", 0, Shape({m}));
  return Maybe<void>::Ok();
}

Maybe<void> InferDataType4MatrixVectorProduct(user_op::InferContext* ctx) {
  DataType dtype = ctx->InputDType("a", 0);
  CHECK_EQ_OR_RETURN(ctx->InputDType("b", 0), dtype)
      << "InferDataType Failed. Expected " << DataType_Name(dtype) << ", but got "
      << DataType_Name(ctx->InputDType("b", 0));
  ctx->SetOutputDType("out", 0, dtype);
  return Maybe<void>::Ok();
}

Maybe<void> InferTensorDesc4MatrixVectorProductGradA(user_op::InferContext* ctx) {
  /*
  A(m, k) matmul B(k) -> (m, k) matmul (k, 1) -> (m, 1) -> (m)
  GradA = dy (m) matmul B(k) -> (m, 1) (k, 1)_transpose
  */
  const user_op::TensorDesc& dy = ctx->InputTensorDesc("dy", 0);
  const user_op::TensorDesc& b = ctx->InputTensorDesc("b", 0);
  int64_t m = dy.shape().At(0);
  int64_t n = b.shape().At(0);
  ctx->SetOutputShape("dx", 0, Shape({m, n}));
  return Maybe<void>::Ok();
}

Maybe<void> InferTensorDesc4MatrixVectorProductGradB(user_op::InferContext* ctx) {
  /*
  A(m, k) matmul B(k) -> (m, k) matmul (k, 1) -> (m, 1) -> (m)
  GradB = dy_transpose (1, m) matmul A(m, k)
  */
  const user_op::TensorDesc& a = ctx->InputTensorDesc("a", 0);
  int64_t n = a.shape().At(1);
  ctx->SetOutputShape("dx", 0, Shape({n}));
  return Maybe<void>::Ok();
}

Maybe<void> InferDataType4Grad(user_op::InferContext* ctx) {
  DataType dtype = ctx->InputDType("dy", 0);
  ctx->SetOutputDType("dx", 0, dtype);
  return Maybe<void>::Ok();
}

}  // namespace

/* static */ Maybe<void> MatrixVectorProductOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  return InferTensorDesc4MatrixVectorProduct(ctx);
}

/*static*/ Maybe<void> MatrixVectorProductOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> MatrixVectorProductOp::GetSbp(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Split(user_op::OpArg("a", 0), 0)
      .Broadcast(user_op::OpArg("b", 0))
      .Split(user_op::OpArg("out", 0), 0)
      .Build();
  ctx->NewBuilder()
      .Split(user_op::OpArg("a", 0), 1)
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

/* static */ Maybe<void> MatrixVectorProductOp::InferDataType(user_op::InferContext* ctx) {
  return InferDataType4MatrixVectorProduct(ctx);
}

/* static */ Maybe<void> MatrixVectorProductGradAOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferTensorDesc4MatrixVectorProductGradA(ctx);
}

/*static*/ Maybe<void> MatrixVectorProductGradAOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> MatrixVectorProductGradAOp::GetSbp(user_op::SbpContext* ctx) {
  /*
  A(m, k) matmul B(k) -> (m, k) matmul (k, 1) -> (m, 1) -> (m)
  GradA = dy (m) matmul B(k) -> (m, 1) (k, 1)_transpose
  */
  ctx->NewBuilder()
      .Split(user_op::OpArg("dy", 0), 0)
      .Broadcast(user_op::OpArg("b", 0))
      .Split(user_op::OpArg("dx", 0), 0)
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

/* static */ Maybe<void> MatrixVectorProductGradAOp::InferDataType(user_op::InferContext* ctx) {
  return InferDataType4Grad(ctx);
}

/* static */ Maybe<void> MatrixVectorProductGradBOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferTensorDesc4MatrixVectorProductGradB(ctx);
}

/*static*/ Maybe<void> MatrixVectorProductGradBOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> MatrixVectorProductGradBOp::GetSbp(user_op::SbpContext* ctx) {
  /*
  A(m, k) matmul B(k) -> (m, k) matmul (k, 1) -> (m, 1) -> (m)
  dy = (m, )
  GradB = dy_transpose (1, m) matmul A(m, k)
  */
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("dy", 0))
      .Split(user_op::OpArg("a", 0), 1)
      .Split(user_op::OpArg("dx", 0), 0)
      .Build();
  ctx->NewBuilder()
      .Split(user_op::OpArg("dy", 0), 0)
      .Split(user_op::OpArg("a", 0), 0)
      .PartialSum(user_op::OpArg("dx", 0))
      .Build();
  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("dy", 0))
      .Broadcast(user_op::OpArg("a", 0))
      .PartialSum(user_op::OpArg("dx", 0))
      .Build();
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("dy", 0))
      .PartialSum(user_op::OpArg("a", 0))
      .PartialSum(user_op::OpArg("dx", 0))
      .Build();
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> MatrixVectorProductGradBOp::InferDataType(user_op::InferContext* ctx) {
  return InferDataType4Grad(ctx);
}

}  // namespace oneflow
