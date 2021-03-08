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
#include "oneflow/core/ndarray/binary_func.h"
#include "oneflow/user/ops/math_binary_broadcast_seq.h"

namespace oneflow {

namespace {

bool IsScalarTensor(const user_op::TensorDesc* tensor) {
  return tensor->shape().NumAxes() == 1 && tensor->shape().At(0) == 1;
}

Maybe<void> InferTensorDescBinaryBroadcastNormal(user_op::InferContext* ctx) {
  const user_op::TensorDesc* tensor_x = ctx->TensorDesc4ArgNameAndIndex("x", 0);
  const user_op::TensorDesc* tensor_y = ctx->TensorDesc4ArgNameAndIndex("y", 0);
  user_op::TensorDesc* tensor_z = ctx->TensorDesc4ArgNameAndIndex("z", 0);
  CHECK_EQ_OR_RETURN(tensor_x->data_type(), tensor_y->data_type());
  size_t output_num_axes = std::max(tensor_x->shape().NumAxes(), tensor_y->shape().NumAxes());
  if (IsScalarTensor(tensor_x)) {
    *tensor_z = *tensor_y;
  } else if (IsScalarTensor(tensor_y)) {
    *tensor_z = *tensor_x;
  } else {
    const auto& x_shape = CreateLeftExtendedShape(ShapeView(tensor_x->shape()), output_num_axes);
    const auto& y_shape = CreateLeftExtendedShape(ShapeView(tensor_y->shape()), output_num_axes);
    *tensor_z = *tensor_x;
    Shape out_shape(x_shape);
    FOR_RANGE(int64_t, i, 0, x_shape.NumAxes()) {
      CHECK_OR_RETURN(x_shape.At(i) == 1 || y_shape.At(i) == 1 || x_shape.At(i) == y_shape.At(i))
          << "op: " << ctx->user_op_conf().op_name()
          << ", type: " << ctx->user_op_conf().op_type_name() << ", i: " << i
          << ", x_shape: " << x_shape << ", y_shape: " << y_shape;
      out_shape.Set(i, std::max(x_shape.At(i), y_shape.At(i)));
    }
    *tensor_z->mut_shape() = out_shape;
  }
  tensor_z->set_is_dynamic(tensor_x->is_dynamic() || tensor_y->is_dynamic());
  return Maybe<void>::Ok();
}

Maybe<void> InferTensorDescBinaryBroadcastLogical(user_op::InferContext* ctx) {
  JUST(InferTensorDescBinaryBroadcastNormal(ctx));
  *ctx->Dtype4ArgNameAndIndex("z", 0) = DataType::kInt8;
  return Maybe<void>::Ok();
}

template<template<typename> class binary_func>
void GenPartialSbpSign(user_op::SbpContext* ctx) {}

template<>
void GenPartialSbpSign<BinaryFuncAdd>(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("x", 0))
      .PartialSum(user_op::OpArg("y", 0))
      .PartialSum(user_op::OpArg("z", 0))
      .Build();
}

template<>
void GenPartialSbpSign<BinaryFuncSub>(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("x", 0))
      .PartialSum(user_op::OpArg("y", 0))
      .PartialSum(user_op::OpArg("z", 0))
      .Build();
}

template<>
void GenPartialSbpSign<BinaryFuncMul>(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .Broadcast(user_op::OpArg("x", 0))
      .PartialSum(user_op::OpArg("y", 0))
      .PartialSum(user_op::OpArg("z", 0))
      .Build();
  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("x", 0))
      .Broadcast(user_op::OpArg("y", 0))
      .PartialSum(user_op::OpArg("z", 0))
      .Build();
}

template<>
void GenPartialSbpSign<BinaryFuncDiv>(user_op::SbpContext* ctx) {
  ctx->NewBuilder()
      .PartialSum(user_op::OpArg("x", 0))
      .Broadcast(user_op::OpArg("y", 0))
      .PartialSum(user_op::OpArg("z", 0))
      .Build();
}

template<template<typename> class binary_func>
Maybe<void> GetBinaryBroadcastSbpSignature(user_op::SbpContext* ctx) {
  const Shape& x_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0).shape();
  const Shape& y_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("y", 0).shape();
  if (x_shape.NumAxes() < y_shape.NumAxes()) {
    FOR_RANGE(int64_t, i, 0, y_shape.NumAxes() - x_shape.NumAxes()) {
      ctx->NewBuilder()
          .Broadcast(user_op::OpArg("x", 0))
          .Split(user_op::OpArg("y", 0), i)
          .Split(user_op::OpArg("z", 0), i)
          .Build();
    }
    FOR_RANGE(int64_t, i, 0, x_shape.NumAxes()) {
      ctx->NewBuilder()
          .Split(user_op::OpArg("x", 0), x_shape.NumAxes() - 1 - i)
          .Split(user_op::OpArg("y", 0), y_shape.NumAxes() - 1 - i)
          .Split(ctx->outputs(), y_shape.NumAxes() - 1 - i)
          .Build();
    }
  } else if (x_shape.NumAxes() > y_shape.NumAxes()) {
    FOR_RANGE(int64_t, i, 0, x_shape.NumAxes() - y_shape.NumAxes()) {
      ctx->NewBuilder()
          .Split(user_op::OpArg("x", 0), i)
          .Broadcast(user_op::OpArg("y", 0))
          .Split(user_op::OpArg("z", 0), i)
          .Build();
    }
    FOR_RANGE(int64_t, i, 0, y_shape.NumAxes()) {
      ctx->NewBuilder()
          .Split(user_op::OpArg("x", 0), x_shape.NumAxes() - 1 - i)
          .Split(user_op::OpArg("y", 0), y_shape.NumAxes() - 1 - i)
          .Split(ctx->outputs(), x_shape.NumAxes() - 1 - i)
          .Build();
    }
  } else {
    FOR_RANGE(int64_t, i, 0, x_shape.NumAxes()) {
      if (x_shape.At(i) == 1 && y_shape.At(i) == 1) { continue; }
      if (x_shape.At(i) == y_shape.At(i)) {
        ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
      } else if (x_shape.At(i) == 1) {
        ctx->NewBuilder()
            .Broadcast(user_op::OpArg("x", 0))
            .Split(user_op::OpArg("y", 0), i)
            .Split(ctx->outputs(), i)
            .Build();
      } else if (y_shape.At(i) == 1) {
        ctx->NewBuilder()
            .Split(user_op::OpArg("x", 0), i)
            .Broadcast(user_op::OpArg("y", 0))
            .Split(ctx->outputs(), i)
            .Build();
      } else {
        UNIMPLEMENTED();
      }
    }
  }
  GenPartialSbpSign<binary_func>(ctx);
  return Maybe<void>::Ok();
}

}  // namespace

#define REGISTER_BINARY_BROADCAST_USER_OP(op_name, sbp_suffix, tensor_suffix) \
  REGISTER_USER_OP(op_name)                                                   \
      .Input("x")                                                             \
      .Input("y")                                                             \
      .Output("z")                                                            \
      .SetTensorDescInferFn(InferTensorDescBinaryBroadcast##tensor_suffix)    \
      .SetGetSbpFn(GetBinaryBroadcastSbpSignature<BinaryFunc##sbp_suffix>);

#define REGISTER_BINARY_BROADCAST_NORMAL_USER_OP(op_name, suffix) \
  REGISTER_BINARY_BROADCAST_USER_OP(op_name, suffix, Normal)

#define REGISTER_BINARY_BROADCAST_LOGICAL_USER_OP(op_name, suffix) \
  REGISTER_BINARY_BROADCAST_USER_OP(op_name, suffix, Logical)

OF_PP_FOR_EACH_TUPLE(REGISTER_BINARY_BROADCAST_NORMAL_USER_OP, MATH_BINARY_BROADCAST_FUNC_SEQ)
OF_PP_FOR_EACH_TUPLE(REGISTER_BINARY_BROADCAST_LOGICAL_USER_OP,
                     MATH_BINARY_BROADCAST_LOGICAL_FUNC_SEQ)

}  // namespace oneflow
