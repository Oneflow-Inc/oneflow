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
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

namespace {

bool IsScalarTensor(const user_op::TensorDesc* tensor) {
  return tensor->shape().NumAxes() == 1 && tensor->shape().At(0) == 1;
}

bool IsZeroDimTensor(const user_op::TensorDesc* tensor) { return tensor->shape().NumAxes() == 0; }

Maybe<void> InferTensorDescBinaryBroadcastNormal(user_op::InferContext* ctx) {
  const user_op::TensorDesc& tensor_x = ctx->InputTensorDesc("x", 0);
  const user_op::TensorDesc& tensor_y = ctx->InputTensorDesc("y", 0);
  user_op::TensorDesc* tensor_z = ctx->MutOutputTensorDesc("z", 0);

  size_t output_num_axes = std::max(tensor_x.shape().NumAxes(), tensor_y.shape().NumAxes());
  if (IsZeroDimTensor(&tensor_x)) {
    ctx->SetOutputShape("z", 0, ctx->InputShape("y", 0));
    ctx->SetOutputIsDynamic("z", 0, ctx->InputIsDynamic("y", 0));
  } else if (IsZeroDimTensor(&tensor_y)) {
    ctx->SetOutputShape("z", 0, ctx->InputShape("x", 0));
    ctx->SetOutputIsDynamic("z", 0, ctx->InputIsDynamic("x", 0));
  } else if (IsScalarTensor(&tensor_x)) {
    ctx->SetOutputShape("z", 0, ctx->InputShape("y", 0));
    ctx->SetOutputIsDynamic("z", 0, ctx->InputIsDynamic("y", 0));
  } else if (IsScalarTensor(&tensor_y)) {
    ctx->SetOutputShape("z", 0, ctx->InputShape("x", 0));
    ctx->SetOutputIsDynamic("z", 0, ctx->InputIsDynamic("x", 0));
  } else {
    const auto& x_shape = CreateLeftExtendedShape(ShapeView(tensor_x.shape()), output_num_axes);
    const auto& y_shape = CreateLeftExtendedShape(ShapeView(tensor_y.shape()), output_num_axes);
    ctx->SetOutputShape("z", 0, ctx->InputShape("x", 0));
    ctx->SetOutputIsDynamic("z", 0, ctx->InputIsDynamic("x", 0));
    Shape out_shape(x_shape);
    FOR_RANGE(int64_t, i, 0, x_shape.NumAxes()) {
      if (x_shape.At(i) != 1 && y_shape.At(i) != 1 && x_shape.At(i) != y_shape.At(i)) {
        return Error::RuntimeError()
               << "The size of tensor a (" << x_shape.At(i) << ") must match the size of tensor b ("
               << y_shape.At(i) << ") at non-singleton dimension " << i;
      }
      out_shape.Set(i, (x_shape.At(i) == 0 || y_shape.At(i) == 0)
                           ? 0
                           : std::max(x_shape.At(i), y_shape.At(i)));
    }
    tensor_z->set_shape(out_shape);
  }
  tensor_z->set_is_dynamic(tensor_x.is_dynamic() || tensor_y.is_dynamic());
  return Maybe<void>::Ok();
}

Maybe<void> InferTensorDescBinaryBroadcastLogical(user_op::InferContext* ctx) {
  return InferTensorDescBinaryBroadcastNormal(ctx);
}

Maybe<void> InferDataTypeBinaryBroadcastNormal(user_op::InferContext* ctx) {
  const user_op::TensorDesc& tensor_x = ctx->InputTensorDesc("x", 0);
  const user_op::TensorDesc& tensor_y = ctx->InputTensorDesc("y", 0);
  CHECK_EQ_OR_RETURN(tensor_x.data_type(), tensor_y.data_type())
      << "InferDataType Failed. Expected " << DataType_Name(tensor_x.data_type()) << ", but got "
      << DataType_Name(tensor_y.data_type());
  ctx->SetOutputDType("z", 0, ctx->InputDType("x", 0));
  return Maybe<void>::Ok();
}

Maybe<void> InferDataTypeBinaryBroadcastLogical(user_op::InferContext* ctx) {
  const user_op::TensorDesc& tensor_x = ctx->InputTensorDesc("x", 0);
  const user_op::TensorDesc& tensor_y = ctx->InputTensorDesc("y", 0);
  CHECK_EQ_OR_RETURN(tensor_x.data_type(), tensor_y.data_type())
      << "InferDataType Failed. Expected " << DataType_Name(tensor_x.data_type()) << ", but got "
      << DataType_Name(tensor_y.data_type());
  ctx->SetOutputDType("z", 0, DataType::kBool);
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
void GenPartialSbpSign<BinaryFuncNanSum>(user_op::SbpContext* ctx) {
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

#define REGISTER_BINARY_BROADCAST_NORMAL_USER_OP(op_name, suffix)                        \
  /* static */ Maybe<void> op_name::InferLogicalTensorDesc(user_op::InferContext* ctx) { \
    return InferTensorDescBinaryBroadcastNormal(ctx);                                    \
  }                                                                                      \
  /*static*/ Maybe<void> op_name::InferPhysicalTensorDesc(user_op::InferContext* ctx) {  \
    return InferLogicalTensorDesc(ctx);                                                  \
  }                                                                                      \
  /* static */ Maybe<void> op_name::GetSbp(user_op::SbpContext* ctx) {                   \
    return GetBinaryBroadcastSbpSignature<BinaryFunc##suffix>(ctx);                      \
  }                                                                                      \
  /* static */ Maybe<void> op_name::InferDataType(user_op::InferContext* ctx) {          \
    return InferDataTypeBinaryBroadcastNormal(ctx);                                      \
  }

#define REGISTER_BINARY_BROADCAST_LOGICAL_USER_OP(op_name, suffix)                       \
  /* static */ Maybe<void> op_name::InferLogicalTensorDesc(user_op::InferContext* ctx) { \
    return InferTensorDescBinaryBroadcastLogical(ctx);                                   \
  }                                                                                      \
  /*static*/ Maybe<void> op_name::InferPhysicalTensorDesc(user_op::InferContext* ctx) {  \
    return InferLogicalTensorDesc(ctx);                                                  \
  }                                                                                      \
  /* static */ Maybe<void> op_name::GetSbp(user_op::SbpContext* ctx) {                   \
    return GetBinaryBroadcastSbpSignature<BinaryFunc##suffix>(ctx);                      \
  }                                                                                      \
  /* static */ Maybe<void> op_name::InferDataType(user_op::InferContext* ctx) {          \
    return InferDataTypeBinaryBroadcastLogical(ctx);                                     \
  }

OF_PP_FOR_EACH_TUPLE(REGISTER_BINARY_BROADCAST_NORMAL_USER_OP, MATH_BINARY_BROADCAST_FUNC_SEQ_ODS)
OF_PP_FOR_EACH_TUPLE(REGISTER_BINARY_BROADCAST_LOGICAL_USER_OP,
                     MATH_BINARY_BROADCAST_LOGICAL_FUNC_SEQ_ODS)

}  // namespace oneflow
