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
using namespace user_op;

Maybe<void> GetSbpSignature_(SbpContext* ctx) {
  const Shape& x_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0).shape();
  const Shape& y_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("y", 0).shape();

  FOR_RANGE(int64_t, i, 0, x_shape.NumAxes()) {
    if (x_shape.At(i) == 1 && y_shape.At(i) == 1) { continue; }
    if (x_shape.At(i) == y_shape.At(i)) {
      ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
    } else {
      UNIMPLEMENTED();
    }
  }
  return Maybe<void>::Ok();
}

Maybe<void> InferTensorDesc_(InferContext* ctx) {
  const TensorDesc& tensor_x = ctx->InputTensorDesc("x", 0);
  const TensorDesc& tensor_y = ctx->InputTensorDesc("y", 0);

  CHECK_EQ_OR_RETURN(tensor_x.shape().NumAxes(), tensor_y.shape().NumAxes())
      << "Shape of tensor x and y should be same";

  FOR_RANGE(int64_t, i, 0, tensor_x.shape().NumAxes()) {
    CHECK_EQ_OR_RETURN(tensor_x.shape().At(i), tensor_y.shape().At(i));
  }

  TensorDesc* tensor_dx = ctx->MutOutputTensorDesc("dx", 0);
  TensorDesc* tensor_dy = ctx->MutOutputTensorDesc("dy", 0);

  if (tensor_dx) { tensor_dx->set_shape(tensor_x.shape()); }

  if (tensor_dy) { tensor_dy->set_shape(tensor_y.shape()); }

  return Maybe<void>::Ok();
}

Maybe<void> InferDataType_(InferContext* ctx) {
  const TensorDesc& tensor_dz = ctx->InputTensorDesc("dz", 0);
  TensorDesc* tensor_dx = ctx->MutOutputTensorDesc("dx", 0);
  TensorDesc* tensor_dy = ctx->MutOutputTensorDesc("dy", 0);

  if (tensor_dx) { tensor_dx->set_data_type(tensor_dz.data_type()); }

  if (tensor_dy) { tensor_dy->set_data_type(tensor_dz.data_type()); }

  return Maybe<void>::Ok();
}

}  // namespace

#define DEF_ELEMENTWISE_XIMUM_FW_OP(op_class_name_prefix)                                        \
  /* static */ Maybe<void> op_class_name_prefix##Op::InferLogicalTensorDesc(                     \
      user_op::InferContext* ctx) {                                                              \
    return user_op::TensorDescInferFnUtil::Unchanged(ctx);                                       \
  }                                                                                              \
                                                                                                 \
  /*static*/ Maybe<void> op_class_name_prefix##Op::InferPhysicalTensorDesc(                      \
      user_op::InferContext* ctx) {                                                              \
    return InferLogicalTensorDesc(ctx);                                                          \
  }                                                                                              \
                                                                                                 \
  /* static */ Maybe<void> op_class_name_prefix##Op::GetSbp(user_op::SbpContext* ctx) {          \
    return user_op::GetSbpFnUtil::SplitForEachAxis(ctx);                                         \
  }                                                                                              \
                                                                                                 \
  /* static */ Maybe<void> op_class_name_prefix##Op::InferDataType(user_op::InferContext* ctx) { \
    return user_op::TensorDescInferFnUtil::UnchangedDataType(ctx);                               \
  }

#define DEF_ELEMENTWISE_XIMUM_BW_OP(op_class_name_prefix)                                       \
  /* static */ Maybe<void> op_class_name_prefix##BackwardOp::InferLogicalTensorDesc(            \
      user_op::InferContext* ctx) {                                                             \
    return InferTensorDesc_(ctx);                                                               \
  }                                                                                             \
                                                                                                \
  /*static*/ Maybe<void> op_class_name_prefix##BackwardOp::InferPhysicalTensorDesc(             \
      user_op::InferContext* ctx) {                                                             \
    return InferLogicalTensorDesc(ctx);                                                         \
  }                                                                                             \
                                                                                                \
  /* static */ Maybe<void> op_class_name_prefix##BackwardOp::GetSbp(user_op::SbpContext* ctx) { \
    return GetSbpSignature_(ctx);                                                               \
  }                                                                                             \
                                                                                                \
  /* static */ Maybe<void> op_class_name_prefix##BackwardOp::InferDataType(                     \
      user_op::InferContext* ctx) {                                                             \
    return InferDataType_(ctx);                                                                 \
  }

#define REGISTER_ELEMENTWISE_XIMUM_OP(op_type_name, op_class_name_prefix) \
  DEF_ELEMENTWISE_XIMUM_FW_OP(op_class_name_prefix);                      \
  DEF_ELEMENTWISE_XIMUM_BW_OP(op_class_name_prefix);

REGISTER_ELEMENTWISE_XIMUM_OP("elementwise_maximum", ElementwiseMaximum);
REGISTER_ELEMENTWISE_XIMUM_OP("elementwise_minimum", ElementwiseMinimum);

}  // namespace oneflow
