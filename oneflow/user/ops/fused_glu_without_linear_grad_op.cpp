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

/* static */ auto FusedGluWithoutLinearGradOp::GetSbp(user_op::SbpContext* ctx) -> Maybe<void> {
  // check existance of optional args
  bool is_split_mode = false;
  if (ctx->user_op_conf().has_input("matmul_vx", 0)) { is_split_mode = true; }

  for (int64_t i = 0;
       i < ctx->LogicalTensorDesc4InputArgNameAndIndex("dy", 0).shape().NumAxes() - 1; ++i) {
    if (is_split_mode) {
      ctx->NewBuilder()
          .Split(user_op::OpArg("dy", 0), i)
          .Split(user_op::OpArg("matmul_wx", 0), i)
          .Split(user_op::OpArg("matmul_vx", 0), i)
          .Split(ctx->outputs(), i)
          .Build();
    } else {
      ctx->NewBuilder()
          .Split(user_op::OpArg("dy", 0), i)
          .Split(user_op::OpArg("matmul_wx", 0), i)
          .Split(ctx->outputs(), i)
          .Build();
    }
  }

  return Maybe<void>::Ok();
}

/* static */ auto FusedGluWithoutLinearGradOp::InferLogicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  // obtain input shape
  const Shape& dy_shape = ctx->InputShape("dy", 0);
  const Shape& matmul_wx_shape = ctx->InputShape("matmul_wx", 0);

  // check existance of optional args
  bool is_split_mode = false;
  if (ctx->has_input("matmul_vx", 0)) { is_split_mode = true; }

  // obtain dimensions of dy and matmul_wx
  size_t dy_num_axes = dy_shape.NumAxes();
  size_t matmul_wx_num_axes = matmul_wx_shape.NumAxes();

  // check dimensions of dy and matmul_wx
  CHECK_GT_OR_RETURN(dy_num_axes, 1)
      << "number of axes of \'dy\' should have be greater than 1, yet get " << dy_num_axes;
  CHECK_GT_OR_RETURN(matmul_wx_num_axes, 1)
      << "number of axes of \'matmul_wx\' should have be greater than 1, yet get "
      << matmul_wx_num_axes;
  CHECK_EQ_OR_RETURN(dy_num_axes, matmul_wx_num_axes)
      << "number of axes of \'dy\'(" << dy_num_axes
      << ") is not consistant with the one of \'matmul_wx\'(" << matmul_wx_num_axes << ")";

  // check input shapes of dy and matmul_wx
  for (uint64_t i = 0; i < dy_num_axes - 1; i++) {
    size_t dy_size = dy_shape.At(i);
    size_t matmul_wx_size = matmul_wx_shape.At(i);
    CHECK_EQ_OR_RETURN(dy_size, matmul_wx_size)
        << "dimension " << i << "of \'dy\'(" << dy_size << ") and \'matmul_wx\'(" << matmul_wx_size
        << ") is not consistent";
  }
  if (is_split_mode) {
    CHECK_EQ_OR_RETURN(dy_shape.At(dy_num_axes - 1), matmul_wx_shape.At(matmul_wx_num_axes - 1))
        << "the last dimension of \'dy\'(" << dy_shape.At(dy_num_axes - 1)
        << ") is not consistant with the last dimension of \'matmul_wx\'("
        << matmul_wx_shape.At(matmul_wx_num_axes - 1) << ")";
  } else {
    CHECK_EQ_OR_RETURN(2 * dy_shape.At(dy_num_axes - 1), matmul_wx_shape.At(matmul_wx_num_axes - 1))
        << "two times of the last dimension of \'dy\'(" << 2 * dy_shape.At(dy_num_axes - 1)
        << ") is not consistant with the last dimension of \'matmul_wx\'("
        << matmul_wx_shape.At(matmul_wx_num_axes - 1) << ")";
  }

  // check both dimensions and input shapes of matmul_vx (optional)
  if (is_split_mode) {
    // obtain input shape
    const Shape& matmul_vx_shape = ctx->InputShape("matmul_vx", 0);

    // check dimensions of matmul_vx
    size_t matmul_vx_num_axes = matmul_vx_shape.NumAxes();
    CHECK_GT_OR_RETURN(matmul_vx_num_axes, 1)
        << "number of axes of \'matmul_vx\' should have be greater than 1, yet get "
        << matmul_vx_num_axes;
    CHECK_EQ_OR_RETURN(matmul_vx_num_axes, dy_num_axes)
        << "number of axes of \'dy\'(" << dy_num_axes
        << ") is not consistant with the one of \'matmul_vx\'(" << matmul_vx_num_axes << ")";

    // check input shapes of dy and matmul_vx
    for (uint64_t i = 0; i < dy_num_axes - 1; i++) {
      size_t dy_size = dy_shape.At(i);
      size_t matmul_vx_size = matmul_vx_shape.At(i);
      CHECK_EQ_OR_RETURN(dy_size, matmul_vx_size)
          << "dimension " << i << "of \'dy\'(" << dy_size << ") and \'matmul_vx\'("
          << matmul_vx_size << ") is not consistent";
    }
    CHECK_EQ_OR_RETURN(matmul_vx_shape.At(matmul_vx_num_axes - 1), dy_shape.At(dy_num_axes - 1))
        << "the last dimension of \'dy\'(" << dy_shape.At(dy_num_axes - 1)
        << ") is not consistant with the last dimension of \'matmul_vx\'("
        << matmul_vx_shape.At(matmul_vx_num_axes - 1) << ")";
  }

  // set shape of the output tensor d_matmul_wx
  Shape d_matmul_wx_shape = matmul_wx_shape;  // borrow from input shape
  user_op::TensorDesc* d_matmul_wx_tensor = ctx->MutOutputTensorDesc("d_matmul_wx", 0);
  d_matmul_wx_tensor->set_shape(d_matmul_wx_shape);

  // set shape of the output tensor d_matmul_vx (optional)
  if (is_split_mode) {
    const Shape& matmul_vx_shape = ctx->InputShape("matmul_vx", 0);
    Shape d_matmul_vx_shape = matmul_vx_shape;  // borrow from input shape
    user_op::TensorDesc* d_matmul_vx_tensor = ctx->MutOutputTensorDesc("d_matmul_vx", 0);
    d_matmul_vx_tensor->set_shape(d_matmul_vx_shape);
  }

  return Maybe<void>::Ok();
}

/* static */ auto FusedGluWithoutLinearGradOp::InferPhysicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  return InferLogicalTensorDesc(ctx);
}

/* static */ auto FusedGluWithoutLinearGradOp::InferDataType(user_op::InferContext* ctx)
    -> Maybe<void> {
  // obtain input data types
  DataType dy_dtype = ctx->InputDType("dy", 0);

  // check types of matmul_wx
  CHECK_EQ_OR_RETURN(ctx->InputDType("matmul_wx", 0), dy_dtype)
      << "data type of \'matmul_wx\' is not consitant with \'dy\'";

  bool is_split_mode = ctx->has_input("matmul_vx", 0);

  // check types of matmul_vx (optional)
  if (is_split_mode) {
    CHECK_EQ_OR_RETURN(ctx->InputDType("matmul_vx", 0), dy_dtype)
        << "data type of \'matmul_vx\' is not consitant with \'dy\'";
  }

  // set output data type
  ctx->SetOutputDType("d_matmul_wx", 0, dy_dtype);
  if (is_split_mode) { ctx->SetOutputDType("d_matmul_vx", 0, dy_dtype); }

  return Maybe<void>::Ok();
}

}  // namespace oneflow
