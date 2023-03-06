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

/* static */ auto FusedGluOp::GetSbp(user_op::SbpContext* ctx) -> Maybe<void> {
  // check whether the user provide weight tensor v
  bool is_split_mode = false;
  if (ctx->user_op_conf().has_input("v", 0)) { is_split_mode = true; }

  bool has_b = ctx->user_op_conf().has_input("b", 0);
  bool has_c = ctx->user_op_conf().has_input("c", 0);

  // check whether the user provide bais tensors
  CHECK_OR_RETURN(!(has_b && (is_split_mode && !has_c)))
      << "expected existance of c, when provide tensors w, v and b";
  bool has_bias = false;
  if (has_b && (is_split_mode && has_c)) {
    has_bias = true;
  } else if (has_b && (!is_split_mode)) {
    has_bias = true;
  } else {
    has_bias = false;
  }

  // data parallelism
  for (int64_t i = 0; i < ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0).shape().NumAxes() - 1;
       ++i) {
    if (is_split_mode && has_bias) {
      ctx->NewBuilder()
          .Split(user_op::OpArg("x", 0), i)
          .Broadcast(user_op::OpArg("w", 0))
          .Broadcast(user_op::OpArg("b", 0))
          .Broadcast(user_op::OpArg("v", 0))
          .Broadcast(user_op::OpArg("c", 0))
          .Split(ctx->outputs(), i)
          .Build();
    } else if (is_split_mode && !has_bias) {
      ctx->NewBuilder()
          .Split(user_op::OpArg("x", 0), i)
          .Broadcast(user_op::OpArg("w", 0))
          .Broadcast(user_op::OpArg("v", 0))
          .Split(ctx->outputs(), i)
          .Build();
    } else if (!is_split_mode && has_bias) {
      ctx->NewBuilder()
          .Split(user_op::OpArg("x", 0), i)
          .Broadcast(user_op::OpArg("w", 0))
          .Broadcast(user_op::OpArg("b", 0))
          .Split(ctx->outputs(), i)
          .Build();
    } else if (!is_split_mode && !has_bias) {
      ctx->NewBuilder()
          .Split(user_op::OpArg("x", 0), i)
          .Broadcast(user_op::OpArg("w", 0))
          .Split(ctx->outputs(), i)
          .Build();
    }
  }

  // model parallelism
  if (is_split_mode && has_bias) {
    ctx->NewBuilder()
        .Broadcast(user_op::OpArg("x", 0))
        .Split(user_op::OpArg("w", 0), 0)
        .Split(user_op::OpArg("b", 0), 0)
        .Split(user_op::OpArg("v", 0), 0)
        .Split(user_op::OpArg("c", 0), 0)
        .Split(ctx->outputs(),
               ctx->LogicalTensorDesc4InputArgNameAndIndex("y", 0).shape().NumAxes() - 1)
        .Build();
  } else if (is_split_mode && !has_bias) {
    ctx->NewBuilder()
        .Broadcast(user_op::OpArg("x", 0))
        .Split(user_op::OpArg("w", 0), 0)
        .Split(user_op::OpArg("v", 0), 0)
        .Split(ctx->outputs(),
               ctx->LogicalTensorDesc4InputArgNameAndIndex("y", 0).shape().NumAxes() - 1)
        .Build();
  }

  return Maybe<void>::Ok();
}

/* static */ auto FusedGluOp::InferLogicalTensorDesc(user_op::InferContext* ctx) -> Maybe<void> {
  // obtain input shape
  const Shape& x_shape = ctx->InputShape("x", 0);
  const Shape& w_shape = ctx->InputShape("w", 0);

  // check whether the user provide weight tensor v
  bool is_split_mode = false;
  if (ctx->has_input("v", 0)) { is_split_mode = true; }

  bool has_b = ctx->has_input("b", 0);
  bool has_c = ctx->has_input("c", 0);

  // check whether the user provide bais tensors
  CHECK_OR_RETURN(!(has_b && (is_split_mode && !has_c)))
      << "expected existance of c, when provide tensors w, v and b";
  bool has_bias = false;
  if (has_b && (is_split_mode && has_c)) {
    has_bias = true;
  } else if (has_b && (!is_split_mode)) {
    has_bias = true;
  } else {
    has_bias = false;
  }

  // check dimensions of x, w and b
  CHECK_GT_OR_RETURN(x_shape.NumAxes(), 1)
      << "number of axes of \'x\' should have be greater than 1, yet get " << x_shape.NumAxes();
  CHECK_EQ_OR_RETURN(w_shape.NumAxes(), 2)
      << "number of axes of \'w\' should have be equal to 2, yet get " << w_shape.NumAxes();
  if (has_bias) {
    const Shape& b_shape = ctx->InputShape("b", 0);
    CHECK_EQ_OR_RETURN(b_shape.NumAxes(), 1)
        << "number of axes of \'b\' should have be equal to 1, yet get " << b_shape.NumAxes();
  }

  // check input shapes of w and b
  size_t x_num_axes = x_shape.NumAxes();
  CHECK_EQ_OR_RETURN(w_shape.At(1), x_shape.At(x_num_axes - 1))
      << "dimension 1 of \'w\'(" << w_shape.At(1)
      << ") is not consistant with the last dimension of \'x\'(" << x_shape.At(x_num_axes - 1)
      << ")";
  if (has_bias) {
    const Shape& b_shape = ctx->InputShape("b", 0);
    CHECK_EQ_OR_RETURN(b_shape.At(0), w_shape.At(0))
        << "dimension 0 of \'b\'(" << b_shape.At(0)
        << ") is not consistant with dimension 0 of \'w\'(" << w_shape.At(0) << ")";
  }
  if (!is_split_mode) {
    CHECK_EQ_OR_RETURN(w_shape.At(1) % 2, 0) << "dimension 1 of \'w\' is not divisible by 2";
  }

  // check both dimensions and input shapes of v and c (optional)
  if (is_split_mode) {
    const Shape& v_shape = ctx->InputShape("v", 0);

    CHECK_EQ_OR_RETURN(v_shape.NumAxes(), 2)
        << "number of axes of \'v\' should have be equal to 2, yet get " << v_shape.NumAxes();
    CHECK_OR_RETURN(v_shape == w_shape) << "the shape of \'v\' is not consistant with \'w\'";

    if (has_bias) {
      const Shape& b_shape = ctx->InputShape("b", 0);
      const Shape& c_shape = ctx->InputShape("c", 0);
      CHECK_EQ_OR_RETURN(c_shape.NumAxes(), 1)
          << "number of axes of \'c\' should have be equal to 1, yet get " << c_shape.NumAxes();
      CHECK_OR_RETURN(c_shape == b_shape) << "the shape of \'c\' is not consistant with \'b\'";
    }
  }

  // set shape of the output tensor y
  Shape y_shape = x_shape;  // borrow from input shape
  size_t y_num_axes = x_num_axes;
  if (is_split_mode) {
    y_shape.Set(y_num_axes - 1, w_shape.At(0));
  } else {
    y_shape.Set(y_num_axes - 1, w_shape.At(0) / 2);
  }
  user_op::TensorDesc* y_tensor = ctx->MutOutputTensorDesc("y", 0);
  y_tensor->set_shape(y_shape);

  // set shape of the output tensors of both matmul_wx and matmul_vx
  Shape matmul_wx_shape = x_shape;  // borrow from input shape
  matmul_wx_shape.Set(x_num_axes - 1, w_shape.At(0));
  user_op::TensorDesc* matmul_wx_tensor = ctx->MutOutputTensorDesc("matmul_wx", 0);
  matmul_wx_tensor->set_shape(matmul_wx_shape);
  if (is_split_mode) {
    user_op::TensorDesc* matmul_vx_tensor = ctx->MutOutputTensorDesc("matmul_vx", 0);
    matmul_vx_tensor->set_shape(y_shape);
  }

  return Maybe<void>::Ok();
}

/* static */ auto FusedGluOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) -> Maybe<void> {
  return InferLogicalTensorDesc(ctx);
}

/* static */ auto FusedGluOp::InferDataType(user_op::InferContext* ctx) -> Maybe<void> {
  // obtain input data types
  DataType x_dtype = ctx->InputDType("x", 0);

  // check whether the user provide weight tensor v
  bool is_split_mode = false;
  if (ctx->has_input("v", 0)) { is_split_mode = true; }

  bool has_b = ctx->has_input("b", 0);
  bool has_c = ctx->has_input("c", 0);

  // check whether the user provide bais tensors
  CHECK_OR_RETURN(!(has_b && (is_split_mode && !has_c)))
      << "expected existance of c, when provide tensors w, v and b";
  bool has_bias = false;
  if (has_b && (is_split_mode && has_c)) {
    has_bias = true;
  } else if (has_b && (!is_split_mode)) {
    has_bias = true;
  } else {
    has_bias = false;
  }

  // check types of x, w and b
  CHECK_EQ_OR_RETURN(ctx->InputDType("w", 0), x_dtype)
      << "data type of \'w\' is not consitant with \'x\'";
  if (has_bias) {
    CHECK_EQ_OR_RETURN(ctx->InputDType("b", 0), x_dtype)
        << "data type of \'b\' is not consitant with \'x\'";
  }

  // check types of v and c (optional)
  if (is_split_mode) {
    CHECK_EQ_OR_RETURN(ctx->InputDType("v", 0), x_dtype)
        << "data type of \'v\' is not consitant with \'x\'";
    if (has_bias) {
      CHECK_EQ_OR_RETURN(ctx->InputDType("c", 0), x_dtype)
          << "data type of \'c\' is not consitant with \'x\'";
    }
  }

  // set output data type
  ctx->SetOutputDType("y", 0, x_dtype);
  ctx->SetOutputDType("matmul_wx", 0, x_dtype);
  if (is_split_mode) { ctx->SetOutputDType("matmul_vx", 0, x_dtype); }

  return Maybe<void>::Ok();
}

}  // namespace oneflow
