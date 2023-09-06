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

/* static */ auto FusedGluQuantOp::GetSbp(user_op::SbpContext* ctx) -> Maybe<void> {
  // check whether the user provide weight tensor v
  bool is_split_mode = false;
  if (ctx->user_op_conf().has_input("v", 0)) { is_split_mode = true; }
  if (is_split_mode) {
    CHECK_OR_RETURN(ctx->user_op_conf().has_input("v_scale", 0))
        << "expected v_scale for split mode";
    CHECK_OR_RETURN(ctx->user_op_conf().has_input("v_bias", 0)) << "expected v_bias for split mode";
  }

  std::vector<user_op::OpArg> scalar_args;
  if (ctx->user_op_conf().has_input("in_zero_point", 0)) {
    scalar_args.emplace_back("in_zero_point", 0);
  }
  if (ctx->user_op_conf().has_input("in_scale", 0)) { scalar_args.emplace_back("in_scale", 0); }

  std::vector<user_op::OpArg> vector_args;
  if (ctx->user_op_conf().has_input("weight_scale", 0)) {
    vector_args.emplace_back("weight_scale", 0);
  }
  if (ctx->user_op_conf().has_input("weight_acc", 0)) { vector_args.emplace_back("weight_acc", 0); }
  if (ctx->user_op_conf().has_input("scale", 0)) { vector_args.emplace_back("scale", 0); }
  if (ctx->user_op_conf().has_input("bias", 0)) { vector_args.emplace_back("bias", 0); }

  if (is_split_mode) {
    if (ctx->user_op_conf().has_input("v_weight_scale", 0)) {
      vector_args.emplace_back("v_weight_scale", 0);
    }
    if (ctx->user_op_conf().has_input("v_weight_acc", 0)) {
      vector_args.emplace_back("v_weight_acc", 0);
    }
    if (ctx->user_op_conf().has_input("v_scale", 0)) { vector_args.emplace_back("v_scale", 0); }
    if (ctx->user_op_conf().has_input("v_bias", 0)) { vector_args.emplace_back("v_bias", 0); }
  }

  // data parallelism
  for (int64_t i = 0; i < ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0).shape().NumAxes() - 1;
       ++i) {
    if (is_split_mode) {
      ctx->NewBuilder()
          .Split(user_op::OpArg("x", 0), i)
          .Broadcast(user_op::OpArg("w", 0))
          .Broadcast(user_op::OpArg("v", 0))
          .Broadcast(scalar_args)
          .Broadcast(vector_args)
          .Split(ctx->outputs(), i)
          .Build();
    } else {
      ctx->NewBuilder()
          .Split(user_op::OpArg("x", 0), i)
          .Broadcast(user_op::OpArg("w", 0))
          .Broadcast(scalar_args)
          .Broadcast(vector_args)
          .Split(ctx->outputs(), i)
          .Build();
    }
  }

  // model parallelism
  if (is_split_mode) {
    ctx->NewBuilder()
        .Broadcast(user_op::OpArg("x", 0))
        .Split(user_op::OpArg("w", 0), 0)
        .Split(user_op::OpArg("v", 0), 0)
        .Broadcast(scalar_args)
        .Split(vector_args, 0)
        .Split(ctx->outputs(),
               ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0).shape().NumAxes() - 1)
        .Build();
  }

  return Maybe<void>::Ok();
}

/* static */ auto FusedGluQuantOp::InferLogicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  // obtain input shape
  const Shape& x_shape = ctx->InputShape("x", 0);
  const Shape& w_shape = ctx->InputShape("w", 0);

  // check whether the user provide weight tensor v
  bool is_split_mode = false;
  if (ctx->has_input("v", 0)) { is_split_mode = true; }

  // check dimensions of x, w and b
  CHECK_GT_OR_RETURN(x_shape.NumAxes(), 1)
      << "number of axes of \'x\' should have be greater than 1, yet get " << x_shape.NumAxes();
  CHECK_EQ_OR_RETURN(w_shape.NumAxes(), 2)
      << "number of axes of \'w\' should have be equal to 2, yet get " << w_shape.NumAxes();

  // check input shapes of w and b
  size_t x_num_axes = x_shape.NumAxes();
  CHECK_EQ_OR_RETURN(w_shape.At(1), x_shape.At(x_num_axes - 1))
      << "dimension 1 of \'w\'(" << w_shape.At(1)
      << ") is not consistant with the last dimension of \'x\'(" << x_shape.At(x_num_axes - 1)
      << ")";

  if (ctx->has_input("scale", 0)) {
    CHECK_OR_RETURN(ctx->has_input("bias", 0));
    const user_op::TensorDesc& scale = ctx->InputTensorDesc("scale", 0);
    CHECK_EQ_OR_RETURN(scale.shape(), Shape({w_shape.At(0)}));
    const user_op::TensorDesc& bias = ctx->InputTensorDesc("bias", 0);
    CHECK_EQ_OR_RETURN(bias.shape(), Shape({w_shape.At(0)}));
  }
  if (ctx->has_input("in_scale", 0)) {
    CHECK_OR_RETURN(ctx->has_input("in_zero_point", 0));
    CHECK_OR_RETURN(ctx->has_input("weight_scale", 0));
    CHECK_OR_RETURN(ctx->has_input("weight_acc", 0));
    const user_op::TensorDesc& in_zero_point = ctx->InputTensorDesc("in_zero_point", 0);
    CHECK_EQ_OR_RETURN(in_zero_point.shape().Count(0), 1);
    const user_op::TensorDesc& in_scale = ctx->InputTensorDesc("in_scale", 0);
    CHECK_EQ_OR_RETURN(in_scale.shape().Count(0), 1);
    const user_op::TensorDesc& weight_scale = ctx->InputTensorDesc("weight_scale", 0);
    CHECK_EQ_OR_RETURN(weight_scale.shape(), Shape({w_shape.At(0)}));
    const user_op::TensorDesc& weight_acc = ctx->InputTensorDesc("weight_acc", 0);
    CHECK_EQ_OR_RETURN(weight_acc.shape(), Shape({w_shape.At(0)}));
    if (ctx->has_input("bias", 0)) {
      const user_op::TensorDesc& bias = ctx->InputTensorDesc("bias", 0);
      CHECK_EQ_OR_RETURN(bias.shape(), Shape({w_shape.At(0)}));
    }
  }

  if (!is_split_mode) {
    CHECK_EQ_OR_RETURN(w_shape.At(1) % 2, 0) << "dimension 1 of \'w\' is not divisible by 2";
  }

  // check both dimensions and input shapes of v and v_scale, v_bias (optional)
  if (is_split_mode) {
    const Shape& v_shape = ctx->InputShape("v", 0);

    CHECK_EQ_OR_RETURN(v_shape.NumAxes(), 2)
        << "number of axes of \'v\' should have be equal to 2, yet get " << v_shape.NumAxes();
    CHECK_OR_RETURN(v_shape == w_shape) << "the shape of \'v\' is not consistant with \'w\'";

    if (ctx->has_input("v_scale", 0)) {
      CHECK_OR_RETURN(ctx->has_input("v_bias", 0));
      const user_op::TensorDesc& v_scale = ctx->InputTensorDesc("v_scale", 0);
      CHECK_EQ_OR_RETURN(v_scale.shape(), Shape({v_shape.At(0)}));
      const user_op::TensorDesc& v_bias = ctx->InputTensorDesc("v_bias", 0);
      CHECK_EQ_OR_RETURN(v_bias.shape(), Shape({v_shape.At(0)}));
    }
    if (ctx->has_input("v_weight_scale", 0)) {
      CHECK_OR_RETURN(ctx->has_input("v_weight_acc", 0));
      const user_op::TensorDesc& v_weight_scale = ctx->InputTensorDesc("v_weight_scale", 0);
      CHECK_EQ_OR_RETURN(v_weight_scale.shape(), Shape({v_shape.At(0)}));
      const user_op::TensorDesc& v_weight_acc = ctx->InputTensorDesc("v_weight_acc", 0);
      CHECK_EQ_OR_RETURN(v_weight_acc.shape(), Shape({v_shape.At(0)}));
      if (ctx->has_input("v_bias", 0)) {
        const user_op::TensorDesc& v_bias = ctx->InputTensorDesc("v_bias", 0);
        CHECK_EQ_OR_RETURN(v_bias.shape(), Shape({v_shape.At(0)}));
      }
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

/* static */ auto FusedGluQuantOp::InferPhysicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  return InferLogicalTensorDesc(ctx);
}

/* static */ auto FusedGluQuantOp::InferDataType(user_op::InferContext* ctx) -> Maybe<void> {
  DataType out_dtype = ctx->Attr<DataType>("out_dtype");
  DataType x_dtype = ctx->InputDType("x", 0);

  bool is_split_mode = false;
  if (ctx->has_input("v", 0)) { is_split_mode = true; }

  CHECK_EQ_OR_RETURN(ctx->InputDType("w", 0), x_dtype)
      << "data type of \'w\' is not consitant with \'x\'";

  if (is_split_mode) {
    CHECK_EQ_OR_RETURN(ctx->InputDType("v", 0), x_dtype)
        << "data type of \'v\' is not consitant with \'x\'";
  }

  // set output data type
  ctx->SetOutputDType("y", 0, out_dtype);
  ctx->SetOutputDType("matmul_wx", 0, out_dtype);
  if (is_split_mode) { ctx->SetOutputDType("matmul_vx", 0, out_dtype); }

  return Maybe<void>::Ok();
}

}  // namespace oneflow
