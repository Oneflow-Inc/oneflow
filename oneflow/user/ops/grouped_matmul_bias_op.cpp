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
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/just.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/infer_util.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

/* static */ Maybe<void> GroupedMatmulBiasOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const int64_t input_size = ctx->input_size("xs");
  CHECK_EQ_OR_RETURN(ctx->input_size("weights"), input_size);
  const bool has_biases = ctx->has_input("biases", 0);
  if (has_biases) { CHECK_EQ_OR_RETURN(ctx->input_size("biases"), input_size); }
  CHECK_EQ_OR_RETURN(ctx->output_size("ys"), input_size);

  const DataType data_type = ctx->InputTensorDesc("xs", 0).data_type();
  for (int64_t i = 0; i < input_size; ++i) {
    const user_op::TensorDesc& x_desc = ctx->InputTensorDesc("xs", i);
    CHECK_EQ_OR_RETURN(x_desc.data_type(), data_type);
    CHECK_GE_OR_RETURN(x_desc.shape().NumAxes(), 2);
    const int64_t k = x_desc.shape().At(x_desc.shape().NumAxes() - 1);
    const user_op::TensorDesc& weight_desc = ctx->InputTensorDesc("weights", i);
    CHECK_EQ_OR_RETURN(weight_desc.shape().NumAxes(), 2);
    CHECK_EQ_OR_RETURN(weight_desc.shape().At(1), k);
    const int64_t n = weight_desc.shape().At(0);
    if (has_biases) {
      const user_op::TensorDesc& bias_desc = ctx->InputTensorDesc("biases", i);
      CHECK_EQ_OR_RETURN(bias_desc.shape().NumAxes(), 1);
      CHECK_EQ_OR_RETURN(bias_desc.shape().At(0), n);
    }
    user_op::TensorDesc* y_desc = ctx->MutOutputTensorDesc("ys", i);
    y_desc->set_data_type(data_type);
    DimVector out_dim_vec = x_desc.shape().dim_vec();
    out_dim_vec.back() = n;
    y_desc->set_shape(Shape(out_dim_vec));
  }
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> GroupedMatmulBiasOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> GroupedMatmulBiasOp::GetSbp(user_op::SbpContext* ctx) {
  {
    // s0 x b
    auto builder = ctx->NewBuilder();
    for (int64_t i = 0; i < ctx->user_op_conf().input_size("xs"); ++i) {
      builder.Split(user_op::OpArg("xs", i), 0);
    }
    for (int i = 0; i < ctx->user_op_conf().input_size("weights"); ++i) {
      builder.Broadcast(user_op::OpArg("weights", i));
    }
    for (int i = 0; i < ctx->user_op_conf().input_size("biases"); ++i) {
      builder.Broadcast(user_op::OpArg("biases", i));
    }
    for (int i = 0; i < ctx->user_op_conf().output_size("ys"); ++i) {
      builder.Split(user_op::OpArg("ys", i), 0);
    }
    builder.Build();
  }

  {
    // b x s0
    auto builder = ctx->NewBuilder();
    for (int64_t i = 0; i < ctx->user_op_conf().input_size("xs"); ++i) {
      builder.Broadcast(user_op::OpArg("xs", i));
    }
    for (int i = 0; i < ctx->user_op_conf().input_size("weights"); ++i) {
      builder.Split(user_op::OpArg("weights", i), 0);
    }
    for (int i = 0; i < ctx->user_op_conf().input_size("biases"); ++i) {
      builder.Split(user_op::OpArg("biases", i), 0);
    }
    for (int i = 0; i < ctx->user_op_conf().output_size("ys"); ++i) {
      builder.Split(user_op::OpArg("ys", i),
                    ctx->LogicalTensorDesc4InputArgNameAndIndex("xs", i).shape().NumAxes() - 1);
    }
    builder.Build();
  }

  return Maybe<void>::Ok();
}

/* static */ Maybe<void> GroupedMatmulBiasOp::InferDataType(user_op::InferContext* ctx) {
  const user_op::TensorDesc& first_in_desc = ctx->InputTensorDesc("xs", 0);
  for (const auto& in_arg_pair : ctx->inputs()) {
    const user_op::TensorDesc& in_desc =
        ctx->InputTensorDesc(in_arg_pair.first, in_arg_pair.second);
    CHECK_EQ_OR_RETURN(in_desc.data_type(), first_in_desc.data_type())
        << "InferDataType Failed. Expected " << DataType_Name(first_in_desc.data_type())
        << ", but got " << DataType_Name(in_desc.data_type());
  }
  for (int32_t i = 0; i < ctx->output_size("ys"); i++) {
    user_op::TensorDesc* y_desc = ctx->MutOutputTensorDesc("ys", i);
    y_desc->set_data_type(first_in_desc.data_type());
  }
  return Maybe<void>::Ok();
}

}  // namespace oneflow
