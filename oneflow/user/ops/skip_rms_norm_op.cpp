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

oneflow::DataType InferParamDataType(const DataType x_data_type) {
  return (x_data_type == DataType::kFloat16 || x_data_type == DataType::kBFloat16)
             ? DataType::kFloat
             : x_data_type;
}

}  // namespace

/* static */ auto SkipRmsNormOp::GetSbp(user_op::SbpContext* ctx) -> Maybe<void> {
  for (int64_t i = 0; i < ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0).shape().NumAxes() - 1;
       ++i) {
    ctx->NewBuilder()
        .Split(user_op::OpArg("x", 0), i)
        .Split(user_op::OpArg("skip", 0), i)
        .Broadcast(user_op::OpArg("bias", 0))
        .Broadcast(user_op::OpArg("weight", 0))
        .Split(ctx->outputs(), i)
        .Build();
  }

  return Maybe<void>::Ok();
}

/* static */ auto SkipRmsNormOp::InferLogicalTensorDesc(user_op::InferContext* ctx) -> Maybe<void> {
  // check shape of x
  const Shape& x_shape = ctx->InputShape("x", 0);
  CHECK_GE_OR_RETURN(x_shape.NumAxes(), 2)
      << "number of axes of \'x\' should have be greater than or equal to 2, yet get "
      << x_shape.NumAxes();

  // check shape of weight and bias
  if (ctx->has_input("weight", 0)) {
    const Shape& weight_shape = ctx->InputShape("weight", 0);
    CHECK_EQ_OR_RETURN(weight_shape.NumAxes(), 1)
        << "number of axes of \'weight\' should be equal to 1, yet get " << weight_shape.NumAxes();
    CHECK_EQ_OR_RETURN(weight_shape.At(0), x_shape.At(x_shape.NumAxes() - 1))
        << "the size of \'weight\'(" << weight_shape.At(0)
        << ") is not consistant with the last dimension of \'x\'("
        << x_shape.At(x_shape.NumAxes() - 1) << ")";
  }
  if (ctx->has_input("bias", 0)) {
    const Shape& bias_shape = ctx->InputShape("bias", 0);
    CHECK_EQ_OR_RETURN(bias_shape.NumAxes(), 1)
        << "number of axes of \'bias\' should be equal to 1, yet get " << bias_shape.NumAxes();
    CHECK_EQ_OR_RETURN(bias_shape.At(0), x_shape.At(x_shape.NumAxes() - 1))
        << "the size of \'bias\'(" << bias_shape.At(0)
        << ") is not consistant with the last dimension of \'x\'("
        << x_shape.At(x_shape.NumAxes() - 1) << ")";
  }

  // check shape of skip
  if (ctx->has_input("skip", 0)) {
    const Shape& skip_shape = ctx->InputShape("skip", 0);
    CHECK_EQ_OR_RETURN(skip_shape, x_shape) << "shape of \'skip\' is not the same as \'x\'";
  }

  // set output shape of y
  user_op::TensorDesc* y_tensor = ctx->MutOutputTensorDesc("y", 0);
  y_tensor->set_shape(x_shape);

  // set output shape of inv_rms
  DimVector inv_rms_dim_vec;
  inv_rms_dim_vec.push_back(x_shape.Count(0, x_shape.NumAxes() - 1));
  Shape inv_rms_shape(inv_rms_dim_vec);
  user_op::TensorDesc* inv_rms_tensor = ctx->MutOutputTensorDesc("inv_rms", 0);
  inv_rms_tensor->set_shape(inv_rms_shape);

  return Maybe<void>::Ok();
}

/* static */ auto SkipRmsNormOp::InferPhysicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  return InferLogicalTensorDesc(ctx);
}

/* static */ auto SkipRmsNormOp::InferDataType(user_op::InferContext* ctx) -> Maybe<void> {
  // obtain input data types
  DataType x_dtype = ctx->InputDType("x", 0);

  // check data type of bias
  if (ctx->has_input("bias", 0)) {
    CHECK_EQ_OR_RETURN(ctx->InputDType("bias", 0), x_dtype)
        << "data type of \'bias\' is not consitant with \'x\'";
  }

  // check data types of weight
  if (ctx->has_input("weight", 0)) {
    CHECK_EQ_OR_RETURN(ctx->InputDType("weight", 0), x_dtype)
        << "data type of \'weight\' is not consitant with \'x\'";
  }

  // check data types of skip
  if (ctx->has_input("skip", 0)) {
    CHECK_EQ_OR_RETURN(ctx->InputDType("skip", 0), x_dtype)
        << "data type of \'skip\' is not consitant with \'x\'";
  }

  // set output data type
  ctx->SetOutputDType("y", 0, x_dtype);
  ctx->SetOutputDType("inv_rms", 0, InferParamDataType(x_dtype));

  return Maybe<void>::Ok();
}

}  // namespace oneflow
