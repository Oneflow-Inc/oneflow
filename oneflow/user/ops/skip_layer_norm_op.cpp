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

oneflow::DataType InferBnParamDataType(const DataType x_data_type) {
  return (x_data_type == DataType::kFloat16 || x_data_type == DataType::kBFloat16)
             ? DataType::kFloat
             : x_data_type;
}

}  // namespace

/* static */ auto SkipLayerNormOp::GetSbp(user_op::SbpContext* ctx) -> Maybe<void> {
  return Maybe<void>::Ok();
}

/* static */ auto SkipLayerNormOp::InferLogicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  // check shape of x
  const Shape& x_shape = ctx->InputShape("x", 0);
  CHECK_GT_OR_RETURN(x_shape.NumAxes(), 1)
      << "number of axes of \'x\' should have be greater than 1, yet get " << x_shape.NumAxes();

#define GAMMA_BETA_BIAS_SHAPE_CHECK(tensor)                                                \
  const Shape& tensor##_shape = ctx->InputShape(#tensor, 0);                               \
  CHECK_EQ_OR_RETURN(tensor##_shape.NumAxes(), 1)                                          \
      << "number of axes of \'" << #tensor << "\' should have be greater than 1, yet get " \
      << tensor##_shape.NumAxes();                                                         \
  CHECK_EQ_OR_RETURN(tensor##_shape.At(0), x_shape.At(x_shape.NumAxes() - 1))              \
      << "dimension 1 of \'" << #tensor << "\'(" << tensor##_shape.At(0)                   \
      << ") is not consistant with the last dimension of \'x\'("                           \
      << x_shape.At(x_shape.NumAxes() - 1) << ")";

  // check shape of gamma, beta and bias
  if (ctx->has_input("gamma", 0)) { GAMMA_BETA_BIAS_SHAPE_CHECK(gamma); }
  if (ctx->has_input("beta", 0)) { GAMMA_BETA_BIAS_SHAPE_CHECK(beta); }
  if (ctx->has_input("bias", 0)) { GAMMA_BETA_BIAS_SHAPE_CHECK(bias); }

#undef GAMMA_BETA_BIAS_SHAPE_CHECK

  // check shape of residual
  if (ctx->has_input("skip", 0)) {
    const Shape& skip_shape = ctx->InputShape("skip", 0);
    CHECK_EQ_OR_RETURN(skip_shape, x_shape) << "shape of \'skip\' is not the same as \'x\'";
  }

  // set output shape of y
  Shape y_shape = x_shape;  // borrow from input shape
  user_op::TensorDesc* y_tensor = ctx->MutOutputTensorDesc("y", 0);
  y_tensor->set_shape(y_shape);

  // set output shape of mean and varience
  DimVector mean_dim_vec;
  mean_dim_vec.push_back(x_shape.Count(0, x_shape.NumAxes() - 1));
  Shape mean_shape(mean_dim_vec);  // borrow from input shape

  user_op::TensorDesc* mean_tensor = ctx->MutOutputTensorDesc("mean", 0);
  user_op::TensorDesc* varience_tensor = ctx->MutOutputTensorDesc("inv_variance", 0);
  mean_tensor->set_shape(mean_shape);
  varience_tensor->set_shape(mean_shape);

  return Maybe<void>::Ok();
}

/* static */ auto SkipLayerNormOp::InferPhysicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  return InferLogicalTensorDesc(ctx);
}

/* static */ auto SkipLayerNormOp::InferDataType(user_op::InferContext* ctx) -> Maybe<void> {
  // obtain input data types
  DataType x_dtype = ctx->InputDType("x", 0);

  // check data type of gamma
  if (ctx->has_input("gamma", 0)) {
    CHECK_EQ_OR_RETURN(ctx->InputDType("gamma", 0), x_dtype)
        << "data type of \'gamma\' is not consitant with \'x\'";
  }

  // check data type of pre_bias
  if (ctx->has_input("bias", 0)) {
    CHECK_EQ_OR_RETURN(ctx->InputDType("bias", 0), x_dtype)
        << "data type of \'bias\' is not consitant with \'x\'";
  }

  // check data types of beta
  if (ctx->has_input("beta", 0)) {
    CHECK_EQ_OR_RETURN(ctx->InputDType("beta", 0), x_dtype)
        << "data type of \'beta\' is not consitant with \'x\'";
  }

  // check data types of pre_residual_1 and pre_residual_2
  if (ctx->has_input("skip", 0)) {
    CHECK_EQ_OR_RETURN(ctx->InputDType("skip", 0), x_dtype)
        << "data type of \'skip\' is not consitant with \'x\'";
  }

  // set output data type
  ctx->SetOutputDType("y", 0, x_dtype);
  ctx->SetOutputDType("mean", 0, InferBnParamDataType(x_dtype));
  ctx->SetOutputDType("inv_variance", 0, InferBnParamDataType(x_dtype));

  return Maybe<void>::Ok();
}

}  // namespace oneflow