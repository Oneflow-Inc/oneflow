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

/* static */ auto AddBiasResidualLayerNormOp::GetSbp(user_op::SbpContext* ctx) -> Maybe<void> {
  return Maybe<void>::Ok();
}

/* static */ auto AddBiasResidualLayerNormOp::InferLogicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  // check shape of x
  const Shape& x_shape = ctx->InputShape("x", 0);
  CHECK_GT_OR_RETURN(x_shape.NumAxes(), 1)
      << "number of axes of \'x\' should have be greater than 1, yet get " << x_shape.NumAxes();

#define GAMMA_BETA_BIAS_SHAPE_CHECK(tensor) \
  const Shape& tensor##_shape = ctx->InputShape(#tensor, 0); \
  CHECK_EQ_OR_RETURN(tensor##_shape.NumAxes(), 1) \
        << "number of axes of \'" << #tensor << "\' should have be greater than 1, yet get " \
        << tensor##_shape.NumAxes(); \
  CHECK_EQ_OR_RETURN(tensor##_shape.At(0), x_shape.At(x_shape.NumAxes() - 1)) \
        << "dimension 1 of \'" << #tensor << "\'(" << tensor##_shape.At(0) \
        << ") is not consistant with the last dimension of \'x\'(" \
        << x_shape.At(x_shape.NumAxes() - 1) << ")"; \

  // check shape of gamma, beta and pre_bias
  if (ctx->has_input("gamma", 0)) { GAMMA_BETA_BIAS_SHAPE_CHECK(gamma); }
  if (ctx->has_input("beta", 0)) { GAMMA_BETA_BIAS_SHAPE_CHECK(beta); }
  if (ctx->has_input("pre_bias", 0)) { GAMMA_BETA_BIAS_SHAPE_CHECK(pre_bias); }

#undef GAMMA_BETA_BIAS_SHAPE_CHECK

  // check shape of residual
  if (ctx->has_input("pre_residual_1", 0)) {
    const Shape& pre_residual_1_shape = ctx->InputShape("pre_residual_1", 0);
    CHECK_EQ_OR_RETURN(pre_residual_1_shape, x_shape)
      << "shape of \'pre_residual_1\' is not the same as \'x\'";
  }
  if (ctx->has_input("pre_residual_2", 0)) {
    CHECK_OR_RETURN(ctx->has_input("pre_residual_2", 0))
        << "must provide pre_residual_1 while pre_residual_2 is provided";
    const Shape& pre_residual_2_shape = ctx->InputShape("pre_residual_2", 0);
    CHECK_EQ_OR_RETURN(pre_residual_2_shape, x_shape)
      << "shape of \'pre_residual_2\' is not the same as \'x\'";
  }

  // set output shape of y
  Shape y_shape = x_shape;  // borrow from input shape
  user_op::TensorDesc* y_tensor = ctx->MutOutputTensorDesc("y", 0);
  y_tensor->set_shape(y_shape);

  // set output shape of mean and varience
  Shape mean_shape = x_shape;  // borrow from input shape
  size_t mean_num_axes = x_shape.NumAxes();
  mean_shape.Set(mean_num_axes - 1, 1);
  user_op::TensorDesc* mean_tensor = ctx->MutOutputTensorDesc("mean", 0);
  user_op::TensorDesc* varience_tensor = ctx->MutOutputTensorDesc("inv_variance", 0);
  mean_tensor->set_shape(mean_shape);
  varience_tensor->set_shape(mean_shape);

  return Maybe<void>::Ok();
}

/* static */ auto AddBiasResidualLayerNormOp::InferPhysicalTensorDesc(user_op::InferContext* ctx)
    -> Maybe<void> {
  return InferLogicalTensorDesc(ctx);
}

/* static */ auto AddBiasResidualLayerNormOp::InferDataType(user_op::InferContext* ctx)
    -> Maybe<void> {
  // obtain input data types
  DataType x_dtype = ctx->InputDType("x", 0);

  // check data type of gamma
  if (ctx->has_input("gamma", 0)) {
    CHECK_EQ_OR_RETURN(ctx->InputDType("gamma", 0), x_dtype)
        << "data type of \'gamma\' is not consitant with \'x\'";
  }

  // check data type of pre_bias
  if (ctx->has_input("pre_bias", 0)) {
    CHECK_EQ_OR_RETURN(ctx->InputDType("pre_bias", 0), x_dtype)
        << "data type of \'pre_bias\' is not consitant with \'x\'";
  }

  // check data types of beta
  if (ctx->has_input("beta", 0)) {
    CHECK_EQ_OR_RETURN(ctx->InputDType("beta", 0), x_dtype)
        << "data type of \'beta\' is not consitant with \'x\'";
  }

  // check data types of pre_residual_1 and pre_residual_2
  if (ctx->has_input("pre_residual_1", 0)) {
    CHECK_EQ_OR_RETURN(ctx->InputDType("pre_residual_1", 0), x_dtype)
        << "data type of \'pre_residual_1\' is not consitant with \'x\'";
  }
  if (ctx->has_input("pre_residual_2", 0)) {
    CHECK_EQ_OR_RETURN(ctx->InputDType("pre_residual_2", 0), x_dtype)
        << "data type of \'pre_residual_2\' is not consitant with \'x\'";
  }

  // set output data type
  ctx->SetOutputDType("y", 0, x_dtype);
  ctx->SetOutputDType("mean", 0, x_dtype);
  ctx->SetOutputDType("inv_variance", 0, x_dtype);

  return Maybe<void>::Ok();
}

}  // namespace oneflow