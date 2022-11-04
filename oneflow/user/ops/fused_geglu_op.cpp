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

/* static */ auto FusedGegluOp::GetSbp(user_op::SbpContext* ctx) -> Maybe<void> {
  // TODO
  return Maybe<void>::Ok();
}

/* static */ auto FusedGegluOp::InferLogicalTensorDesc(user_op::InferContext* ctx) -> Maybe<void> {
  // obtain input shape
  const Shape& input_shape = ctx->InputShape("in", 0);
  const Shape& weight_shape = ctx->InputShape("weight", 0);
  const Shape& bias_shape = ctx->InputShape("bias", 0);

  // check dimensions
  CHECK_EQ_OR_RETURN(input_shape.NumAxes(), 2);
  CHECK_EQ_OR_RETURN(weight_shape.NumAxes(), 2);
  CHECK_EQ_OR_RETURN(bias_shape.NumAxes(), 1);

  // check shape
  CHECK_EQ_OR_RETURN(weight_shape.At(1), input_shape.At(1))
      << "get " << weight_shape.At(1) << " and " << input_shape.At(1);
  CHECK_EQ_OR_RETURN(bias_shape.At(0), weight_shape.At(0))
      << "get " << bias_shape.At(1) << " and " << weight_shape.At(0);

  // set output shape
  CHECK_EQ_OR_RETURN(weight_shape.At(0) % 2, 0);
  ctx->SetOutputShape("out", 0, Shape({input_shape.At(0), weight_shape.At(0) / 2}));

  // set malmul ouput shape
  ctx->SetOutputShape("matmul_out", 0, Shape({input_shape.At(0), weight_shape.At(0)}));

  return Maybe<void>::Ok();
}

/* static */ auto FusedGegluOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) -> Maybe<void> {
  return InferLogicalTensorDesc(ctx);
}

/* static */ auto FusedGegluOp::InferDataType(user_op::InferContext* ctx) -> Maybe<void> {
  // obtain input data types
  DataType input_dtype = ctx->InputDType("in", 0);
  DataType weight_dtype = ctx->InputDType("weight", 0);
  DataType bias_dtype = ctx->InputDType("bias", 0);

  // check types
  CHECK_EQ_OR_RETURN(input_dtype, weight_dtype);
  CHECK_EQ_OR_RETURN(input_dtype, bias_dtype);

  // set output data type
  ctx->SetOutputDType("out", 0, input_dtype);
  ctx->SetOutputDType("matmul_out", 0, input_dtype);

  return Maybe<void>::Ok();
}

}  // namespace oneflow