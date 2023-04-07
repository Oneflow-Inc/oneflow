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

/* static */ Maybe<void> FusedSinusoidalPositionalEncodeOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  const Shape& positions_shape = ctx->InputShape("positions", 0);
  CHECK_GE_OR_RETURN(positions_shape.NumAxes(), 1)
      << "number of axes of \'positions\' should be greater than or equal to 1, yet get "
      << positions_shape.NumAxes();

  const int embedding_dim = ctx->Attr<int>("embedding_dim");
  CHECK_GT_OR_RETURN(embedding_dim, 0)
      << "embedding_dim should be greater than 0, yet get " << embedding_dim;

  Shape out_shape(positions_shape.NumAxes() + 1);
  for (int i = 0; i < positions_shape.NumAxes(); i++) { out_shape[i] = positions_shape.At(i); }
  out_shape[positions_shape.NumAxes()] = embedding_dim;

  ctx->SetOutputShape("encoded_positions", 0, out_shape);

  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> FusedSinusoidalPositionalEncodeOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> FusedSinusoidalPositionalEncodeOp::GetSbp(user_op::SbpContext* ctx) {
  // TODO: I dont think it is necessary to do SBP...

  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FusedSinusoidalPositionalEncodeOp::InferDataType(
    user_op::InferContext* ctx) {
  user_op::TensorDesc* out_desc = ctx->MutOutputTensorDesc("encoded_positions", 0);
  out_desc->set_data_type(DataType::kFloat);

  return Maybe<void>::Ok();
}

}  // namespace oneflow
