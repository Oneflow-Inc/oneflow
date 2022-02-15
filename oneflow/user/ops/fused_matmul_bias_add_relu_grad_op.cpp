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
#include "oneflow/core/common/just.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/infer_util.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

namespace {

Maybe<void> InferTensorDesc4FusedMatmulBackward(user_op::InferContext* ctx) {
  const user_op::TensorDesc& dy_desc = ctx->InputTensorDesc("dy", 0);
  *ctx->OutputShape("dx", 0) = dy_desc.shape();
 
  DimVector dbias_shape(1); 
  dbias_shape.at(0) = dy_desc.shape().At(1); 

  *ctx->OutputShape("dbias", 0) = Shape(dbias_shape);

  return Maybe<void>::Ok();
}


Maybe<void> InferDataType4MatmulBackward(user_op::InferContext* ctx){
  const user_op::TensorDesc& dy_desc = ctx->InputTensorDesc("dy", 0);
  user_op::TensorDesc* dx_desc = ctx->OutputTensorDesc("dx", 0);
  user_op::TensorDesc* dbias_desc = ctx->OutputTensorDesc("dbias", 0);

  *dx_desc->mut_data_type() = dy_desc.data_type();
  *dbias_desc->mut_data_type() = dy_desc.data_type();

  return Maybe<void>::Ok(); 
}


}  // namespace

/* static */ Maybe<void> FusedMatmulBiasAddReluBackwardOp::InferLogicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferTensorDesc4FusedMatmulBackward(ctx);
}

/*static*/ Maybe<void> FusedMatmulBiasAddReluBackwardOp::InferPhysicalTensorDesc(
    user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> FusedMatmulBiasAddReluBackwardOp::GetSbp(user_op::SbpContext* ctx) {
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> FusedMatmulBiasAddReluBackwardOp::InferDataType(user_op::InferContext* ctx) {
  return InferDataType4MatmulBackward(ctx);
}

}  // namespace oneflow
