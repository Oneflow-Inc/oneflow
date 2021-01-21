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

namespace oneflow {
namespace {
using namespace user_op;

Maybe<void> GetSbpSignature(SbpContext* ctx) {
  const Shape& mask_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("mask", 0).shape();
  const Shape& dz_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0).shape();

  FOR_RANGE(int64_t, i, 0, mask_shape.NumAxes()) {
    if (mask_shape.At(i) == 1) { continue; }
    if (mask_shape.At(i) == dz_shape.At(i)) {
      ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
    } else {
      UNIMPLEMENTED();
    }
  }
  return Maybe<void>::Ok();
}

Maybe<void> InferTensorDesc(InferContext* ctx) {
  const TensorDesc* tensor_mask = ctx->TensorDesc4ArgNameAndIndex("mask", 0);
  const TensorDesc* tensor_dz = ctx->TensorDesc4ArgNameAndIndex("in", 0);

  CHECK_EQ_OR_RETURN(tensor_mask->shape().NumAxes(), tensor_dz->shape().NumAxes())
      << "Shape of tensor x and y should be same";

  FOR_RANGE(int64_t, i, 0, tensor_mask->shape().NumAxes()) {
    CHECK_EQ_OR_RETURN(tensor_mask->shape().At(i), tensor_dz->shape().At(i));
  }

  TensorDesc* tensor_dx = ctx->TensorDesc4ArgNameAndIndex("out_true", 0);
  TensorDesc* tensor_dy = ctx->TensorDesc4ArgNameAndIndex("out_false", 0);

  if (tensor_dx) {
    *tensor_dx->mut_data_type() = tensor_dz->data_type();
    *tensor_dx->mut_shape() = tensor_dz->shape();
  }

  if (tensor_dy) {
    *tensor_dy->mut_data_type() = tensor_dz->data_type();
    *tensor_dy->mut_shape() = tensor_dz->shape();
  }

  return Maybe<void>::Ok();
}

Maybe<void> InferBatchAxis(user_op::BatchAxisContext* ctx) {
  OptInt64* dz_batch_axis = ctx->BatchAxis4ArgNameAndIndex("in", 0);
  if (dz_batch_axis->has_value()) {
    CHECK_GE_OR_RETURN(dz_batch_axis->value(), 0);
    CHECK_LE_OR_RETURN(dz_batch_axis->value(),
                       ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0).shape().NumAxes() - 1);
  }
  if (ctx->user_op_conf().has_input("out_true", 0)) {
    *ctx->BatchAxis4ArgNameAndIndex("out_true", 0) = *dz_batch_axis;
  }
  if (ctx->user_op_conf().has_input("out_false", 0)) {
    *ctx->BatchAxis4ArgNameAndIndex("out_false", 0) = *dz_batch_axis;
  }
  return Maybe<void>::Ok();
}
}  // namespace

REGISTER_USER_OP("masked_fork")
    .Input("in")
    .Input("mask")
    .OptionalOutput("out_true")
    .OptionalOutput("out_false")
    .SetTensorDescInferFn(InferTensorDesc)
    .SetGetSbpFn(GetSbpSignature)
    .SetBatchAxisInferFn(InferBatchAxis);
}  // namespace oneflow
