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

namespace user_op {

Maybe<void> InferTensorMinimumDesc(InferContext* ctx) {
  // backward(dz, x, y) -> dx, dy
  const TensorDesc* tensor_x = ctx->TensorDesc4ArgNameAndIndex("x", 0);
  const TensorDesc* tensor_y = ctx->TensorDesc4ArgNameAndIndex("y", 0);
  const TensorDesc* tensor_dz = ctx->TensorDesc4ArgNameAndIndex("dz", 0);

  TensorDesc* tensor_dx = ctx->TensorDesc4ArgNameAndIndex("dx", 0);
  TensorDesc* tensor_dy = ctx->TensorDesc4ArgNameAndIndex("dy", 0);

  *tensor_dx->mut_data_type() = tensor_dz->data_type();
  *tensor_dx->mut_shape() = tensor_x->shape();

  *tensor_dy->mut_data_type() = tensor_dz->data_type();
  *tensor_dy->mut_shape() = tensor_y->shape();

  return Maybe<void>::Ok();
}

REGISTER_USER_OP("broadcast_minimum_backward")
    .Input("dz")
    .Input("x")
    .Input("y")
    .Output("dx")
    .Output("dy")
    .SetTensorDescInferFn(InferTensorMinimumDesc);

// TODO: Add BatchAxisInfer and SBP

//.SetBatchAxisInferFn(user_op::BatchAxisInferFnUtil::NaiveInferBatchAxis);
// .SetGetSbpFn(GetBinaryBroadcastSbpSignature<BinaryFunc##sbp_suffix>);

}  // namespace user_op

}  // namespace oneflow
