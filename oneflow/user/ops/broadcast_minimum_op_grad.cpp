// /*
// Copyright 2020 The OneFlow Authors. All rights reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// */
// #include <cstdint>
// #include "oneflow/core/common/maybe.h"
// #include "oneflow/core/common/util.h"
// #include "oneflow/core/framework/framework.h"

// namespace oneflow {

// namespace {
// Maybe<void> GetSbpSignature(user_op::SbpContext* ctx) {
//   const Shape& x_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("x", 0).shape();
//   const Shape& y_shape = ctx->LogicalTensorDesc4InputArgNameAndIndex("y", 0).shape();

//   FOR_RANGE(int64_t, i, 0, x_shape.NumAxes()) {
//     if (x_shape.At(i) == 1 && y_shape.At(i) == 1) { continue; }
//     if (x_shape.At(i) == y_shape.At(i)) {
//       ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
//     } else {
//       UNIMPLEMENTED();
//     }
//   }
//   return Maybe<void>::Ok();
// }
// }  // namespace

// namespace user_op {

// Maybe<void> InferTensorMaximumDesc(InferContext* ctx) {
//   // backward(dz, x, y) -> dx, dy
//   const TensorDesc* tensor_x = ctx->TensorDesc4ArgNameAndIndex("x", 0);
//   const TensorDesc* tensor_y = ctx->TensorDesc4ArgNameAndIndex("y", 0);
//   const TensorDesc* tensor_dz = ctx->TensorDesc4ArgNameAndIndex("dz", 0);

//   CHECK_EQ_OR_RETURN(tensor_x->shape().NumAxes(), tensor_y->shape().NumAxes())
//       << "Shape of tensor x and y should be same";

//   FOR_RANGE(int64_t, i, 0, tensor_x->shape().NumAxes()) {
//     CHECK_EQ_OR_RETURN(tensor_x->shape().At(i), tensor_y->shape().At(i));
//   }

//   TensorDesc* tensor_dx = ctx->TensorDesc4ArgNameAndIndex("dx", 0);
//   TensorDesc* tensor_dy = ctx->TensorDesc4ArgNameAndIndex("dy", 0);

//   *tensor_dx->mut_data_type() = tensor_dz->data_type();
//   *tensor_dx->mut_shape() = tensor_x->shape();

//   *tensor_dy->mut_data_type() = tensor_dz->data_type();
//   *tensor_dy->mut_shape() = tensor_y->shape();

//   return Maybe<void>::Ok();
// }

// REGISTER_USER_OP("broadcast_maximum_backward")
//     .Input("dz")
//     .Input("x")
//     .Input("y")
//     .Output("dx")
//     .Output("dy")
//     .SetTensorDescInferFn(InferTensorMaximumDesc)
//     .SetGetSbpFn(GetSbpSignature)
//     .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
//       OptInt64* dz_batch_axis = ctx->BatchAxis4ArgNameAndIndex("dz", 0);
//       if (dz_batch_axis->has_value()) {
//         CHECK_GE_OR_RETURN(dz_batch_axis->value(), 0);
//         CHECK_LE_OR_RETURN(
//             dz_batch_axis->value(),
//             ctx->LogicalTensorDesc4InputArgNameAndIndex("dz", 0).shape().NumAxes() - 1);
//       }
//       *ctx->BatchAxis4ArgNameAndIndex("dx", 0) = *dz_batch_axis;
//       *ctx->BatchAxis4ArgNameAndIndex("dy", 0) = *dz_batch_axis;
//       return Maybe<void>::Ok();
//     });

// REGISTER_USER_OP("broadcast_minimum_backward")
//     .Input("dz")
//     .Input("x")
//     .Input("y")
//     .Output("dx")
//     .Output("dy")
//     .SetTensorDescInferFn(InferTensorMaximumDesc) // same as maximum
//     .SetGetSbpFn(GetSbpSignature)
//     .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
//       OptInt64* dz_batch_axis = ctx->BatchAxis4ArgNameAndIndex("dz", 0);
//       if (dz_batch_axis->has_value()) {
//         CHECK_GE_OR_RETURN(dz_batch_axis->value(), 0);
//         CHECK_LE_OR_RETURN(
//             dz_batch_axis->value(),
//             ctx->LogicalTensorDesc4InputArgNameAndIndex("dz", 0).shape().NumAxes() - 1);
//       }
//       *ctx->BatchAxis4ArgNameAndIndex("dx", 0) = *dz_batch_axis;
//       *ctx->BatchAxis4ArgNameAndIndex("dy", 0) = *dz_batch_axis;
//       return Maybe<void>::Ok();
//     });

// }  // namespace user_op

// }  // namespace oneflow
