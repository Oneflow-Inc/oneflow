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

REGISTER_USER_OP("ctc_loss")
    .Input("log_probs")
    .Input("targets")
    .Input("input_lengths")
    .Input("target_lengths")
    .Output("loss")
    .Attr<int>("blank")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* log_probs = ctx->TensorDesc4ArgNameAndIndex("log_probs", 0);
      const user_op::TensorDesc* targets = ctx->TensorDesc4ArgNameAndIndex("targets", 0);
      const user_op::TensorDesc* input_lengths =
          ctx->TensorDesc4ArgNameAndIndex("input_lengths", 0);
      const user_op::TensorDesc* target_lengths =
          ctx->TensorDesc4ArgNameAndIndex("target_lengths", 0);
      CHECK_EQ_OR_RETURN(log_probs->shape().At(1), targets->shape().At(0));
      CHECK_EQ_OR_RETURN(log_probs->shape().At(1), input_lengths->shape().At(0));
      CHECK_EQ_OR_RETURN(log_probs->shape().At(1), target_lengths->shape().At(0));
      CHECK_GE_OR_RETURN(ctx->Attr<int>("blank"), 0);
      *ctx->Shape4ArgNameAndIndex("loss", 0) = Shape({log_probs->shape().At(1)});
      *ctx->Dtype4ArgNameAndIndex("loss", 0) = *ctx->Dtype4ArgNameAndIndex("log_probs", 0);
      return Maybe<void>::Ok();
    });
// .SetBatchAxisInferFn([](user_op::BatchAxisContext* ctx) -> Maybe<void> {
//   *ctx->BatchAxis4ArgNameAndIndex("loss", 0) = *ctx->BatchAxis4ArgNameAndIndex("log_probs", 0);
//   return Maybe<void>::Ok();
// })
// .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
//   const user_op::TensorDesc& prediction_tensor =
//       ctx->LogicalTensorDesc4InputArgNameAndIndex("log_probs", 0);
//   FOR_RANGE(int64_t, i, 0, prediction_tensor.shape().NumAxes()) {
//     ctx->NewBuilder().Split(ctx->inputs(), i).Split(ctx->outputs(), i).Build();
//   }
//   return Maybe<void>::Ok();
// });

}  // namespace oneflow
