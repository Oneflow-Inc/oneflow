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

REGISTER_USER_OP("ctc_beam_search_decoder")
    .Input("log_probs")
    .Input("input_lengths")
    .Output("decoded")
    .Output("log_probability")
    .Attr<int32_t>("beam_width")
    .Attr<int32_t>("top_paths")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* log_probs = ctx->TensorDesc4ArgNameAndIndex("log_probs", 0);
      const user_op::TensorDesc* input_lengths =
          ctx->TensorDesc4ArgNameAndIndex("input_lengths", 0);
      const int64_t batch_size = log_probs->shape().At(1);
      const int64_t beam_width = ctx->Attr<int32_t>("beam_width");
      const int64_t top_paths = ctx->Attr<int32_t>("top_paths");
      CHECK_EQ_OR_RETURN(batch_size, input_lengths->shape().At(0));
      CHECK_GE_OR_RETURN(beam_width, 0);
      CHECK_GE_OR_RETURN(top_paths, 0);
      CHECK_LE_OR_RETURN(top_paths, beam_width);
      *ctx->OutputShape("decoded", 0) = Shape({batch_size, log_probs->shape().At(0)});
      *ctx->OutputShape("log_probability", 0) = Shape({batch_size, top_paths});
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder()
          .Split(user_op::OpArg("log_probs", 0), 1)  // `log_probs` batch axis is 1
          .Split(user_op::OpArg("input_lengths", 0), 0)
          .Split(user_op::OpArg("decoded", 0), 0)
          .Split(user_op::OpArg("log_probability", 0), 0)
          .Build();
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("decoded", 0) = ctx->InputDType("input_lengths", 0);
      *ctx->OutputDType("log_probability", 0) = ctx->InputDType("log_probs", 0);
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
