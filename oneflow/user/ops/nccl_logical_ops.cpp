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
#include "oneflow/core/operator/operator.h"

namespace oneflow {

REGISTER_USER_OP("_nccl_logical_op_all_reduce")
    .Input("in")
    .Output("out")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->TensorDesc4ArgNameAndIndex("out", 0) = *ctx->TensorDesc4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn(user_op::BatchAxisInferFnUtil::NaiveInferBatchAxis)
    .SetInferSbpSignatureFn([](user_op::InferSbpSignatureFnContext* ctx) -> Maybe<void> {
      auto* bn2sbp = ctx->mutable_sbp_signature()->mutable_bn_in_op2sbp_parallel();
      const SbpParallel& in_sbp_hint = ctx->SbpParallelHint4InputArgNameAndIndex("in", 0);
      CHECK(in_sbp_hint.has_partial_sum_parallel());
      const std::string& ibn = GenRepeatedBn("in", 0);
      const std::string& obn = GenRepeatedBn("out", 0);
      SbpParallel in_p;
      in_p.mutable_partial_sum_parallel();
      (*bn2sbp)[ibn] = in_p;

      SbpParallel out_b;
      out_b.mutable_broadcast_parallel();
      (*bn2sbp)[obn] = out_b;
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("_nccl_logical_op_reduce_scatter")
    .Input("in")
    .Output("out")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->TensorDesc4ArgNameAndIndex("out", 0) = *ctx->TensorDesc4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetBatchAxisInferFn(user_op::BatchAxisInferFnUtil::NaiveInferBatchAxis)
    .SetInferSbpSignatureFn([](user_op::InferSbpSignatureFnContext* ctx) -> Maybe<void> {
      auto* bn2sbp = ctx->mutable_sbp_signature()->mutable_bn_in_op2sbp_parallel();
      const SbpParallel& in_sbp_hint = ctx->SbpParallelHint4InputArgNameAndIndex("in", 0);
      CHECK(in_sbp_hint.has_partial_sum_parallel());
      const std::string& ibn = GenRepeatedBn("in", 0);
      const std::string& obn = GenRepeatedBn("out", 0);
      SbpParallel in_p;
      in_p.mutable_partial_sum_parallel();
      (*bn2sbp)[ibn] = in_p;

      SbpParallel out_b;
      out_b.mutable_split_parallel()->set_axis(0);
      (*bn2sbp)[obn] = out_b;
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
