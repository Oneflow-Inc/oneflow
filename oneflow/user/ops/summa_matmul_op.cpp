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

REGISTER_USER_OP("summa_matmul_ab")
    .Input("a")
    .Input("b")
    .Output("out")
    .Attr<double>("alpha", 1.0)
    .SetLogicalTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc* a = ctx->TensorDesc4ArgNameAndIndex("a", 0);
      const user_op::TensorDesc* b = ctx->TensorDesc4ArgNameAndIndex("b", 0);
      const int32_t num_axes = a->shape().NumAxes();
      CHECK_EQ_OR_RETURN(a->shape().NumAxes(), 2);
      CHECK_EQ_OR_RETURN(b->shape().NumAxes(), 2);
      const int m = a->shape().At(0);
      const int n = b->shape().At(1);
      *ctx->Shape4ArgNameAndIndex("out", 0) = Shape({m, n});
      *ctx->IsDynamic4ArgNameAndIndex("out", 0) = *ctx->IsDynamic4ArgNameAndIndex("a", 0);
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->Dtype4ArgNameAndIndex("out", 0) = *ctx->Dtype4ArgNameAndIndex("a", 0);
      return Maybe<void>::Ok();
    })
    .SetParallelDistributionInferFn(
        [](user_op::InferParallelDistributionFnContext* ctx) -> Maybe<void> {
          ParallelDistribution parallel_distribution;
          parallel_distribution.add_sbp_parallel()->mutable_split_parallel()->set_axis(0);
          parallel_distribution.add_sbp_parallel()->mutable_split_parallel()->set_axis(1);
          *ctx->ParallelDistribution4ArgNameAndIndex("a", 0) = parallel_distribution;
          *ctx->ParallelDistribution4ArgNameAndIndex("b", 0) = parallel_distribution;
          *ctx->ParallelDistribution4ArgNameAndIndex("out", 0) = parallel_distribution;
          return Maybe<void>::Ok();
        });
}
