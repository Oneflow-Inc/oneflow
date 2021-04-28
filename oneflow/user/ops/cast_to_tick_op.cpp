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

namespace {

REGISTER_USER_OP("cast_to_tick")
    .Input("in")
    .Output("out")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* out_shape = ctx->Shape4ArgNameAndIndex("out", 0);
      *out_shape = Shape({1});
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->Dtype4ArgNameAndIndex("out", 0) = *ctx->Dtype4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetParallelDistributionInferFn([](user_op::InferParallelDistributionFnContext* ctx)
                                        -> Maybe<void> {
      const ParallelDistribution& in_dis_hint =
          ctx->ParallelDistributionHint4InputArgNameAndIndex("in", 0);
      const Shape& parallel_hierarchy = ctx->parallel_hierarchy();
      CHECK_EQ(in_dis_hint.sbp_parallel_size(), parallel_hierarchy.NumAxes());

      ParallelDistribution* in_distribution = ctx->ParallelDistribution4ArgNameAndIndex("in", 0);
      ParallelDistribution* out_distribution = ctx->ParallelDistribution4ArgNameAndIndex("out", 0);
      in_distribution->clear_sbp_parallel();
      out_distribution->clear_sbp_parallel();
      // in use hint
      in_distribution->CopyFrom(in_dis_hint);

      for (int32_t i = 0; i < parallel_hierarchy.NumAxes(); ++i) {
        // out dim1 = broadcast
        out_distribution->add_sbp_parallel()->mutable_broadcast_parallel();
      }
      return Maybe<void>::Ok();
    });

}  // namespace

}  // namespace oneflow
