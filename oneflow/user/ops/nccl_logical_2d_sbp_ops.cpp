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

REGISTER_USER_OP("_nccl_logical_2D_same_dim0_all_reduce")
    .Input("in")
    .Output("out")
    .Attr<std::vector<std::string>>("in_distribution")
    .Attr<std::vector<std::string>>("out_distribution")
    .SetLogicalTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputShape("out", 0) = ctx->InputShape("in", 0);
      *ctx->OutputIsDynamic("out", 0) = ctx->InputIsDynamic("in", 0);
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("out", 0) = ctx->InputDType("in", 0);
      return Maybe<void>::Ok();
    })
    .SetParallelDistributionInferFn([](user_op::InferParallelDistributionFnContext* ctx)
                                        -> Maybe<void> {
      cfg::ParallelDistribution* in_distribution =
          ctx->ParallelDistribution4ArgNameAndIndex("in", 0);
      cfg::ParallelDistribution* out_distribution =
          ctx->ParallelDistribution4ArgNameAndIndex("out", 0);
      in_distribution->clear_sbp_parallel();
      out_distribution->clear_sbp_parallel();
      ParseParallelDistributionFromStringVec(
          ctx->user_op_conf().attr<std::vector<std::string>>("in_distribution"), in_distribution);
      ParseParallelDistributionFromStringVec(
          ctx->user_op_conf().attr<std::vector<std::string>>("out_distribution"), out_distribution);
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("_nccl_logical_2D_same_dim1_all_reduce")
    .Input("in")
    .Output("out")
    .Attr<std::vector<std::string>>("in_distribution")
    .Attr<std::vector<std::string>>("out_distribution")
    .SetLogicalTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputShape("out", 0) = ctx->InputShape("in", 0);
      *ctx->OutputIsDynamic("out", 0) = ctx->InputIsDynamic("in", 0);
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("out", 0) = ctx->InputDType("in", 0);
      return Maybe<void>::Ok();
    })
    .SetParallelDistributionInferFn([](user_op::InferParallelDistributionFnContext* ctx)
                                        -> Maybe<void> {
      cfg::ParallelDistribution* in_distribution =
          ctx->ParallelDistribution4ArgNameAndIndex("in", 0);
      cfg::ParallelDistribution* out_distribution =
          ctx->ParallelDistribution4ArgNameAndIndex("out", 0);
      in_distribution->clear_sbp_parallel();
      out_distribution->clear_sbp_parallel();

      ParseParallelDistributionFromStringVec(
          ctx->user_op_conf().attr<std::vector<std::string>>("in_distribution"), in_distribution);
      ParseParallelDistributionFromStringVec(
          ctx->user_op_conf().attr<std::vector<std::string>>("out_distribution"), out_distribution);

      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("_nccl_logical_2D_same_dim0_all_gather")
    .Input("in")
    .Output("out")
    .Attr<std::vector<std::string>>("in_distribution")
    .Attr<std::vector<std::string>>("out_distribution")
    .SetLogicalTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputShape("out", 0) = ctx->InputShape("in", 0);
      *ctx->OutputIsDynamic("out", 0) = ctx->InputIsDynamic("in", 0);
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("out", 0) = ctx->InputDType("in", 0);
      return Maybe<void>::Ok();
    })
    .SetParallelDistributionInferFn([](user_op::InferParallelDistributionFnContext* ctx)
                                        -> Maybe<void> {
      cfg::ParallelDistribution* in_distribution =
          ctx->ParallelDistribution4ArgNameAndIndex("in", 0);
      cfg::ParallelDistribution* out_distribution =
          ctx->ParallelDistribution4ArgNameAndIndex("out", 0);
      in_distribution->clear_sbp_parallel();
      out_distribution->clear_sbp_parallel();

      ParseParallelDistributionFromStringVec(
          ctx->user_op_conf().attr<std::vector<std::string>>("in_distribution"), in_distribution);
      ParseParallelDistributionFromStringVec(
          ctx->user_op_conf().attr<std::vector<std::string>>("out_distribution"), out_distribution);

      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("_nccl_logical_2D_same_dim0_all_gather_noncontinuous")
    .Input("in")
    .Output("out")
    .Attr<std::vector<std::string>>("in_distribution")
    .Attr<std::vector<std::string>>("out_distribution")
    .Attr<int64_t>("in_dim1_split_axis", -1)
    .SetLogicalTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputShape("out", 0) = ctx->InputShape("in", 0);
      *ctx->OutputIsDynamic("out", 0) = ctx->InputIsDynamic("in", 0);
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("out", 0) = ctx->InputDType("in", 0);
      return Maybe<void>::Ok();
    })
    .SetParallelDistributionInferFn([](user_op::InferParallelDistributionFnContext* ctx)
                                        -> Maybe<void> {
      cfg::ParallelDistribution* in_distribution =
          ctx->ParallelDistribution4ArgNameAndIndex("in", 0);
      cfg::ParallelDistribution* out_distribution =
          ctx->ParallelDistribution4ArgNameAndIndex("out", 0);
      in_distribution->clear_sbp_parallel();
      out_distribution->clear_sbp_parallel();

      ParseParallelDistributionFromStringVec(
          ctx->user_op_conf().attr<std::vector<std::string>>("in_distribution"), in_distribution);
      ParseParallelDistributionFromStringVec(
          ctx->user_op_conf().attr<std::vector<std::string>>("out_distribution"), out_distribution);

      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("_nccl_logical_2D_same_dim0_all2all")
    .Input("in")
    .Output("out")
    .Attr<std::vector<std::string>>("in_distribution")
    .Attr<std::vector<std::string>>("out_distribution")
    .Attr<int64_t>("in_dim1_split_axis", -1)
    .Attr<int64_t>("out_dim1_split_axis", -1)
    .SetLogicalTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputShape("out", 0) = ctx->InputShape("in", 0);
      *ctx->OutputIsDynamic("out", 0) = ctx->InputIsDynamic("in", 0);
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("out", 0) = ctx->InputDType("in", 0);
      return Maybe<void>::Ok();
    })
    .SetParallelDistributionInferFn([](user_op::InferParallelDistributionFnContext* ctx)
                                        -> Maybe<void> {
      cfg::ParallelDistribution* in_distribution =
          ctx->ParallelDistribution4ArgNameAndIndex("in", 0);
      cfg::ParallelDistribution* out_distribution =
          ctx->ParallelDistribution4ArgNameAndIndex("out", 0);
      in_distribution->clear_sbp_parallel();
      out_distribution->clear_sbp_parallel();

      ParseParallelDistributionFromStringVec(
          ctx->user_op_conf().attr<std::vector<std::string>>("in_distribution"), in_distribution);
      ParseParallelDistributionFromStringVec(
          ctx->user_op_conf().attr<std::vector<std::string>>("out_distribution"), out_distribution);
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("_nccl_logical_2D_same_dim0_reduce_scatter")
    .Input("in")
    .Output("out")
    .Attr<std::vector<std::string>>("in_distribution")
    .Attr<std::vector<std::string>>("out_distribution")
    .SetLogicalTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputShape("out", 0) = ctx->InputShape("in", 0);
      *ctx->OutputIsDynamic("out", 0) = ctx->InputIsDynamic("in", 0);
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("out", 0) = ctx->InputDType("in", 0);
      return Maybe<void>::Ok();
    })
    .SetParallelDistributionInferFn([](user_op::InferParallelDistributionFnContext* ctx)
                                        -> Maybe<void> {
      cfg::ParallelDistribution* in_distribution =
          ctx->ParallelDistribution4ArgNameAndIndex("in", 0);
      cfg::ParallelDistribution* out_distribution =
          ctx->ParallelDistribution4ArgNameAndIndex("out", 0);
      in_distribution->clear_sbp_parallel();
      out_distribution->clear_sbp_parallel();

      ParseParallelDistributionFromStringVec(
          ctx->user_op_conf().attr<std::vector<std::string>>("in_distribution"), in_distribution);
      ParseParallelDistributionFromStringVec(
          ctx->user_op_conf().attr<std::vector<std::string>>("out_distribution"), out_distribution);

      return Maybe<void>::Ok();
    });

}  // namespace oneflow
