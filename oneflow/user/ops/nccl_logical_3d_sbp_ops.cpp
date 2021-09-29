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

REGISTER_USER_OP("_nccl_logical_3D_change_dim0_all_reduce")
    .Input("in")
    .Output("out")
    .SetLogicalTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputShape("out", 0) = ctx->InputShape("in", 0);
      *ctx->IsDynamic4ArgNameAndIndex("out", 0) = *ctx->IsDynamic4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->Dtype4ArgNameAndIndex("out", 0) = *ctx->Dtype4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetParallelDistributionInferFn(
        [](user_op::InferParallelDistributionFnContext* ctx) -> Maybe<void> {
          const cfg::ParallelDistribution& in_dis_hint =
              ctx->ParallelDistributionHint4InputArgNameAndIndex("in", 0);
          CHECK_EQ_OR_RETURN(in_dis_hint.sbp_parallel_size(), 3);
          CHECK_OR_RETURN(in_dis_hint.sbp_parallel(0).has_partial_sum_parallel());
          const Shape& parallel_hierarchy = ctx->parallel_hierarchy();
          CHECK_EQ_OR_RETURN(parallel_hierarchy.NumAxes(), 3);

          cfg::ParallelDistribution* in_distribution =
              ctx->ParallelDistribution4ArgNameAndIndex("in", 0);
          cfg::ParallelDistribution* out_distribution =
              ctx->ParallelDistribution4ArgNameAndIndex("out", 0);
          in_distribution->clear_sbp_parallel();
          out_distribution->clear_sbp_parallel();
          // in use hint
          in_distribution->CopyFrom(in_dis_hint);

          out_distribution->add_sbp_parallel()->mutable_broadcast_parallel();
          *out_distribution->add_sbp_parallel() = in_dis_hint.sbp_parallel(1);
          *out_distribution->add_sbp_parallel() = in_dis_hint.sbp_parallel(2);

          return Maybe<void>::Ok();
        });

REGISTER_USER_OP("_nccl_logical_3D_change_dim1_all_reduce")
    .Input("in")
    .Output("out")
    .SetLogicalTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputShape("out", 0) = ctx->InputShape("in", 0);
      *ctx->IsDynamic4ArgNameAndIndex("out", 0) = *ctx->IsDynamic4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->Dtype4ArgNameAndIndex("out", 0) = *ctx->Dtype4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetParallelDistributionInferFn(
        [](user_op::InferParallelDistributionFnContext* ctx) -> Maybe<void> {
          const cfg::ParallelDistribution& in_dis_hint =
              ctx->ParallelDistributionHint4InputArgNameAndIndex("in", 0);
          CHECK_EQ_OR_RETURN(in_dis_hint.sbp_parallel_size(), 3);
          CHECK_OR_RETURN(in_dis_hint.sbp_parallel(1).has_partial_sum_parallel());
          const Shape& parallel_hierarchy = ctx->parallel_hierarchy();
          CHECK_EQ_OR_RETURN(parallel_hierarchy.NumAxes(), 3);

          cfg::ParallelDistribution* in_distribution =
              ctx->ParallelDistribution4ArgNameAndIndex("in", 0);
          cfg::ParallelDistribution* out_distribution =
              ctx->ParallelDistribution4ArgNameAndIndex("out", 0);
          in_distribution->clear_sbp_parallel();
          out_distribution->clear_sbp_parallel();
          // in use hint
          in_distribution->CopyFrom(in_dis_hint);

          *out_distribution->add_sbp_parallel() = in_dis_hint.sbp_parallel(0);
          out_distribution->add_sbp_parallel()->mutable_broadcast_parallel();
          *out_distribution->add_sbp_parallel() = in_dis_hint.sbp_parallel(2);

          return Maybe<void>::Ok();
        });

REGISTER_USER_OP("_nccl_logical_3D_change_dim1_all_gather")
    .Input("in")
    .Output("out")
    .SetLogicalTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputShape("out", 0) = ctx->InputShape("in", 0);
      *ctx->IsDynamic4ArgNameAndIndex("out", 0) = *ctx->IsDynamic4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->Dtype4ArgNameAndIndex("out", 0) = *ctx->Dtype4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetParallelDistributionInferFn(
        [](user_op::InferParallelDistributionFnContext* ctx) -> Maybe<void> {
          const cfg::ParallelDistribution& in_dis_hint =
              ctx->ParallelDistributionHint4InputArgNameAndIndex("in", 0);
          CHECK_EQ_OR_RETURN(in_dis_hint.sbp_parallel_size(), 3);
          CHECK_OR_RETURN(in_dis_hint.sbp_parallel(1).has_split_parallel());
          CHECK_EQ_OR_RETURN(in_dis_hint.sbp_parallel(1).split_parallel().axis(), 0);
          const Shape& parallel_hierarchy = ctx->parallel_hierarchy();
          CHECK_EQ_OR_RETURN(parallel_hierarchy.NumAxes(), 3);

          cfg::ParallelDistribution* in_distribution =
              ctx->ParallelDistribution4ArgNameAndIndex("in", 0);
          cfg::ParallelDistribution* out_distribution =
              ctx->ParallelDistribution4ArgNameAndIndex("out", 0);
          in_distribution->clear_sbp_parallel();
          out_distribution->clear_sbp_parallel();
          // in use hint
          in_distribution->CopyFrom(in_dis_hint);
          *out_distribution->add_sbp_parallel() = in_dis_hint.sbp_parallel(0);
          out_distribution->add_sbp_parallel()->mutable_broadcast_parallel();
          *out_distribution->add_sbp_parallel() = in_dis_hint.sbp_parallel(2);

          return Maybe<void>::Ok();
        });

REGISTER_USER_OP("_nccl_logical_3D_change_dim1_all_gather_noncontinuous")
    .Input("in")
    .Output("out")
    .Attr<int64_t>("in_dim1_split_axis", -1)
    .SetLogicalTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputShape("out", 0) = ctx->InputShape("in", 0);
      *ctx->IsDynamic4ArgNameAndIndex("out", 0) = *ctx->IsDynamic4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->Dtype4ArgNameAndIndex("out", 0) = *ctx->Dtype4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetParallelDistributionInferFn(
        [](user_op::InferParallelDistributionFnContext* ctx) -> Maybe<void> {
          const cfg::ParallelDistribution& in_dis_hint =
              ctx->ParallelDistributionHint4InputArgNameAndIndex("in", 0);
          CHECK_EQ_OR_RETURN(in_dis_hint.sbp_parallel_size(), 3);
          CHECK_OR_RETURN(in_dis_hint.sbp_parallel(1).has_split_parallel());
          CHECK_GT_OR_RETURN(in_dis_hint.sbp_parallel(1).split_parallel().axis(), 0);
          const Shape& parallel_hierarchy = ctx->parallel_hierarchy();
          CHECK_EQ_OR_RETURN(parallel_hierarchy.NumAxes(), 3);

          cfg::ParallelDistribution* in_distribution =
              ctx->ParallelDistribution4ArgNameAndIndex("in", 0);
          cfg::ParallelDistribution* out_distribution =
              ctx->ParallelDistribution4ArgNameAndIndex("out", 0);
          in_distribution->clear_sbp_parallel();
          out_distribution->clear_sbp_parallel();
          // in use hint
          in_distribution->CopyFrom(in_dis_hint);
          *out_distribution->add_sbp_parallel() = in_dis_hint.sbp_parallel(0);
          out_distribution->add_sbp_parallel()->mutable_broadcast_parallel();
          *out_distribution->add_sbp_parallel() = in_dis_hint.sbp_parallel(2);

          return Maybe<void>::Ok();
        });

REGISTER_USER_OP("_nccl_logical_3D_change_dim1_reduce_scatter")
    .Input("in")
    .Output("out")
    .SetLogicalTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputShape("out", 0) = ctx->InputShape("in", 0);
      *ctx->IsDynamic4ArgNameAndIndex("out", 0) = *ctx->IsDynamic4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->Dtype4ArgNameAndIndex("out", 0) = *ctx->Dtype4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetParallelDistributionInferFn(
        [](user_op::InferParallelDistributionFnContext* ctx) -> Maybe<void> {
          const cfg::ParallelDistribution& in_dis_hint =
              ctx->ParallelDistributionHint4InputArgNameAndIndex("in", 0);
          CHECK_EQ_OR_RETURN(in_dis_hint.sbp_parallel_size(), 3);
          CHECK_OR_RETURN(in_dis_hint.sbp_parallel(1).has_partial_sum_parallel());
          const Shape& parallel_hierarchy = ctx->parallel_hierarchy();
          CHECK_EQ_OR_RETURN(parallel_hierarchy.NumAxes(), 3);

          cfg::ParallelDistribution* in_distribution =
              ctx->ParallelDistribution4ArgNameAndIndex("in", 0);
          cfg::ParallelDistribution* out_distribution =
              ctx->ParallelDistribution4ArgNameAndIndex("out", 0);
          in_distribution->clear_sbp_parallel();
          out_distribution->clear_sbp_parallel();
          // in use hint
          in_distribution->CopyFrom(in_dis_hint);
          *out_distribution->add_sbp_parallel() = in_dis_hint.sbp_parallel(0);
          out_distribution->add_sbp_parallel()->mutable_split_parallel()->set_axis(0);
          *out_distribution->add_sbp_parallel() = in_dis_hint.sbp_parallel(2);

          return Maybe<void>::Ok();
        });

REGISTER_USER_OP("_nccl_logical_3D_change_dim1_reduce_scatter_noncontinuous")
    .Input("in")
    .Output("out")
    .Attr<int64_t>("out_dim1_split_axis", -1)
    .SetLogicalTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputShape("out", 0) = ctx->InputShape("in", 0);
      *ctx->IsDynamic4ArgNameAndIndex("out", 0) = *ctx->IsDynamic4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->Dtype4ArgNameAndIndex("out", 0) = *ctx->Dtype4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetParallelDistributionInferFn(
        [](user_op::InferParallelDistributionFnContext* ctx) -> Maybe<void> {
          const cfg::ParallelDistribution& in_dis_hint =
              ctx->ParallelDistributionHint4InputArgNameAndIndex("in", 0);
          CHECK_EQ_OR_RETURN(in_dis_hint.sbp_parallel_size(), 3);
          CHECK_OR_RETURN(in_dis_hint.sbp_parallel(1).has_partial_sum_parallel());
          const Shape& parallel_hierarchy = ctx->parallel_hierarchy();
          CHECK_EQ_OR_RETURN(parallel_hierarchy.NumAxes(), 3);

          cfg::ParallelDistribution* in_distribution =
              ctx->ParallelDistribution4ArgNameAndIndex("in", 0);
          cfg::ParallelDistribution* out_distribution =
              ctx->ParallelDistribution4ArgNameAndIndex("out", 0);
          in_distribution->clear_sbp_parallel();
          out_distribution->clear_sbp_parallel();
          // in use hint
          in_distribution->CopyFrom(in_dis_hint);
          *out_distribution->add_sbp_parallel() = in_dis_hint.sbp_parallel(0);
          out_distribution->add_sbp_parallel()->mutable_split_parallel()->set_axis(
              ctx->user_op_conf().attr<int64_t>("out_dim1_split_axis"));
          *out_distribution->add_sbp_parallel() = in_dis_hint.sbp_parallel(2);

          return Maybe<void>::Ok();
        });

REGISTER_USER_OP("_nccl_logical_3D_change_dim2_all_reduce")
    .Input("in")
    .Output("out")
    .SetLogicalTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputShape("out", 0) = ctx->InputShape("in", 0);
      *ctx->IsDynamic4ArgNameAndIndex("out", 0) = *ctx->IsDynamic4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->Dtype4ArgNameAndIndex("out", 0) = *ctx->Dtype4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetParallelDistributionInferFn(
        [](user_op::InferParallelDistributionFnContext* ctx) -> Maybe<void> {
          const cfg::ParallelDistribution& in_dis_hint =
              ctx->ParallelDistributionHint4InputArgNameAndIndex("in", 0);
          CHECK_EQ_OR_RETURN(in_dis_hint.sbp_parallel_size(), 3);
          CHECK_OR_RETURN(in_dis_hint.sbp_parallel(2).has_partial_sum_parallel());
          const Shape& parallel_hierarchy = ctx->parallel_hierarchy();
          CHECK_EQ_OR_RETURN(parallel_hierarchy.NumAxes(), 3);

          cfg::ParallelDistribution* in_distribution =
              ctx->ParallelDistribution4ArgNameAndIndex("in", 0);
          cfg::ParallelDistribution* out_distribution =
              ctx->ParallelDistribution4ArgNameAndIndex("out", 0);
          in_distribution->clear_sbp_parallel();
          out_distribution->clear_sbp_parallel();
          // in use hint
          in_distribution->CopyFrom(in_dis_hint);
          *out_distribution->add_sbp_parallel() = in_dis_hint.sbp_parallel(0);
          *out_distribution->add_sbp_parallel() = in_dis_hint.sbp_parallel(1);
          out_distribution->add_sbp_parallel()->mutable_broadcast_parallel();

          return Maybe<void>::Ok();
        });

REGISTER_USER_OP("_nccl_logical_3D_change_dim2_all_gather")
    .Input("in")
    .Output("out")
    .SetLogicalTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputShape("out", 0) = ctx->InputShape("in", 0);
      *ctx->IsDynamic4ArgNameAndIndex("out", 0) = *ctx->IsDynamic4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->Dtype4ArgNameAndIndex("out", 0) = *ctx->Dtype4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetParallelDistributionInferFn(
        [](user_op::InferParallelDistributionFnContext* ctx) -> Maybe<void> {
          const cfg::ParallelDistribution& in_dis_hint =
              ctx->ParallelDistributionHint4InputArgNameAndIndex("in", 0);
          CHECK_EQ_OR_RETURN(in_dis_hint.sbp_parallel_size(), 3);
          CHECK_OR_RETURN(in_dis_hint.sbp_parallel(2).has_split_parallel());
          CHECK_EQ_OR_RETURN(in_dis_hint.sbp_parallel(2).split_parallel().axis(), 0);
          const Shape& parallel_hierarchy = ctx->parallel_hierarchy();
          CHECK_EQ_OR_RETURN(parallel_hierarchy.NumAxes(), 3);

          cfg::ParallelDistribution* in_distribution =
              ctx->ParallelDistribution4ArgNameAndIndex("in", 0);
          cfg::ParallelDistribution* out_distribution =
              ctx->ParallelDistribution4ArgNameAndIndex("out", 0);
          in_distribution->clear_sbp_parallel();
          out_distribution->clear_sbp_parallel();
          // in use hint
          in_distribution->CopyFrom(in_dis_hint);
          *out_distribution->add_sbp_parallel() = in_dis_hint.sbp_parallel(0);
          *out_distribution->add_sbp_parallel() = in_dis_hint.sbp_parallel(1);
          out_distribution->add_sbp_parallel()->mutable_broadcast_parallel();

          return Maybe<void>::Ok();
        });

REGISTER_USER_OP("_nccl_logical_3D_change_dim2_all_gather_noncontinuous")
    .Input("in")
    .Output("out")
    .Attr<int64_t>("in_dim2_split_axis", -1)
    .SetLogicalTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputShape("out", 0) = ctx->InputShape("in", 0);
      *ctx->IsDynamic4ArgNameAndIndex("out", 0) = *ctx->IsDynamic4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->Dtype4ArgNameAndIndex("out", 0) = *ctx->Dtype4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetParallelDistributionInferFn(
        [](user_op::InferParallelDistributionFnContext* ctx) -> Maybe<void> {
          const cfg::ParallelDistribution& in_dis_hint =
              ctx->ParallelDistributionHint4InputArgNameAndIndex("in", 0);
          CHECK_EQ_OR_RETURN(in_dis_hint.sbp_parallel_size(), 3);
          CHECK_OR_RETURN(in_dis_hint.sbp_parallel(2).has_split_parallel());
          CHECK_GT_OR_RETURN(in_dis_hint.sbp_parallel(2).split_parallel().axis(), 0);
          const Shape& parallel_hierarchy = ctx->parallel_hierarchy();
          CHECK_EQ_OR_RETURN(parallel_hierarchy.NumAxes(), 3);

          cfg::ParallelDistribution* in_distribution =
              ctx->ParallelDistribution4ArgNameAndIndex("in", 0);
          cfg::ParallelDistribution* out_distribution =
              ctx->ParallelDistribution4ArgNameAndIndex("out", 0);
          in_distribution->clear_sbp_parallel();
          out_distribution->clear_sbp_parallel();
          // in use hint
          in_distribution->CopyFrom(in_dis_hint);
          *out_distribution->add_sbp_parallel() = in_dis_hint.sbp_parallel(0);
          *out_distribution->add_sbp_parallel() = in_dis_hint.sbp_parallel(1);
          out_distribution->add_sbp_parallel()->mutable_broadcast_parallel();

          return Maybe<void>::Ok();
        });

REGISTER_USER_OP("_nccl_logical_3D_change_dim2_reduce_scatter")
    .Input("in")
    .Output("out")
    .SetLogicalTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputShape("out", 0) = ctx->InputShape("in", 0);
      *ctx->IsDynamic4ArgNameAndIndex("out", 0) = *ctx->IsDynamic4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->Dtype4ArgNameAndIndex("out", 0) = *ctx->Dtype4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetParallelDistributionInferFn(
        [](user_op::InferParallelDistributionFnContext* ctx) -> Maybe<void> {
          const cfg::ParallelDistribution& in_dis_hint =
              ctx->ParallelDistributionHint4InputArgNameAndIndex("in", 0);
          CHECK_EQ_OR_RETURN(in_dis_hint.sbp_parallel_size(), 3);
          CHECK_OR_RETURN(in_dis_hint.sbp_parallel(2).has_partial_sum_parallel());
          const Shape& parallel_hierarchy = ctx->parallel_hierarchy();
          CHECK_EQ_OR_RETURN(parallel_hierarchy.NumAxes(), 3);

          cfg::ParallelDistribution* in_distribution =
              ctx->ParallelDistribution4ArgNameAndIndex("in", 0);
          cfg::ParallelDistribution* out_distribution =
              ctx->ParallelDistribution4ArgNameAndIndex("out", 0);
          in_distribution->clear_sbp_parallel();
          out_distribution->clear_sbp_parallel();
          // in use hint
          in_distribution->CopyFrom(in_dis_hint);
          *out_distribution->add_sbp_parallel() = in_dis_hint.sbp_parallel(0);
          *out_distribution->add_sbp_parallel() = in_dis_hint.sbp_parallel(1);
          out_distribution->add_sbp_parallel()->mutable_split_parallel()->set_axis(0);

          return Maybe<void>::Ok();
        });

REGISTER_USER_OP("_nccl_logical_3D_change_dim2_reduce_scatter_noncontinuous")
    .Input("in")
    .Output("out")
    .Attr<int64_t>("out_dim2_split_axis", -1)
    .SetLogicalTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputShape("out", 0) = ctx->InputShape("in", 0);
      *ctx->IsDynamic4ArgNameAndIndex("out", 0) = *ctx->IsDynamic4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->Dtype4ArgNameAndIndex("out", 0) = *ctx->Dtype4ArgNameAndIndex("in", 0);
      return Maybe<void>::Ok();
    })
    .SetParallelDistributionInferFn(
        [](user_op::InferParallelDistributionFnContext* ctx) -> Maybe<void> {
          const cfg::ParallelDistribution& in_dis_hint =
              ctx->ParallelDistributionHint4InputArgNameAndIndex("in", 0);
          CHECK_EQ_OR_RETURN(in_dis_hint.sbp_parallel_size(), 3);
          CHECK_OR_RETURN(in_dis_hint.sbp_parallel(2).has_partial_sum_parallel());
          const Shape& parallel_hierarchy = ctx->parallel_hierarchy();
          CHECK_EQ_OR_RETURN(parallel_hierarchy.NumAxes(), 3);

          cfg::ParallelDistribution* in_distribution =
              ctx->ParallelDistribution4ArgNameAndIndex("in", 0);
          cfg::ParallelDistribution* out_distribution =
              ctx->ParallelDistribution4ArgNameAndIndex("out", 0);
          in_distribution->clear_sbp_parallel();
          out_distribution->clear_sbp_parallel();
          // in use hint
          in_distribution->CopyFrom(in_dis_hint);
          *out_distribution->add_sbp_parallel() = in_dis_hint.sbp_parallel(0);
          *out_distribution->add_sbp_parallel() = in_dis_hint.sbp_parallel(1);
          out_distribution->add_sbp_parallel()->mutable_split_parallel()->set_axis(
              ctx->user_op_conf().attr<int64_t>("out_dim2_split_axis"));

          return Maybe<void>::Ok();
        });

REGISTER_USER_OP("_nccl_logical_3D_change_dim2_all2all")
    .Input("in")
    .Output("out")
    .Attr<int64_t>("in_dim2_split_axis", -1)
    .Attr<int64_t>("out_dim2_split_axis", -1)
    .SetLogicalTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputShape("out", 0) = ctx->InputShape("in", 0);
      *ctx->OutputIsDynamic("out", 0) = ctx->InputIsDynamic("in", 0);
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("out", 0) = ctx->InputDType("in", 0);
      return Maybe<void>::Ok();
    })
    .SetParallelDistributionInferFn(
        [](user_op::InferParallelDistributionFnContext* ctx) -> Maybe<void> {
          const cfg::ParallelDistribution& in_dis_hint =
              ctx->ParallelDistributionHint4InputArgNameAndIndex("in", 0);
          CHECK_GE_OR_RETURN(in_dis_hint.sbp_parallel_size(), 2);
          // (*, S(in_dim1_split_axis)) -> (*, S(out_dim1_split_axis))
          const int64_t in_split_axis = ctx->user_op_conf().attr<int64_t>("in_dim2_split_axis");
          const int64_t out_split_axis = ctx->user_op_conf().attr<int64_t>("out_dim2_split_axis");
          CHECK_OR_RETURN(in_dis_hint.sbp_parallel(2).has_split_parallel());
          CHECK_EQ_OR_RETURN(in_dis_hint.sbp_parallel(2).split_parallel().axis(), in_split_axis);
          const Shape& parallel_hierarchy = ctx->parallel_hierarchy();
          CHECK_EQ_OR_RETURN(parallel_hierarchy.NumAxes(), 3);

          cfg::ParallelDistribution* in_distribution =
              ctx->ParallelDistribution4ArgNameAndIndex("in", 0);
          cfg::ParallelDistribution* out_distribution =
              ctx->ParallelDistribution4ArgNameAndIndex("out", 0);
          in_distribution->clear_sbp_parallel();
          out_distribution->clear_sbp_parallel();
          in_distribution->CopyFrom(in_dis_hint);
          *out_distribution->add_sbp_parallel() = in_dis_hint.sbp_parallel(0);
          *out_distribution->add_sbp_parallel() = in_dis_hint.sbp_parallel(1);
          // out dim2 = Split(out_split_axis)
          out_distribution->add_sbp_parallel()->mutable_split_parallel()->set_axis(out_split_axis);
          return Maybe<void>::Ok();
        });

}  // namespace oneflow
