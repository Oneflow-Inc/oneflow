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
#include "oneflow/user/ops/comm_net_device_infer_util.h"

namespace oneflow {

REGISTER_NO_GRAD_USER_OP("_nccl_logical_all_reduce")
    .Input("in")
    .Output("out")
    .SetLogicalTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputShape("out", 0) = ctx->InputShape("in", 0);
      *ctx->OutputIsDynamic("out", 0) = ctx->InputIsDynamic("in", 0);
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("out", 0) = ctx->InputDType("in", 0);
      return Maybe<void>::Ok();
    })
    .SetNdSbpInferFn([](user_op::InferNdSbpFnContext* ctx) -> Maybe<void> {
      const cfg::NdSbp& in_dis_hint = ctx->NdSbpHint4InputArgNameAndIndex("in", 0);
      cfg::NdSbp* in_distribution = ctx->NdSbp4ArgNameAndIndex("in", 0);
      cfg::NdSbp* out_distribution = ctx->NdSbp4ArgNameAndIndex("out", 0);
      CHECK_GE_OR_RETURN(in_dis_hint.sbp_parallel_size(), 1);
      for (const auto& sbp_hint : in_dis_hint.sbp_parallel()) {
        CHECK_OR_RETURN(sbp_hint.has_partial_sum_parallel());
      }

      in_distribution->clear_sbp_parallel();
      out_distribution->clear_sbp_parallel();

      // P2B
      const Shape& parallel_hierarchy = ctx->parallel_hierarchy();
      CHECK_GE_OR_RETURN(parallel_hierarchy.NumAxes(), 1);
      for (int32_t i = 0; i < parallel_hierarchy.NumAxes(); ++i) {
        in_distribution->add_sbp_parallel()->mutable_partial_sum_parallel();
        out_distribution->add_sbp_parallel()->mutable_broadcast_parallel();
      }
      return Maybe<void>::Ok();
    })
    .SetDeviceInferFn(DeviceInferFn<&SyncLaunched>)
    .SetGetSbpFn(user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast);

REGISTER_NO_GRAD_USER_OP("_nccl_logical_reduce_scatter")
    .Input("in")
    .Output("out")
    .SetLogicalTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputShape("out", 0) = ctx->InputShape("in", 0);
      *ctx->OutputIsDynamic("out", 0) = ctx->InputIsDynamic("in", 0);
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("out", 0) = ctx->InputDType("in", 0);
      return Maybe<void>::Ok();
    })
    .SetNdSbpInferFn([](user_op::InferNdSbpFnContext* ctx) -> Maybe<void> {
      const cfg::NdSbp& in_dis_hint = ctx->NdSbpHint4InputArgNameAndIndex("in", 0);
      cfg::NdSbp* in_distribution = ctx->NdSbp4ArgNameAndIndex("in", 0);
      cfg::NdSbp* out_distribution = ctx->NdSbp4ArgNameAndIndex("out", 0);
      CHECK_GE_OR_RETURN(in_dis_hint.sbp_parallel_size(), 1);
      for (const auto& sbp_hint : in_dis_hint.sbp_parallel()) {
        CHECK_OR_RETURN(sbp_hint.has_partial_sum_parallel());
      }

      in_distribution->clear_sbp_parallel();
      out_distribution->clear_sbp_parallel();

      // P2S
      const Shape& parallel_hierarchy = ctx->parallel_hierarchy();
      CHECK_GE_OR_RETURN(parallel_hierarchy.NumAxes(), 1);
      for (int32_t i = 0; i < parallel_hierarchy.NumAxes(); ++i) {
        in_distribution->add_sbp_parallel()->mutable_partial_sum_parallel();
        out_distribution->add_sbp_parallel()->mutable_split_parallel()->set_axis(0);
      }
      return Maybe<void>::Ok();
    })
    .SetDeviceInferFn(DeviceInferFn<&SyncLaunched>)
    .SetGetSbpFn(user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast);

REGISTER_NO_GRAD_USER_OP("_nccl_logical_all_gather")
    .Input("in")
    .Output("out")
    .SetLogicalTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputShape("out", 0) = ctx->InputShape("in", 0);
      *ctx->OutputIsDynamic("out", 0) = ctx->InputIsDynamic("in", 0);
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("out", 0) = ctx->InputDType("in", 0);
      return Maybe<void>::Ok();
    })
    .SetNdSbpInferFn([](user_op::InferNdSbpFnContext* ctx) -> Maybe<void> {
      const cfg::NdSbp& in_dis_hint = ctx->NdSbpHint4InputArgNameAndIndex("in", 0);
      cfg::NdSbp* in_distribution = ctx->NdSbp4ArgNameAndIndex("in", 0);
      cfg::NdSbp* out_distribution = ctx->NdSbp4ArgNameAndIndex("out", 0);
      CHECK_GE_OR_RETURN(in_dis_hint.sbp_parallel_size(), 1);
      for (const auto& sbp_hint : in_dis_hint.sbp_parallel()) {
        CHECK_OR_RETURN(sbp_hint.has_split_parallel());
        CHECK_EQ_OR_RETURN(sbp_hint.split_parallel().axis(), 0);
      }

      in_distribution->clear_sbp_parallel();
      out_distribution->clear_sbp_parallel();

      // S(0)->B
      const Shape& parallel_hierarchy = ctx->parallel_hierarchy();
      CHECK_GE_OR_RETURN(parallel_hierarchy.NumAxes(), 1);
      for (int32_t i = 0; i < parallel_hierarchy.NumAxes(); ++i) {
        in_distribution->add_sbp_parallel()->mutable_split_parallel()->set_axis(0);
        out_distribution->add_sbp_parallel()->mutable_broadcast_parallel();
      }
      return Maybe<void>::Ok();
    })
    .SetDeviceInferFn(DeviceInferFn<&SyncLaunched>)
    .SetGetSbpFn(user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast);

REGISTER_NO_GRAD_USER_OP("_nccl_logical_all_gather_noncontinuous")
    .Input("in")
    .Output("out")
    .Attr<int64_t>("in_split_axis", -1)
    .SetLogicalTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputShape("out", 0) = ctx->InputShape("in", 0);
      *ctx->OutputIsDynamic("out", 0) = ctx->InputIsDynamic("in", 0);
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("out", 0) = ctx->InputDType("in", 0);
      return Maybe<void>::Ok();
    })
    .SetNdSbpInferFn([](user_op::InferNdSbpFnContext* ctx) -> Maybe<void> {
      const cfg::NdSbp& in_dis_hint = ctx->NdSbpHint4InputArgNameAndIndex("in", 0);
      CHECK_GE_OR_RETURN(in_dis_hint.sbp_parallel_size(), 1);
      const int64_t in_split_axis = ctx->user_op_conf().attr<int64_t>("in_split_axis");
      CHECK_GE_OR_RETURN(in_split_axis, 1);
      for (const auto& sbp_hint : in_dis_hint.sbp_parallel()) {
        CHECK_OR_RETURN(sbp_hint.has_split_parallel());
        CHECK_EQ_OR_RETURN(sbp_hint.split_parallel().axis(), in_split_axis);
      }

      cfg::NdSbp* in_distribution = ctx->NdSbp4ArgNameAndIndex("in", 0);
      cfg::NdSbp* out_distribution = ctx->NdSbp4ArgNameAndIndex("out", 0);
      in_distribution->clear_sbp_parallel();
      out_distribution->clear_sbp_parallel();

      // S(1)->(B)
      const Shape& parallel_hierarchy = ctx->parallel_hierarchy();
      CHECK_GE_OR_RETURN(parallel_hierarchy.NumAxes(), 1);
      for (int32_t i = 0; i < parallel_hierarchy.NumAxes(); ++i) {
        in_distribution->add_sbp_parallel()->mutable_split_parallel()->set_axis(in_split_axis);
        out_distribution->add_sbp_parallel()->mutable_broadcast_parallel();
      }
      return Maybe<void>::Ok();
    })
    .SetDeviceInferFn(DeviceInferFn<&SyncLaunched>)
    .SetGetSbpFn(user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast);

REGISTER_NO_GRAD_USER_OP("_nccl_logical_s2s")
    .Input("in")
    .Output("out")
    .Attr<int64_t>("in_split_axis", -1)
    .Attr<int64_t>("out_split_axis", -1)
    .SetLogicalTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputShape("out", 0) = ctx->InputShape("in", 0);
      *ctx->OutputIsDynamic("out", 0) = ctx->InputIsDynamic("in", 0);
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("out", 0) = ctx->InputDType("in", 0);
      return Maybe<void>::Ok();
    })
    .SetNdSbpInferFn([](user_op::InferNdSbpFnContext* ctx) -> Maybe<void> {
      const int64_t in_split_axis = ctx->user_op_conf().attr<int64_t>("in_split_axis");
      const int64_t out_split_axis = ctx->user_op_conf().attr<int64_t>("out_split_axis");
      const cfg::NdSbp& in_dis_hint = ctx->NdSbpHint4InputArgNameAndIndex("in", 0);
      cfg::NdSbp* in_distribution = ctx->NdSbp4ArgNameAndIndex("in", 0);
      cfg::NdSbp* out_distribution = ctx->NdSbp4ArgNameAndIndex("out", 0);
      CHECK_GE_OR_RETURN(in_dis_hint.sbp_parallel_size(), 1);
      for (const auto& sbp_hint : in_dis_hint.sbp_parallel()) {
        CHECK_OR_RETURN(sbp_hint.has_split_parallel());
        CHECK_EQ_OR_RETURN(sbp_hint.split_parallel().axis(), in_split_axis);
      }

      in_distribution->clear_sbp_parallel();
      out_distribution->clear_sbp_parallel();

      // S(in)->S(out)
      const Shape& parallel_hierarchy = ctx->parallel_hierarchy();
      CHECK_GE_OR_RETURN(parallel_hierarchy.NumAxes(), 1);
      for (int32_t i = 0; i < parallel_hierarchy.NumAxes(); ++i) {
        in_distribution->add_sbp_parallel()->mutable_split_parallel()->set_axis(in_split_axis);
        out_distribution->add_sbp_parallel()->mutable_split_parallel()->set_axis(out_split_axis);
      }
      return Maybe<void>::Ok();
    })
    .SetDeviceInferFn(DeviceInferFn<&SyncLaunched>)
    .SetGetSbpFn(user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast);

}  // namespace oneflow
