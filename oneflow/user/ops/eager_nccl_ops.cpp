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
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/common/decorator.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/user/ops/comm_net_device_infer_util.h"

namespace oneflow {

REGISTER_NO_GRAD_USER_OP("eager_nccl_all_reduce")
    .Input("in")
    .Output("out")
    .Attr<std::string>("parallel_conf")
    .Attr<bool>("async_launch", false)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputShape("out", 0) = ctx->InputShape("in", 0);
      return Maybe<void>::Ok();
    })
    .SetDeviceInferFn(DeviceInferFn<&IsAsyncLaunched>)
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder()
          .PartialSum(user_op::OpArg("in", 0))
          .Broadcast(user_op::OpArg("out", 0))
          .Build();
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("out", 0) = ctx->InputDType("in", 0);
      return Maybe<void>::Ok();
    });

REGISTER_USER_OP("eager_nccl_broadcast")
    .Input("in")
    .Output("out")
    .Attr<std::string>("parallel_conf")
    .Attr<int64_t>("root", 0)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputShape("out", 0) = ctx->InputShape("in", 0);
      return Maybe<void>::Ok();
    })
    .SetDeviceInferFn(DeviceInferFn<&SyncLaunched>)
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      ctx->NewBuilder()
          .PartialSum(user_op::OpArg("in", 0))
          .Broadcast(user_op::OpArg("out", 0))
          .Build();
      ctx->NewBuilder()
          .Broadcast(user_op::OpArg("in", 0))
          .Broadcast(user_op::OpArg("out", 0))
          .Build();
      ctx->NewBuilder()
          .Split(user_op::OpArg("in", 0), 0)
          .Broadcast(user_op::OpArg("out", 0))
          .Build();
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("out", 0) = ctx->InputDType("in", 0);
      return Maybe<void>::Ok();
    });

REGISTER_NO_GRAD_USER_OP("eager_nccl_reduce")
    .Input("in")
    .Output("out")
    .Attr<std::string>("parallel_conf")
    .Attr<int64_t>("root", 0)
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputShape("out", 0) = ctx->InputShape("in", 0);
      return Maybe<void>::Ok();
    })
    .SetDeviceInferFn(DeviceInferFn<&SyncLaunched>)
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      UNIMPLEMENTED_THEN_RETURN() << "consistent tensor are not supported";
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("out", 0) = ctx->InputDType("in", 0);
      return Maybe<void>::Ok();
    });

REGISTER_NO_GRAD_USER_OP("eager_nccl_reduce_scatter")
    .Input("in")
    .Output("out")
    .Attr<std::string>("parallel_conf")
    .Attr<std::string>("op_type", "sum")
    .SetLogicalTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputShape("out", 0) = ctx->InputShape("in", 0);
      return Maybe<void>::Ok();
    })
    .SetPhysicalTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* out_shape = ctx->OutputShape("out", 0);
      const Shape& shape = ctx->InputShape("in", 0);
      DimVector dim_vec;
      if (shape.NumAxes() > 0) {
        dim_vec.insert(dim_vec.end(), shape.dim_vec().cbegin(), shape.dim_vec().cend());
      }
      const cfg::SbpParallel& out_sbp_para = ctx->SbpParallel4ArgNameAndIndex("out", 0);
      const int64_t& parallel_num = ctx->parallel_ctx().parallel_num();
      if (parallel_num > 1) {
        const int64_t& split_axis = out_sbp_para.split_parallel().axis();
        CHECK_LT_OR_RETURN(split_axis, dim_vec.size());
        BalancedSplitter bs(shape.At(split_axis), parallel_num);
        dim_vec[split_axis] = bs.At(ctx->parallel_ctx().parallel_id()).size();
      }
      *out_shape = Shape(dim_vec);
      return Maybe<void>::Ok();
    })
    .SetDeviceInferFn(DeviceInferFn<&SyncLaunched>)
    .SetNdSbpInferFn([](user_op::InferNdSbpFnContext* ctx) -> Maybe<void> {
      const cfg::NdSbp& in_dis_hint = ctx->NdSbpHint4InputArgNameAndIndex("in", 0);
      cfg::NdSbp* in_nd_sbp = ctx->NdSbp4ArgNameAndIndex("in", 0);
      cfg::NdSbp* out_nd_sbp = ctx->NdSbp4ArgNameAndIndex("out", 0);
      CHECK_GE_OR_RETURN(in_dis_hint.sbp_parallel_size(), 1);
      for (const auto& sbp_hint : in_dis_hint.sbp_parallel()) {
        CHECK_OR_RETURN(sbp_hint.has_partial_sum_parallel() || sbp_hint.has_broadcast_parallel());
      }
      in_nd_sbp->clear_sbp_parallel();
      out_nd_sbp->clear_sbp_parallel();

      // P2S or B2S
      const Shape& parallel_hierarchy = ctx->parallel_hierarchy();
      CHECK_GE_OR_RETURN(parallel_hierarchy.NumAxes(), 1);
      in_nd_sbp->CopyFrom(in_dis_hint);
      for (int32_t i = 0; i < parallel_hierarchy.NumAxes(); ++i) {
        out_nd_sbp->add_sbp_parallel()->mutable_split_parallel()->set_axis(0);
      }
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn(user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast)
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("out", 0) = ctx->InputDType("in", 0);
      return Maybe<void>::Ok();
    });

REGISTER_NO_GRAD_USER_OP("eager_nccl_all_gather")
    .Input("in")
    .Output("out")
    .Attr<std::string>("parallel_conf")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
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
      cfg::NdSbp* in_nd_sbp = ctx->NdSbp4ArgNameAndIndex("in", 0);
      cfg::NdSbp* out_nd_sbp = ctx->NdSbp4ArgNameAndIndex("out", 0);
      CHECK_GE_OR_RETURN(in_dis_hint.sbp_parallel_size(), 1);
      for (const auto& sbp_hint : in_dis_hint.sbp_parallel()) {
        CHECK_OR_RETURN(sbp_hint.has_split_parallel());
        CHECK_EQ_OR_RETURN(sbp_hint.split_parallel().axis(), 0);
      }

      in_nd_sbp->clear_sbp_parallel();
      out_nd_sbp->clear_sbp_parallel();

      // S(0)->B
      const Shape& parallel_hierarchy = ctx->parallel_hierarchy();
      CHECK_GE_OR_RETURN(parallel_hierarchy.NumAxes(), 1);
      for (int32_t i = 0; i < parallel_hierarchy.NumAxes(); ++i) {
        in_nd_sbp->add_sbp_parallel()->mutable_split_parallel()->set_axis(0);
        out_nd_sbp->add_sbp_parallel()->mutable_broadcast_parallel();
      }
      return Maybe<void>::Ok();
    })
    .SetDeviceInferFn(DeviceInferFn<&SyncLaunched>)
    .SetGetSbpFn(user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast);

REGISTER_NO_GRAD_USER_OP("eager_nccl_s2s")
    .Input("in")
    .Output("out")
    .Attr<int64_t>("in_split_axis", -1)
    .Attr<int64_t>("out_split_axis", -1)
    .Attr<std::string>("parallel_conf")
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
      cfg::NdSbp* in_nd_sbp = ctx->NdSbp4ArgNameAndIndex("in", 0);
      cfg::NdSbp* out_nd_sbp = ctx->NdSbp4ArgNameAndIndex("out", 0);
      CHECK_GE_OR_RETURN(in_dis_hint.sbp_parallel_size(), 1);
      for (const auto& sbp_hint : in_dis_hint.sbp_parallel()) {
        CHECK_OR_RETURN(sbp_hint.has_split_parallel());
        CHECK_EQ_OR_RETURN(sbp_hint.split_parallel().axis(), in_split_axis);
      }

      in_nd_sbp->clear_sbp_parallel();
      out_nd_sbp->clear_sbp_parallel();

      // S(in)->S(out)
      const Shape& parallel_hierarchy = ctx->parallel_hierarchy();
      CHECK_GE_OR_RETURN(parallel_hierarchy.NumAxes(), 1);
      for (int32_t i = 0; i < parallel_hierarchy.NumAxes(); ++i) {
        in_nd_sbp->add_sbp_parallel()->mutable_split_parallel()->set_axis(in_split_axis);
        out_nd_sbp->add_sbp_parallel()->mutable_split_parallel()->set_axis(out_split_axis);
      }
      return Maybe<void>::Ok();
    })
    .SetDeviceInferFn(DeviceInferFn<&SyncLaunched>)
    .SetGetSbpFn(user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast);
}  // namespace oneflow
