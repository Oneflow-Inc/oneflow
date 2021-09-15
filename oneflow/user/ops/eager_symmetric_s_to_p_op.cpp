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
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/user/ops/comm_net_device_infer_util.h"

namespace oneflow {

REGISTER_NO_GRAD_USER_OP("eager_symmetric_s_to_p")
    .Input("in")
    .Output("out")
    .Attr<int64_t>("in_split_axis", -1)
    .Attr<std::string>("parallel_conf")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputShape("out", 0) = ctx->InputShape("in", 0);
      return Maybe<void>::Ok();
    })
    .SetDeviceInferFn(DeviceInferFn<&SyncLaunched>)
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& in = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
      FOR_RANGE(int64_t, i, 0, in.shape().NumAxes()) {
        ctx->NewBuilder()
            .Split(user_op::OpArg("in", 0), i)
            .PartialSum(user_op::OpArg("out", 0))
            .Build();
      }
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("out", 0) = ctx->InputDType("in", 0);
      return Maybe<void>::Ok();
    })
    .SetNdSbpInferFn([](user_op::InferNdSbpFnContext* ctx) -> Maybe<void> {
      const int64_t in_split_axis = ctx->user_op_conf().attr<int64_t>("in_split_axis");
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

      const Shape& parallel_hierarchy = ctx->parallel_hierarchy();
      CHECK_GE_OR_RETURN(parallel_hierarchy.NumAxes(), 1);
      for (int32_t i = 0; i < parallel_hierarchy.NumAxes(); ++i) {
        in_nd_sbp->add_sbp_parallel()->mutable_split_parallel()->set_axis(in_split_axis);
        out_nd_sbp->add_sbp_parallel()->mutable_partial_sum_parallel();
      }
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
