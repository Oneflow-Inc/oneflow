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
#include "oneflow/core/framework/op_generated.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/user/ops/nccl_logical_util.h"
#include "oneflow/user/ops/comm_net_device_infer_util.h"

namespace oneflow {

/* static */ Maybe<void> _ncclLogicalFusionOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  int32_t nccl_size = ctx->input_size("in");
  CHECK_EQ_OR_RETURN(nccl_size, ctx->output_size("out"));
  for (int32_t i = 0; i < nccl_size; ++i) {
    ctx->SetOutputShape("out", i, ctx->InputShape("in", i));
    ctx->SetOutputIsDynamic("out", i, ctx->InputIsDynamic("in", i));
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<void> _ncclLogicalFusionOp::GetSbp(user_op::SbpContext* ctx) {
  return user_op::GetSbpFnUtil::DefaultBroadcastToBroadcast(ctx);
}

/* static */ Maybe<void> _ncclLogicalFusionOp::InferNdSbp(user_op::InferNdSbpFnContext* ctx) {
  int32_t nccl_size = ctx->input_size("in");
  CHECK_EQ_OR_RETURN(nccl_size, ctx->output_size("out"));
  const std::vector<std::string>& src_nd_sbp_str_list =
      ctx->Attr<std::vector<std::string>>("src_nd_sbp_str_list");
  const std::vector<std::string>& dst_nd_sbp_str_list =
      ctx->Attr<std::vector<std::string>>("dst_nd_sbp_str_list");
  const Shape& hierarchy = ctx->Attr<Shape>("hierarchy");
  for (int32_t i = 0; i < nccl_size; ++i) {
    NdSbp* input_nd_sbp = ctx->NdSbp4ArgNameAndIndex("in", i);
    NdSbp* output_nd_sbp = ctx->NdSbp4ArgNameAndIndex("out", i);
    input_nd_sbp->clear_sbp_parallel();
    output_nd_sbp->clear_sbp_parallel();
    CHECK_OR_RETURN(ParseNdSbpFromLongString(src_nd_sbp_str_list.at(i), input_nd_sbp))
        << Error::RuntimeError() << " Cannot parse str: " << src_nd_sbp_str_list.at(i)
        << " to input nd_sbp attr of op : " << ctx->op_name();
    CHECK_OR_RETURN(ParseNdSbpFromLongString(dst_nd_sbp_str_list.at(i), output_nd_sbp))
        << Error::RuntimeError() << " Cannot parse str: " << dst_nd_sbp_str_list.at(i)
        << " to output nd_sbp attr of op : " << ctx->op_name();
    CHECK_EQ_OR_RETURN(input_nd_sbp->sbp_parallel_size(), hierarchy->NumAxes());
  }

  return Maybe<void>::Ok();
}

/* static */ Maybe<void> _ncclLogicalFusionOp::InferDataType(user_op::InferContext* ctx) {
  int32_t nccl_size = ctx->input_size("in");
  CHECK_EQ_OR_RETURN(nccl_size, ctx->output_size("out"));
  for (int32_t i = 0; i < nccl_size; ++i) {
    ctx->SetOutputDType("out", i, ctx->InputDType("in", i));
  }
  return Maybe<void>::Ok();
}

/* static */ Maybe<Symbol<Stream>> _ncclLogicalFusionOp::InferDeviceAndStream(
    user_op::DeviceAndStreamInferContext* ctx) {
  return DeviceAndStreamInferFn(ctx);
}

}  // namespace oneflow
