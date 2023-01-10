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
#include "oneflow/core/common/decorator.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/user/ops/comm_net_device_infer_util.h"
#include "oneflow/core/framework/op_generated.h"

namespace oneflow {

// Can only be called in local TODO: move this comment to ods
/* static */ Maybe<void> EagerBToSOp::InferLogicalTensorDesc(user_op::InferContext* ctx) {
  const Shape& shape = ctx->Attr<Shape>("shape");
  const std::string& out_parallel_conf_txt = ctx->Attr<std::string>("out_parallel_conf");
  const int64_t out_split_axis = ctx->Attr<int64_t>("out_split_axis");
  Symbol<ParallelDesc> out_parallel_desc = JUST(TxtStringToPlacement(out_parallel_conf_txt));
  DimVector dim_vec{shape.dim_vec()};
  int64_t out_parallel_num = out_parallel_desc->parallel_num();
  if (out_parallel_num > 1) {
    CHECK_LT_OR_RETURN(out_split_axis, shape.NumAxes());
    BalancedSplitter bs(shape.At(out_split_axis), out_parallel_num);
    const auto& opt_parallel_id = JUST(GetParallelId4CurrentProcessCtx(out_parallel_desc));
    int64_t parallel_id = opt_parallel_id->value_or(0);
    dim_vec[out_split_axis] = bs.At(parallel_id).size();
  }
  ctx->SetOutputShape("out", 0, Shape(dim_vec));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> EagerBToSOp::InferPhysicalTensorDesc(user_op::InferContext* ctx) {
  return InferLogicalTensorDesc(ctx);
}

/* static */ Maybe<void> EagerBToSOp::GetSbp(user_op::SbpContext* ctx) {
  return Error::TypeError() << "eager_b_to_s op doesn't support global tensor!";
}

/* static */ Maybe<void> EagerBToSOp::InferNdSbp(user_op::InferNdSbpFnContext* ctx) {
  return Error::TypeError() << "eager_b_to_s op doesn't support global tensor!";
}

/* static */ Maybe<void> EagerBToSOp::InferDataType(user_op::InferContext* ctx) {
  ctx->SetOutputDType("out", 0, ctx->InputDType("in", 0));
  return Maybe<void>::Ok();
}

/* static */ Maybe<Symbol<Stream>> EagerBToSOp::InferDeviceAndStream(
    user_op::DeviceAndStreamInferContext* ctx) {
  return DeviceAndStreamInferFn(ctx);
}

}  // namespace oneflow
