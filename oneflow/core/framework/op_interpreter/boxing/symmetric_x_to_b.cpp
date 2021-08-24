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
#include "oneflow/core/framework/id_util.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/boxing/eager_boxing_interpreter.h"
#include "oneflow/core/framework/op_interpreter/boxing/eager_boxing_interpreter_util.h"
#include "oneflow/core/common/decorator.h"

namespace oneflow {

namespace {

Maybe<void> RawCheckSymXToB(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out) {
  CHECK_EQ_OR_RETURN(in->nd_sbp()->sbp_parallel_size(), 1);
  CHECK_EQ_OR_RETURN(out->nd_sbp()->sbp_parallel_size(), 1);
  CHECK_OR_RETURN(EagerBoxingInterpreterUtil::IsBroadcastNdSbp(out->nd_sbp()));
  CHECK_OR_RETURN(in->placement() == out->placement());
  CHECK_OR_RETURN(in->placement()->device_type() == DeviceType::kGPU);
  return Maybe<void>::Ok();
}

static constexpr auto* CheckSymXToB = DECORATE(&RawCheckSymXToB, ThreadLocal);

Maybe<one::UserOpExpr> EagerNcclAllReduce(Symbol<ParallelDesc> parallel_desc) {
  return one::OpBuilder("eager_nccl_all_reduce", *JUST(UniqueStr("eager_nccl_all_reduce")))
      .Input("in")
      .Output("out")
      .Attr<std::string>("parallel_conf", PbMessage2TxtString(parallel_desc->parallel_conf()))
      .Build();
}

static constexpr auto* CachedEagerNcclAllReduceOpExpr = DECORATE(&EagerNcclAllReduce, ThreadLocal);

Maybe<one::UserOpExpr> EagerNcclAllGather(Symbol<ParallelDesc> parallel_desc) {
  return one::OpBuilder("eager_nccl_all_gather", *JUST(UniqueStr("eager_nccl_all_gather")))
      .Input("in")
      .Output("out")
      .Attr<std::string>("parallel_conf", PbMessage2TxtString(parallel_desc->parallel_conf()))
      .Build();
}

static constexpr auto* CachedEagerNcclAllGatherOpExpr = DECORATE(&EagerNcclAllGather, ThreadLocal);

}  // namespace

Maybe<one::Tensor> SymXToB(const std::shared_ptr<one::Tensor>& tensor, Symbol<PlacedNdSbp> in,
                           Symbol<PlacedNdSbp> out) {
  const auto& tensor_nd_sbp = JUST(tensor->nd_sbp());
  CHECK_OR_RETURN(tensor_nd_sbp == in->nd_sbp());
  const auto& tensor_placement = JUST(tensor->parallel_desc());
  CHECK_OR_RETURN(tensor_placement == in->placement());
  if (EagerBoxingInterpreterUtil::IsBroadcastSbp(tensor_nd_sbp->sbp_parallel(0))) { return tensor; }
  if (EagerBoxingInterpreterUtil::IsPartialSumSbp(tensor_nd_sbp->sbp_parallel(0))) {
    const auto& op_expr = JUST(CachedEagerNcclAllReduceOpExpr(in->placement()));
    return JUST(one::OpInterpUtil::Dispatch<one::Tensor>(*op_expr, {tensor}));
  }
  if (EagerBoxingInterpreterUtil::IsSplitSbp(tensor_nd_sbp->sbp_parallel(0), 0)) {
    const auto& op_expr = JUST(CachedEagerNcclAllGatherOpExpr(in->placement()));
    return JUST(one::OpInterpUtil::Dispatch<one::Tensor>(*op_expr, {tensor}));
  }
  UNIMPLEMENTED_THEN_RETURN();
}

COMMAND(RegisterBoxingFunction("symmetric-x-to-b", CheckSymXToB, &SymXToB));

}  // namespace oneflow
