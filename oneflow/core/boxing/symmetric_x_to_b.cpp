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
#include "oneflow/core/boxing/eager_boxing_interpreter.h"
#include "oneflow/core/common/decorator.h"

namespace oneflow {

namespace {

bool IsBroadcastSbp(const cfg::SbpParallel& sbp) { return sbp.has_broadcast_parallel(); }

bool IsPartialSumSbp(const cfg::SbpParallel& sbp) { return sbp.has_partial_sum_parallel(); }

bool IsSplitSbp(const cfg::SbpParallel& sbp, int64_t axis) {
  return (sbp.has_split_parallel() && sbp.split_parallel().axis() == axis);
}

bool IsAllBroadcastNdSbp(Symbol<cfg::NdSbp> nd_sbp) {
  for (const auto& sbp_parallel : nd_sbp->sbp_parallel()) {
    if (!sbp_parallel.has_broadcast_parallel()) { return false; }
  }
  return true;
}

bool IsSplitSbpWithAxisNotEqualZero(const cfg::SbpParallel& sbp) {
  return sbp.has_split_parallel() && sbp.split_parallel().axis() != 0;
}

Maybe<Symbol<cfg::NdSbp>> GetAllSplitNdSbpWithAxisEqualZero(int64_t ndim) {
  cfg::NdSbp split_nd_sbp;
  for (int64_t i = 0; i < ndim; ++i) {
    split_nd_sbp.mutable_sbp_parallel()->Add()->mutable_split_parallel()->set_axis(0);
  }
  return SymbolOf(split_nd_sbp);
}

auto* CachedGetAllSplitNdSbpWithAxisEqualZero =
    DECORATE(&GetAllSplitNdSbpWithAxisEqualZero, ThreadLocal);

Maybe<void> RawCheckSymmetricXToB(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out) {
  CHECK_EQ_OR_RETURN(in->nd_sbp()->sbp_parallel_size(), 1);
  CHECK_EQ_OR_RETURN(out->nd_sbp()->sbp_parallel_size(), 1);
  CHECK_OR_RETURN(IsAllBroadcastNdSbp(out->nd_sbp()));
  CHECK_OR_RETURN(in->placement() == out->placement());
  CHECK_OR_RETURN(in->placement()->device_type() == DeviceType::kGPU);
  return Maybe<void>::Ok();
}

static constexpr auto* CheckSymmetricXToB = DECORATE(&RawCheckSymmetricXToB, ThreadLocal);

}  // namespace

Maybe<one::Tensor> SymmetricXToB(const std::shared_ptr<one::Tensor>& tensor, Symbol<PlacedNdSbp> in,
                                 Symbol<PlacedNdSbp> out) {
  const auto& tensor_nd_sbp = JUST(tensor->nd_sbp());
  CHECK_OR_RETURN(tensor_nd_sbp == in->nd_sbp());
  const auto& tensor_placement = JUST(tensor->parallel_desc());
  CHECK_OR_RETURN(tensor_placement == in->placement());
  if (IsBroadcastSbp(tensor_nd_sbp->sbp_parallel(0))) { return tensor; }
  if (IsPartialSumSbp(tensor_nd_sbp->sbp_parallel(0))) {
    const auto& NcclPToBBoxingFunction = *JUST(GetBoxingFunction("nccl-p-to-b", in, out));
    return JUST(NcclPToBBoxingFunction(tensor, in, out));
  }
  if (IsSplitSbp(tensor_nd_sbp->sbp_parallel(0), 0)) {
    const auto& NcclSToBBoxingFunction = *JUST(GetBoxingFunction("nccl-s-to-b", in, out));
    return JUST(NcclSToBBoxingFunction(tensor, in, out));
  }
  if (IsSplitSbpWithAxisNotEqualZero(in->nd_sbp()->sbp_parallel(0))) {
    Symbol<cfg::NdSbp> split_with_zero_axis_nd_sbp =
        JUST(CachedGetAllSplitNdSbpWithAxisEqualZero(in->nd_sbp()->sbp_parallel_size()));
    const auto& split_with_zero_axis_in_placed_nd_sbp =
        JUST(PlacedNdSbp::New(split_with_zero_axis_nd_sbp, tensor_placement));
    const auto& NcclSToSBoxingFunction =
        *JUST(GetBoxingFunction("nccl-s-to-s", in, split_with_zero_axis_in_placed_nd_sbp));
    const auto& tensor_with_s0_sbp =
        JUST(NcclSToSBoxingFunction(tensor, in, split_with_zero_axis_in_placed_nd_sbp));
    const auto& NcclSToBBoxingFunction =
        *JUST(GetBoxingFunction("nccl-s-to-b", split_with_zero_axis_in_placed_nd_sbp, out));
    return JUST(
        NcclSToBBoxingFunction(tensor_with_s0_sbp, split_with_zero_axis_in_placed_nd_sbp, out));
  }
  UNIMPLEMENTED_THEN_RETURN();
}

COMMAND(RegisterBoxingFunction("symmetric-x-to-b", CheckSymmetricXToB, &SymmetricXToB));

}  // namespace oneflow
