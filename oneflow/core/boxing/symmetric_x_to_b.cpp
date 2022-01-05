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

Maybe<void> RawCheckSymmetricXToB(Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out,
                                  const Shape& logical_shape) {
  CHECK_EQ_OR_RETURN(in->nd_sbp()->sbp_parallel_size(), 1);
  CHECK_EQ_OR_RETURN(out->nd_sbp()->sbp_parallel_size(), 1);
  CHECK_OR_RETURN(IsAllBroadcastNdSbp(out->nd_sbp()));
  if (in->nd_sbp()->sbp_parallel(0).has_split_parallel()) {
    int64_t split_axis = in->nd_sbp()->sbp_parallel(0).split_parallel().axis();
    CHECK_OR_RETURN(logical_shape.At(split_axis) % in->placement()->parallel_num() == 0);
  }
  CHECK_OR_RETURN(in->placement() == out->placement());
  return Maybe<void>::Ok();
}

static constexpr auto* CheckSymmetricXToB = DECORATE(&RawCheckSymmetricXToB, ThreadLocalCopiable);

}  // namespace

Maybe<one::Tensor> SymmetricXToB(const std::shared_ptr<one::Tensor>& tensor, Symbol<PlacedNdSbp> in,
                                 Symbol<PlacedNdSbp> out) {
  const auto& tensor_nd_sbp = JUST(tensor->nd_sbp());
  CHECK_OR_RETURN(tensor_nd_sbp == in->nd_sbp());
  const auto& tensor_placement = JUST(tensor->parallel_desc());
  CHECK_OR_RETURN(tensor_placement == in->placement());
  if (IsBroadcastSbp(tensor_nd_sbp->sbp_parallel(0))) { return tensor; }
  if (IsPartialSumSbp(tensor_nd_sbp->sbp_parallel(0))) {
    BoxingFunctionT PToBBoxingFunction;
    if (tensor_placement->device_type() == DeviceType::kCUDA) {
      PToBBoxingFunction = *JUST(GetBoxingFunction("nccl-p-to-b", in, out, *tensor->shape()));
    } else if (tensor_placement->device_type() == DeviceType::kCPU) {
      PToBBoxingFunction = *JUST(GetBoxingFunction("ccl-p-to-b", in, out, *tensor->shape()));
    } else {
      UNIMPLEMENTED_THEN_RETURN();
    }
    return JUST(PToBBoxingFunction(tensor, in, out));
  }
  if (IsSplitSbp(tensor_nd_sbp->sbp_parallel(0), 0)) {
    BoxingFunctionT SToBBoxingFunction;
    if (tensor_placement->device_type() == DeviceType::kCUDA) {
      SToBBoxingFunction = *JUST(GetBoxingFunction("nccl-s-to-b", in, out, *tensor->shape()));
    } else if (tensor_placement->device_type() == DeviceType::kCPU) {
      SToBBoxingFunction = *JUST(GetBoxingFunction("ccl-s-to-b", in, out, *tensor->shape()));
    } else {
      UNIMPLEMENTED_THEN_RETURN();
    }
    return JUST(SToBBoxingFunction(tensor, in, out));
  }
  if (IsSplitSbpWithAxisNotEqualZero(in->nd_sbp()->sbp_parallel(0))) {
    Symbol<cfg::NdSbp> split_with_zero_axis_nd_sbp =
        JUST(CachedGetAllSplitNdSbpWithAxisEqualZero(in->nd_sbp()->sbp_parallel_size()));
    const auto& split_with_zero_axis_in_placed_nd_sbp =
        JUST(PlacedNdSbp::New(split_with_zero_axis_nd_sbp, tensor_placement));
    BoxingFunctionT SToSBoxingFunction;
    if (tensor_placement->device_type() == DeviceType::kCUDA) {
      SToSBoxingFunction = *JUST(GetBoxingFunction(
          "nccl-s-to-s", in, split_with_zero_axis_in_placed_nd_sbp, *tensor->shape()));
    } else if (tensor_placement->device_type() == DeviceType::kCPU) {
      SToSBoxingFunction = *JUST(GetBoxingFunction(
          "ccl-s-to-s", in, split_with_zero_axis_in_placed_nd_sbp, *tensor->shape()));
    } else {
      UNIMPLEMENTED_THEN_RETURN();
    }

    const auto& tensor_with_s0_sbp =
        JUST(SToSBoxingFunction(tensor, in, split_with_zero_axis_in_placed_nd_sbp));

    BoxingFunctionT SToBBoxingFunction;
    if (tensor_placement->device_type() == DeviceType::kCUDA) {
      SToBBoxingFunction = *JUST(GetBoxingFunction(
          "nccl-s-to-b", split_with_zero_axis_in_placed_nd_sbp, out, *tensor_with_s0_sbp->shape()));
    } else if (tensor_placement->device_type() == DeviceType::kCPU) {
      SToBBoxingFunction = *JUST(GetBoxingFunction(
          "ccl-s-to-b", split_with_zero_axis_in_placed_nd_sbp, out, *tensor_with_s0_sbp->shape()));
    } else {
      UNIMPLEMENTED_THEN_RETURN();
    }
    return JUST(SToBBoxingFunction(tensor_with_s0_sbp, split_with_zero_axis_in_placed_nd_sbp, out));
  }
  UNIMPLEMENTED_THEN_RETURN();
}

COMMAND(RegisterBoxingFunction("symmetric-x-to-b", CheckSymmetricXToB, &SymmetricXToB));

}  // namespace oneflow
