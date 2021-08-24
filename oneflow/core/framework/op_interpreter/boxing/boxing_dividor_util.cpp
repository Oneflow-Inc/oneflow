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
#include "oneflow/core/framework/op_interpreter/boxing/boxing_dividor_util.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/framework/placed_nd_sbp.h"
#include "oneflow/core/common/decorator.h"
#include "oneflow/core/job/parallel_desc.h"

namespace oneflow {

namespace {

Maybe<BoxingDividor> RawReplaceInDeviceType(DeviceType device_type) {
  return std::make_shared<BoxingDividor>(
      "ReplaceInDeviceType",
      [device_type](Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out) -> Maybe<Symbol<PlacedNdSbp>> {
        const auto& new_placement = JUST(ReplaceDeviceType(in->placement(), device_type));
        return PlacedNdSbp::New(in->nd_sbp(), new_placement);
      });
}

Maybe<BoxingDividor> RawReplaceOutDeviceType(DeviceType device_type) {
  return std::make_shared<BoxingDividor>(
      "ReplaceOutDeviceType",
      [device_type](Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out) -> Maybe<Symbol<PlacedNdSbp>> {
        const auto& new_placement = JUST(ReplaceDeviceType(out->placement(), device_type));
        return PlacedNdSbp::New(out->nd_sbp(), new_placement);
      });
}

}  // namespace

decltype(ReplaceInDeviceType) ReplaceInDeviceType = DECORATE(&RawReplaceInDeviceType, ThreadLocal);
decltype(ReplaceOutDeviceType) ReplaceOutDeviceType =
    DECORATE(&RawReplaceOutDeviceType, ThreadLocal);

namespace {

Maybe<Symbol<PlacedNdSbp>> RawFlattenHierarchy(Symbol<PlacedNdSbp> placed_nd_sbp) {
  CHECK_GE_OR_RETURN(placed_nd_sbp->nd_sbp()->sbp_parallel_size(), 0);
  const auto& first_sbp_parallel = placed_nd_sbp->nd_sbp()->sbp_parallel(0);
  for (const auto& sbp_parallel : placed_nd_sbp->nd_sbp()->sbp_parallel()) {
    CHECK_OR_RETURN(sbp_parallel == first_sbp_parallel);
  }
  std::vector<Symbol<cfg::SbpParallel>> vec{SymbolOf(first_sbp_parallel)};
  const auto& flattened_nd_sbp = JUST(GetNdSbp(vec));
  ParallelConf flattened_parallel_conf(placed_nd_sbp->placement()->parallel_conf());
  flattened_parallel_conf.clear_hierarchy();
  const auto& flattened_placement = SymbolOf(ParallelDesc(flattened_parallel_conf));
  return JUST(PlacedNdSbp::New(flattened_nd_sbp, flattened_placement));
}

static constexpr auto* FlattenHierarchy = DECORATE(&RawFlattenHierarchy, ThreadLocal);

Maybe<BoxingDividor> RawFlattenInHierarchy() {
  return std::make_shared<BoxingDividor>(
      "FlattenInHierarchy",
      [](Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out) -> Maybe<Symbol<PlacedNdSbp>> {
        return FlattenHierarchy(in);
      });
}

}  // namespace

decltype(FlattenInHierarchy) FlattenInHierarchy = DECORATE(&RawFlattenInHierarchy, ThreadLocal);

namespace {

Maybe<Symbol<cfg::NdSbp>> GetPartialSumNdSbp() {
  cfg::NdSbp partial_sum_nd_sbp;
  partial_sum_nd_sbp.mutable_sbp_parallel()->Add()->mutable_partial_sum_parallel();
  return SymbolOf(partial_sum_nd_sbp);
}

auto* CachedGetPartialSumNdSbp = DECORATE(&GetPartialSumNdSbp, ThreadLocal);

Maybe<Symbol<PlacedNdSbp>> RawReplaceNdSbpWithPartialSum(Symbol<PlacedNdSbp> placed_nd_sbp) {
  Symbol<cfg::NdSbp> partial_sum_nd_sbp = JUST(CachedGetPartialSumNdSbp());
  return JUST(PlacedNdSbp::New(partial_sum_nd_sbp, placed_nd_sbp->placement()));
}

static constexpr auto* ReplaceNdSbpWithPartialSum =
    DECORATE(&RawReplaceNdSbpWithPartialSum, ThreadLocal);

Maybe<BoxingDividor> RawOutPlacementAndPartialSum() {
  return std::make_shared<BoxingDividor>(
      "OutPlacementAndPartialSum",
      [](Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out) -> Maybe<Symbol<PlacedNdSbp>> {
        return ReplaceNdSbpWithPartialSum(out);
      });
}

}  // namespace

decltype(OutPlacementAndPartialSum) OutPlacementAndPartialSum =
    DECORATE(&RawOutPlacementAndPartialSum, ThreadLocal);
}  // namespace oneflow
