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
#include "oneflow/core/boxing/boxing_dividor_util.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/framework/placed_nd_sbp.h"
#include "oneflow/core/framework/instructions_builder.h"
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

decltype(ReplaceInDeviceType) ReplaceInDeviceType =
    DECORATE(&RawReplaceInDeviceType, ThreadLocalCached);
decltype(ReplaceOutDeviceType) ReplaceOutDeviceType =
    DECORATE(&RawReplaceOutDeviceType, ThreadLocalCached);

namespace {

Maybe<Symbol<PlacedNdSbp>> RawFlattenHierarchy(Symbol<PlacedNdSbp> placed_nd_sbp) {
  CHECK_GE_OR_RETURN(placed_nd_sbp->nd_sbp()->sbp_parallel_size(), 0)
      << Error::RuntimeError() << "Invalid nd_sbp with ndim equal 0!";
  const auto& first_sbp_parallel = placed_nd_sbp->nd_sbp()->sbp_parallel(0);
  for (const auto& sbp_parallel : placed_nd_sbp->nd_sbp()->sbp_parallel()) {
    CHECK_OR_RETURN(sbp_parallel == first_sbp_parallel)
        << Error::RuntimeError()
        << "Expected all sbps to be on the same in sbp list during flatten sbps list, but find at "
           "least two sbps, "
        << SbpToString(first_sbp_parallel) << " and " << SbpToString(sbp_parallel) << "!";
  }
  std::vector<Symbol<SbpParallel>> vec{SymbolOf(first_sbp_parallel)};
  const auto& flattened_nd_sbp = JUST(GetNdSbp(vec));
  ParallelConf flattened_parallel_conf(placed_nd_sbp->placement()->parallel_conf());
  flattened_parallel_conf.clear_hierarchy();
  const auto& flattened_placement = SymbolOf(ParallelDesc(flattened_parallel_conf));
  return JUST(PlacedNdSbp::New(flattened_nd_sbp, flattened_placement));
}

static constexpr auto* FlattenHierarchy = DECORATE(&RawFlattenHierarchy, ThreadLocalCached);

Maybe<BoxingDividor> RawFlattenInHierarchy() {
  return std::make_shared<BoxingDividor>(
      "FlattenInHierarchy",
      [](Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out) -> Maybe<Symbol<PlacedNdSbp>> {
        return FlattenHierarchy(in);
      });
}

Maybe<Symbol<PlacedNdSbp>> RawUnflattenHierarchy(Symbol<PlacedNdSbp> in_placed_nd_sbp,
                                                 Symbol<PlacedNdSbp> out_placed_nd_sbp) {
  CHECK_GE_OR_RETURN(in_placed_nd_sbp->nd_sbp()->sbp_parallel_size(), 0)
      << Error::RuntimeError() << "Invalid nd_sbp with ndim equal 0!";
  CHECK_GE_OR_RETURN(out_placed_nd_sbp->nd_sbp()->sbp_parallel_size(), 0)
      << Error::RuntimeError() << "Invalid nd_sbp with ndim equal 0!";
  const auto& in_sbp_parallel = in_placed_nd_sbp->nd_sbp()->sbp_parallel(0);
  NdSbp unflattened_nd_sbp;
  for (int64_t i = 0; i < out_placed_nd_sbp->nd_sbp()->sbp_parallel_size(); ++i) {
    unflattened_nd_sbp.mutable_sbp_parallel()->Add()->CopyFrom(in_sbp_parallel);
  }
  return JUST(PlacedNdSbp::New(SymbolOf(unflattened_nd_sbp), out_placed_nd_sbp->placement()));
}

static constexpr auto* UnflattenHierarchy = DECORATE(&RawUnflattenHierarchy, ThreadLocalCached);

Maybe<BoxingDividor> RawUnflattenInHierarchy() {
  return std::make_shared<BoxingDividor>(
      "UnflattenInHierarchy",
      [](Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out) -> Maybe<Symbol<PlacedNdSbp>> {
        return UnflattenHierarchy(in, out);
      });
}

Maybe<BoxingDividor> RawUnflattenOutHierarchy() {
  return std::make_shared<BoxingDividor>(
      "UnflattenOutHierarchy",
      [](Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out) -> Maybe<Symbol<PlacedNdSbp>> {
        return UnflattenHierarchy(out, in);
      });
}

}  // namespace

decltype(FlattenInHierarchy) FlattenInHierarchy =
    DECORATE(&RawFlattenInHierarchy, ThreadLocalCached);
decltype(UnflattenInHierarchy) UnflattenInHierarchy =
    DECORATE(&RawUnflattenInHierarchy, ThreadLocalCached);
decltype(UnflattenOutHierarchy) UnflattenOutHierarchy =
    DECORATE(&RawUnflattenOutHierarchy, ThreadLocalCached);

namespace {

Maybe<Symbol<NdSbp>> GetAllPartialSumNdSbp(int64_t ndim) {
  NdSbp partial_sum_nd_sbp;
  for (int64_t i = 0; i < ndim; ++i) {
    partial_sum_nd_sbp.mutable_sbp_parallel()->Add()->mutable_partial_sum_parallel();
  }
  return SymbolOf(partial_sum_nd_sbp);
}

auto* CachedGetAllPartialSumNdSbp = DECORATE(&GetAllPartialSumNdSbp, ThreadLocalCached);

Maybe<Symbol<PlacedNdSbp>> RawReplaceNdSbpWithPartialSum(Symbol<PlacedNdSbp> placed_nd_sbp) {
  Symbol<NdSbp> partial_sum_nd_sbp =
      JUST(CachedGetAllPartialSumNdSbp(placed_nd_sbp->nd_sbp()->sbp_parallel_size()));
  return JUST(PlacedNdSbp::New(partial_sum_nd_sbp, placed_nd_sbp->placement()));
}

static constexpr auto* ReplaceNdSbpWithPartialSum =
    DECORATE(&RawReplaceNdSbpWithPartialSum, ThreadLocalCached);

Maybe<BoxingDividor> RawOutPlacementAndPartialSum() {
  return std::make_shared<BoxingDividor>(
      "OutPlacementAndPartialSum",
      [](Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out) -> Maybe<Symbol<PlacedNdSbp>> {
        return ReplaceNdSbpWithPartialSum(out);
      });
}

}  // namespace

decltype(OutPlacementAndPartialSum) OutPlacementAndPartialSum =
    DECORATE(&RawOutPlacementAndPartialSum, ThreadLocalCached);

namespace {

Maybe<Symbol<NdSbp>> GetAllBroadcastNdSbp(int64_t ndim) {
  NdSbp broadcast_nd_sbp;
  for (int64_t i = 0; i < ndim; ++i) {
    broadcast_nd_sbp.mutable_sbp_parallel()->Add()->mutable_broadcast_parallel();
  }
  return SymbolOf(broadcast_nd_sbp);
}

auto* CachedGetAllBroadcastNdSbp = DECORATE(&GetAllBroadcastNdSbp, ThreadLocalCached);

Maybe<Symbol<PlacedNdSbp>> RawReplaceNdSbpWithBroadcast(Symbol<PlacedNdSbp> placed_nd_sbp) {
  Symbol<NdSbp> broadcast_nd_sbp =
      JUST(CachedGetAllBroadcastNdSbp(placed_nd_sbp->nd_sbp()->sbp_parallel_size()));
  return JUST(PlacedNdSbp::New(broadcast_nd_sbp, placed_nd_sbp->placement()));
}

static constexpr auto* ReplaceNdSbpWithBroadcast =
    DECORATE(&RawReplaceNdSbpWithBroadcast, ThreadLocalCached);

Maybe<BoxingDividor> RawInPlacementAndBroadcast() {
  return std::make_shared<BoxingDividor>(
      "InPlacementAndBroadcast",
      [](Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out) -> Maybe<Symbol<PlacedNdSbp>> {
        return ReplaceNdSbpWithBroadcast(in);
      });
}

Maybe<BoxingDividor> RawOutPlacementAndBroadcast() {
  return std::make_shared<BoxingDividor>(
      "OutPlacementAndBroadcast",
      [](Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out) -> Maybe<Symbol<PlacedNdSbp>> {
        return ReplaceNdSbpWithBroadcast(out);
      });
}

}  // namespace

decltype(InPlacementAndBroadcast) InPlacementAndBroadcast =
    DECORATE(&RawInPlacementAndBroadcast, ThreadLocalCached);
decltype(OutPlacementAndBroadcast) OutPlacementAndBroadcast =
    DECORATE(&RawOutPlacementAndBroadcast, ThreadLocalCached);

namespace {

Maybe<Symbol<NdSbp>> GetSplitNdSbp(int64_t axis) {
  NdSbp split_nd_sbp;
  split_nd_sbp.mutable_sbp_parallel()->Add()->mutable_split_parallel()->set_axis(axis);
  return SymbolOf(split_nd_sbp);
}

auto* CachedGetSplitNdSbp = DECORATE(&GetSplitNdSbp, ThreadLocalCached);

Maybe<BoxingDividor> RawInPlacementAndSplit(int64_t axis) {
  return std::make_shared<BoxingDividor>(
      "InPlacementAndSplit",
      [=](Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out) -> Maybe<Symbol<PlacedNdSbp>> {
        Symbol<NdSbp> split_nd_sbp = JUST(CachedGetSplitNdSbp(axis));
        return PlacedNdSbp::New(split_nd_sbp, in->placement());
      });
}

Maybe<BoxingDividor> RawOutPlacementAndSplit(int64_t axis) {
  return std::make_shared<BoxingDividor>(
      "OutPlacementAndSplit",
      [=](Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out) -> Maybe<Symbol<PlacedNdSbp>> {
        Symbol<NdSbp> split_nd_sbp = JUST(CachedGetSplitNdSbp(axis));
        return PlacedNdSbp::New(split_nd_sbp, out->placement());
      });
}

}  // namespace

decltype(InPlacementAndSplit) InPlacementAndSplit =
    DECORATE(&RawInPlacementAndSplit, ThreadLocalCached);
decltype(OutPlacementAndSplit) OutPlacementAndSplit =
    DECORATE(&RawOutPlacementAndSplit, ThreadLocalCached);

namespace {

Maybe<Symbol<ParallelDesc>> GetFisrtDeviceOfPlacement(Symbol<ParallelDesc> placement) {
  ParallelConf parallel_conf;
  int64_t machine_id = JUST(placement->MachineId4ParallelId(0));
  int64_t device_id = JUST(placement->DeviceId4ParallelId(0));
  parallel_conf.set_device_tag(placement->device_tag());
  parallel_conf.add_device_name(std::string("@") + std::to_string(machine_id) + ":"
                                + std::to_string(device_id));
  for (int64_t i = 0; i < placement->hierarchy()->NumAxes(); ++i) {
    parallel_conf.mutable_hierarchy()->add_dim(1);
  }
  std::shared_ptr<ParallelDesc> parallel_desc;
  JUST(PhysicalRun([&parallel_desc, &parallel_conf](InstructionsBuilder* builder) -> Maybe<void> {
    parallel_desc = JUST(builder->GetParallelDescSymbol(parallel_conf));
    return Maybe<void>::Ok();
  }));
  return SymbolOf(*parallel_desc);
}

Maybe<BoxingDividor> RawInFirstDeviceAndAllBroadcast() {
  return std::make_shared<BoxingDividor>(
      "InFirstDeviceAndAllBroadcast",
      [](Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out) -> Maybe<Symbol<PlacedNdSbp>> {
        return PlacedNdSbp::New(JUST(CachedGetAllBroadcastNdSbp(in->nd_sbp()->sbp_parallel_size())),
                                JUST(GetFisrtDeviceOfPlacement(in->placement())));
      });
}

Maybe<BoxingDividor> RawOutFirstDeviceAndAllBroadcast() {
  return std::make_shared<BoxingDividor>(
      "OutFirstDeviceAndAllBroadcast",
      [](Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out) -> Maybe<Symbol<PlacedNdSbp>> {
        return PlacedNdSbp::New(
            JUST(CachedGetAllBroadcastNdSbp(out->nd_sbp()->sbp_parallel_size())),
            JUST(GetFisrtDeviceOfPlacement(out->placement())));
      });
}

}  //  namespace

decltype(InFirstDeviceAndAllBroadcast) InFirstDeviceAndAllBroadcast =
    DECORATE(&RawInFirstDeviceAndAllBroadcast, ThreadLocalCached);
decltype(OutFirstDeviceAndAllBroadcast) OutFirstDeviceAndAllBroadcast =
    DECORATE(&RawOutFirstDeviceAndAllBroadcast, ThreadLocalCached);

namespace {

Maybe<Symbol<PlacedNdSbp>> RawPlacementAndRepeatFirstSbp(Symbol<PlacedNdSbp> placed_nd_sbp) {
  const auto& first_sbp_parallel = placed_nd_sbp->nd_sbp()->sbp_parallel(0);
  NdSbp out_nd_sbp;
  for (int64_t i = 0; i < placed_nd_sbp->nd_sbp()->sbp_parallel_size(); ++i) {
    out_nd_sbp.mutable_sbp_parallel()->Add()->CopyFrom(first_sbp_parallel);
  }
  return JUST(PlacedNdSbp::New(SymbolOf(out_nd_sbp), placed_nd_sbp->placement()));
}

static constexpr auto* PlacementAndRepeatFirstSbp =
    DECORATE(&RawPlacementAndRepeatFirstSbp, ThreadLocalCached);

Maybe<BoxingDividor> RawInPlacementAndRepeatFirstSbp() {
  return std::make_shared<BoxingDividor>(
      "InPlacementAndRepeatFirstSbp",
      [](Symbol<PlacedNdSbp> in, Symbol<PlacedNdSbp> out) -> Maybe<Symbol<PlacedNdSbp>> {
        return PlacementAndRepeatFirstSbp(in);
      });
}

}  // namespace

decltype(InPlacementAndRepeatFirstSbp) InPlacementAndRepeatFirstSbp =
    DECORATE(&RawInPlacementAndRepeatFirstSbp, ThreadLocalCached);

}  // namespace oneflow
