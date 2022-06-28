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
#include "oneflow/core/common/decorator.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/framework/placed_nd_sbp.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/boxing/boxing_interpreter_status.h"

namespace oneflow {

namespace {

Maybe<BoxingInterpreterStatus> RawMakeBoxingInterpreterStatus(const std::string& boxing_name,
                                                              const Shape& logical_shape,
                                                              Symbol<PlacedNdSbp> in,
                                                              Symbol<PlacedNdSbp> out) {
  std::vector<std::string> sorted_boxing_names{boxing_name};
  BoxingInterpreterStatus status(SymbolOf(sorted_boxing_names), logical_shape, in, out);
  return status;
}

Maybe<BoxingInterpreterStatus> RawMakeComposedBoxingInterpreterStatus(
    const std::shared_ptr<BoxingInterpreterStatus>& lhs_status,
    const std::shared_ptr<BoxingInterpreterStatus>& rhs_status) {
  CHECK_OR_RETURN(lhs_status->dst_placed_nd_sbp()
                  == rhs_status->src_placed_nd_sbp())  // always true
      << Error::RuntimeError()
      << "Intermediate placed_nd_sbp must be equal when compose boxing interpreter status"
      << ". lhs_status.dst_nd_sbp: " << NdSbpToString(lhs_status->dst_placed_nd_sbp()->nd_sbp())
      << ", rhs_status.dst_nd_sbp: " << NdSbpToString(rhs_status->src_placed_nd_sbp()->nd_sbp())
      << ", lhs_status.dst_placement: "
      << *JUST(PlacementToString(lhs_status->dst_placed_nd_sbp()->placement()))
      << ", rhs_status.dst_placement: "
      << *JUST(PlacementToString(rhs_status->src_placed_nd_sbp()->placement()));
  CHECK_OR_RETURN(lhs_status->logical_shape() == rhs_status->logical_shape())  // always true
      << Error::RuntimeError()
      << "Logical_shape must be equal when compose boxing interpreter status"
      << ". lhs_status.logical_shape: " << (lhs_status->logical_shape().ToString())
      << ". rhs_status.logical_shape: " << (rhs_status->logical_shape().ToString());
  std::vector<std::string> sorted_boxing_names(*lhs_status->sorted_boxing_names());
  sorted_boxing_names.insert(sorted_boxing_names.end(), rhs_status->sorted_boxing_names()->begin(),
                             rhs_status->sorted_boxing_names()->end());
  std::vector<Symbol<PlacedNdSbp>> mid_placed_nd_sbp(*lhs_status->mid_placed_nd_sbp());
  mid_placed_nd_sbp.emplace_back(lhs_status->dst_placed_nd_sbp());
  mid_placed_nd_sbp.insert(mid_placed_nd_sbp.end(), rhs_status->mid_placed_nd_sbp()->begin(),
                           rhs_status->mid_placed_nd_sbp()->end());
  BoxingInterpreterStatus status(sorted_boxing_names, lhs_status->logical_shape(),
                                 lhs_status->src_placed_nd_sbp(), SymbolOf(mid_placed_nd_sbp),
                                 rhs_status->dst_placed_nd_sbp());
  return status;
}

}  // namespace

decltype(MakeBoxingInterpreterStatus) MakeBoxingInterpreterStatus =
    DECORATE(&RawMakeBoxingInterpreterStatus, ThreadLocalCachedCopiable);
decltype(MakeComposedBoxingInterpreterStatus) MakeComposedBoxingInterpreterStatus =
    DECORATE(&RawMakeComposedBoxingInterpreterStatus, ThreadLocalCachedCopiable);

namespace {

Maybe<std::string> RawGetNdSbpRouting(Symbol<PlacedNdSbp> src_placed_nd_sbp,
                                      Symbol<std::vector<Symbol<PlacedNdSbp>>> mid_placed_nd_sbp,
                                      Symbol<PlacedNdSbp> dst_placed_nd_sbp) {
  std::ostringstream ss;
  ss << NdSbpToString(src_placed_nd_sbp->nd_sbp());
  for (const auto& placed_nd_sbp : *mid_placed_nd_sbp) {
    ss << " -> " << NdSbpToString(placed_nd_sbp->nd_sbp());
  }
  ss << " -> " << NdSbpToString(dst_placed_nd_sbp->nd_sbp());
  return ss.str();
}

Maybe<std::string> RawGetPlacementRouting(
    Symbol<PlacedNdSbp> src_placed_nd_sbp,
    Symbol<std::vector<Symbol<PlacedNdSbp>>> mid_placed_nd_sbp,
    Symbol<PlacedNdSbp> dst_placed_nd_sbp) {
  std::ostringstream ss;
  ss << *JUST(PlacementToString(src_placed_nd_sbp->placement()));
  for (const auto& placed_nd_sbp : *mid_placed_nd_sbp) {
    ss << " -> " << *JUST(PlacementToString(placed_nd_sbp->placement()));
  }
  ss << " -> " << *JUST(PlacementToString(dst_placed_nd_sbp->placement()));
  return ss.str();
}

Maybe<std::string> RawGetBoxingDesc(Symbol<std::vector<std::string>> sorted_boxing_names) {
  CHECK_OR_RETURN(!sorted_boxing_names->empty())  // always true
      << Error::RuntimeError() << "boxing_names of eager boxing status can't be empty!";
  std::ostringstream ss;
  ss << sorted_boxing_names->at(0);
  for (size_t i = 1; i < sorted_boxing_names->size(); ++i) {
    ss << " -> " << sorted_boxing_names->at(i);
  }
  return ss.str();
}

static constexpr auto* GetNdSbpRouting = DECORATE(&RawGetNdSbpRouting, ThreadLocalCached);
static constexpr auto* GetPlacementRouting = DECORATE(&RawGetPlacementRouting, ThreadLocalCached);
static constexpr auto* GetBoxingDesc = DECORATE(&RawGetBoxingDesc, ThreadLocalCached);

}  // namespace

const std::string& BoxingInterpreterStatus::boxing_routing() const {
  return *CHECK_JUST(GetBoxingDesc(sorted_boxing_names_));
}

const std::string& BoxingInterpreterStatus::nd_sbp_routing() const {
  return *CHECK_JUST(GetNdSbpRouting(src_placed_nd_sbp_, mid_placed_nd_sbp_, dst_placed_nd_sbp_));
}

const std::string& BoxingInterpreterStatus::placement_routing() const {
  return *CHECK_JUST(
      GetPlacementRouting(src_placed_nd_sbp_, mid_placed_nd_sbp_, dst_placed_nd_sbp_));
}

}  // namespace oneflow
