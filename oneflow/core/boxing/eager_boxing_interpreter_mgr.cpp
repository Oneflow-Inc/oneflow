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
#include <utility>
#include "oneflow/core/common/constant.h"
#include "oneflow/core/common/decorator.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/boxing/eager_boxing_interpreter_mgr.h"
#include "oneflow/core/boxing/eager_boxing_interpreter_util.h"
#include "oneflow/core/boxing/boxing_dividor_util.h"

namespace oneflow {

namespace {

Maybe<bool> IgnoringDeviceTypeEqual(Symbol<ParallelDesc> lhs, Symbol<ParallelDesc> rhs) {
  if (lhs == rhs) { return true; }
  return lhs == JUST(ReplaceDeviceType(rhs, lhs->device_type()));
}

namespace {

Maybe<BoxingExprIf> OptionalCudaCopy(const std::shared_ptr<BoxingExprIf>& core_boxing_expr) {
  return JUST(
      BoxingExpr(JUST(ReplaceInDeviceType(DeviceType::kGPU)), JUST(OptionalBoxing("cuda-copy-h2d")),
                 JUST(BoxingExpr(JUST(ReplaceOutDeviceType(DeviceType::kGPU)), core_boxing_expr,
                                 JUST(OptionalBoxing("cuda-copy-d2h"))))));
}

Maybe<BoxingExprIf> NcclSxToBBoxingExpr() {
  return JUST(BoxingExpr(JUST(InPlacementAndSplit(0)), JUST(OptionalBoxing("nccl-s-to-s")),
                         JUST(BoxingExpr("nccl-s-to-b"))));
}

Maybe<BoxingExprIf> NcclBToSxBoxingExpr() {
  return JUST(BoxingExpr(JUST(InPlacementAndSplit(0)), JUST(BoxingExpr("nccl-b-to-s")),
                         JUST(OptionalBoxing("nccl-s-to-s"))));
}

Maybe<BoxingExprIf> NcclPToSxBoxingExpr() {
  return JUST(BoxingExpr(JUST(OutPlacementAndSplit(0)), JUST(BoxingExpr("nccl-p-to-s")),
                         JUST(OptionalBoxing("nccl-s-to-s"))));
}

Maybe<BoxingExprIf> NToOneBoxingExpr() {
  return JUST(BoxingExpr(
      JUST(InPlacementAndBroadcast()),
      JUST(BoxingExpr("nccl-p-to-b")) | JUST(NcclSxToBBoxingExpr()) | JUST(BoxingExpr("identity")),
      JUST(BoxingExpr("naive-b-to-1"))));
}

Maybe<BoxingExprIf> OneToNBoxingExpr() {
  return JUST(BoxingExpr(JUST(OutPlacementAndPartialSum()), JUST(BoxingExpr("naive-1-to-p")),
                         JUST(BoxingExpr("nccl-p-to-b")) | JUST(NcclPToSxBoxingExpr())
                             | JUST(BoxingExpr("identity"))));
}

Maybe<BoxingExprIf> GenericBoxingExpr() {
  // in_placement contain out_placement or out_placement contain in_placement
  const auto& boxing_expr_with_inclusive_placement =
      JUST(BoxingExpr(JUST(OutPlacementAndBroadcast()), JUST(BoxingExpr("asymmetric-x-to-b")),
                      JUST(BoxingExpr("identity")) | JUST(BoxingExpr("naive-b-to-p"))
                          | JUST(NcclBToSxBoxingExpr())));
  // in_placement and out_placement have no containment relationship
  // n to 1
  const auto& lhs_boxing = JUST(NToOneBoxingExpr());
  // 1 to 1 -> 1 to n
  const auto& rhs_boxing =
      JUST(BoxingExpr(JUST(OutFirstDeviceAndAllBroadcast()), JUST(OptionalBoxing("naive-1-to-1")),
                      JUST(OneToNBoxingExpr())));
  return boxing_expr_with_inclusive_placement
         | JUST(BoxingExpr(JUST(InFirstDeviceAndAllBroadcast()), lhs_boxing, rhs_boxing));
}

Maybe<BoxingExprIf> RawMainBoxingExpr() {
  const auto& core =
      JUST(BoxingExpr("identity")) | JUST(BoxingExpr("flatten-hierarchy"))
      | JUST(BoxingExpr("cuda-copy-h2d")) | JUST(BoxingExpr("cuda-copy-d2h"))
      | JUST(BoxingExpr("nccl-p-to-b")) | JUST(BoxingExpr("nccl-p-to-s"))
      | JUST(BoxingExpr("nccl-b-to-s")) | JUST(BoxingExpr("nccl-s-to-b"))
      | JUST(BoxingExpr("nccl-s-to-s")) | JUST(BoxingExpr("naive-b-to-p"))
      | JUST(BoxingExpr("ccl-s-to-s")) | JUST(BoxingExpr("symmetric-nd-sbp-to-nd-sbp"))
      | JUST(BoxingExpr(JUST(InPlacementAndBroadcast()), JUST(BoxingExpr("nccl-s-to-b")),
                        JUST(BoxingExpr("naive-b-to-p"))))
      | JUST(BoxingExpr("asymmetric-x-to-b")) | JUST(OneToNBoxingExpr()) | JUST(NToOneBoxingExpr())
      | JUST(BoxingExpr("naive-1-to-1")) | JUST(GenericBoxingExpr());
  return core | JUST(OptionalCudaCopy(core));
}

}  // namespace

static constexpr auto* MainBoxingExpr = DECORATE(&RawMainBoxingExpr, ThreadLocal);

Maybe<EagerBoxingInterpreter> GetBoxingInterpreter(Symbol<cfg::NdSbp> in_nd_sbp,
                                                   Symbol<cfg::NdSbp> out_nd_sbp,
                                                   Symbol<ParallelDesc> in_parallel_desc,
                                                   Symbol<ParallelDesc> out_parallel_desc) {
  const auto& in = JUST(PlacedNdSbp::New(in_nd_sbp, in_parallel_desc));
  const auto& out = JUST(PlacedNdSbp::New(out_nd_sbp, out_parallel_desc));
  const auto& main_boxing_expr = JUST(MainBoxingExpr());
  if (TRY(main_boxing_expr->Check(in, out)).IsOk()) {
    const auto& boxing_func = JUST(main_boxing_expr->GetBoxingFunction(in, out));
    return std::shared_ptr<EagerBoxingInterpreter>(new NaiveEagerBoxingInterpreter(boxing_func));
  }

  UNIMPLEMENTED_THEN_RETURN() << Error::BoxingNotSupportedError()
                              << "consistent-to-consistent not supported"
                              << ". from_nd_sbp: " << *JUST(NdSbpToString(in_nd_sbp))
                              << ", to_nd_sbp: " << *JUST(NdSbpToString(out_nd_sbp))
                              << ", from_placement: " << *JUST(PlacementToString(in_parallel_desc))
                              << ", to_placement: " << *JUST(PlacementToString(out_parallel_desc));
}

static constexpr auto* CachedGetBoxingInterpreter = DECORATE(&GetBoxingInterpreter, ThreadLocal);

}  // namespace

Maybe<EagerBoxingInterpreter> EagerBoxingInterpreterManager::GetEagerBoxingInterpreter(
    Symbol<cfg::NdSbp> in_nd_sbp, Symbol<cfg::NdSbp> out_nd_sbp,
    Symbol<ParallelDesc> in_parallel_desc, Symbol<ParallelDesc> out_parallel_desc) const {
  return CachedGetBoxingInterpreter(in_nd_sbp, out_nd_sbp, in_parallel_desc, out_parallel_desc);
}

COMMAND(Global<EagerBoxingInterpreterManager>::SetAllocated(new EagerBoxingInterpreterManager()));

}  // namespace oneflow
