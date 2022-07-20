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
#include "oneflow/core/boxing/boxing_dividor_util.h"

namespace oneflow {

namespace {

Maybe<bool> IgnoringDeviceTypeEqual(Symbol<ParallelDesc> lhs, Symbol<ParallelDesc> rhs) {
  if (lhs == rhs) { return true; }
  return lhs == JUST(ReplaceDeviceType(rhs, lhs->device_type()));
}

namespace {

Maybe<BoxingExprIf> OptionalCudaCopy(const std::shared_ptr<BoxingExprIf>& core_boxing_expr) {
  return JUST(BoxingExpr(JUST(ReplaceInDeviceType(DeviceType::kCUDA)),
                         JUST(OptionalBoxing("copy-h2d")),
                         JUST(BoxingExpr(JUST(ReplaceOutDeviceType(DeviceType::kCUDA)),
                                         core_boxing_expr, JUST(OptionalBoxing("copy-d2h"))))));
}

Maybe<BoxingExprIf> OptionalCpuCopy(const std::shared_ptr<BoxingExprIf>& core_boxing_expr) {
  return JUST(BoxingExpr(JUST(ReplaceInDeviceType(DeviceType::kCPU)),
                         JUST(OptionalBoxing("copy-d2h")),
                         JUST(BoxingExpr(JUST(ReplaceOutDeviceType(DeviceType::kCPU)),
                                         core_boxing_expr, JUST(OptionalBoxing("copy-h2d"))))));
}

Maybe<BoxingExprIf> SymmetricOneDimSxToBBoxingExpr() {
  return JUST(BoxingExpr(JUST(InPlacementAndSplit(0)), JUST(OptionalBoxing("ccl-s-to-s")),
                         JUST(BoxingExpr("ccl-s-to-b"))));
}

Maybe<BoxingExprIf> SymmetricOneDimPToSxBoxingExpr() {
  return JUST(BoxingExpr(JUST(OutPlacementAndSplit(0)), JUST(BoxingExpr("ccl-p-to-s")),
                         JUST(OptionalBoxing("ccl-s-to-s"))));
}

Maybe<BoxingExprIf> SymmetricCyclicNDimToNDimBoxingExpr() {
  return JUST(BoxingExpr(JUST(InPlacementAndRepeatFirstSbp()),
                         JUST(BoxingExpr("symmetric-acyclic-nd-sbp-to-nd-sbp")),
                         JUST(BoxingExpr("symmetric-acyclic-nd-sbp-to-nd-sbp"))))
         | JUST(BoxingExpr(JUST(InPlacementAndBroadcast()),
                           JUST(BoxingExpr("symmetric-acyclic-nd-sbp-to-nd-sbp")),
                           JUST(BoxingExpr("symmetric-acyclic-nd-sbp-to-nd-sbp"))));
}

Maybe<BoxingExprIf> SymmetricNDimToNDimBoxingExpr() {
  return JUST(BoxingExpr("symmetric-acyclic-nd-sbp-to-nd-sbp"))
         | JUST(SymmetricCyclicNDimToNDimBoxingExpr());
}

Maybe<BoxingExprIf> SymmetricOneDimToNDimBoxingExpr() {
  return JUST(BoxingExpr(JUST(UnflattenInHierarchy()), JUST(BoxingExpr("unflatten-hierarchy")),
                         JUST(SymmetricNDimToNDimBoxingExpr()) | JUST(BoxingExpr("identity"))));
}

Maybe<BoxingExprIf> SymmetricNDimToOneDimBoxingExpr() {
  return JUST(BoxingExpr(JUST(UnflattenOutHierarchy()),
                         JUST(SymmetricNDimToNDimBoxingExpr()) | JUST(BoxingExpr("identity")),
                         JUST(BoxingExpr("flatten-hierarchy"))));
}

Maybe<BoxingExprIf> NToOneBoxingExpr() {
  return JUST(BoxingExpr(JUST(InPlacementAndBroadcast()),
                         JUST(BoxingExpr("identity")) | JUST(BoxingExpr("ccl-p-to-b"))
                             | JUST(SymmetricOneDimSxToBBoxingExpr())
                             | JUST(BoxingExpr("naive-p-to-b")) | JUST(BoxingExpr("naive-s-to-b"))
                             | JUST(SymmetricNDimToNDimBoxingExpr())
                             | JUST(BoxingExpr("generic-symmetric-nd-sbp-to-nd-sbp")),
                         JUST(BoxingExpr("naive-b-to-1"))));
}

Maybe<BoxingExprIf> OneToNBoxingExpr() {
  return JUST(BoxingExpr(JUST(OutPlacementAndPartialSum()), JUST(BoxingExpr("naive-1-to-p")),
                         JUST(BoxingExpr("identity")) | JUST(BoxingExpr("ccl-p-to-b"))
                             | JUST(SymmetricOneDimPToSxBoxingExpr())
                             | JUST(BoxingExpr("naive-p-to-b")) | JUST(BoxingExpr("naive-p-to-s"))
                             | JUST(SymmetricNDimToNDimBoxingExpr())
                             | JUST(BoxingExpr("generic-symmetric-nd-sbp-to-nd-sbp"))));
}

Maybe<BoxingExprIf> SymmetricOneDimXToBBoxingExpr() {
  return JUST(BoxingExpr("ccl-p-to-b"))
         | JUST(BoxingExpr(JUST(InPlacementAndSplit(0)),
                           JUST(BoxingExpr("identity")) | JUST(BoxingExpr("ccl-s-to-s")),
                           JUST(BoxingExpr("ccl-s-to-b"))));
}

Maybe<BoxingExprIf> ASymmetricOneDimXToBBoxingExpr() {
  return JUST(BoxingExpr(JUST(InPlacementAndBroadcast()),
                         JUST(BoxingExpr("identity")) | JUST(SymmetricOneDimXToBBoxingExpr()),
                         JUST(BoxingExpr("asymmetric-broadcast"))));
}

Maybe<BoxingExprIf> GenericBoxingExpr() {
  // in_placement contain out_placement or out_placement contain in_placement
  const auto& boxing_expr_with_inclusive_placement =
      JUST(BoxingExpr(JUST(OutPlacementAndBroadcast()), JUST(ASymmetricOneDimXToBBoxingExpr()),
                      JUST(BoxingExpr("identity")) | JUST(BoxingExpr("symmetric-b-to-p"))
                          | JUST(BoxingExpr("symmetric-b-to-s"))));
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
  // clang-format off
  const auto& core = JUST(BoxingExpr("identity"))
                     | JUST(BoxingExpr("copy-h2d"))
                     | JUST(BoxingExpr("copy-d2h"))
                     | JUST(BoxingExpr("ccl-p-to-b"))
                     | JUST(BoxingExpr("ccl-s-to-s"))
                     | JUST(SymmetricOneDimSxToBBoxingExpr())
                     | JUST(SymmetricOneDimPToSxBoxingExpr())
                     | JUST(BoxingExpr("symmetric-b-to-p"))
                     | JUST(BoxingExpr("symmetric-b-to-s"))
                     | JUST(BoxingExpr("symmetric-s-to-p"))
                     | JUST(SymmetricOneDimXToBBoxingExpr())
                     | JUST(ASymmetricOneDimXToBBoxingExpr())
                     | JUST(BoxingExpr("naive-1-to-1"))
                     | JUST(OneToNBoxingExpr())
                     | JUST(NToOneBoxingExpr())
                     | JUST(BoxingExpr("naive-s-to-s"))
                     | JUST(BoxingExpr("naive-s-to-b"))
                     | JUST(BoxingExpr("naive-b-to-s"))
                     | JUST(BoxingExpr("naive-p-to-b"))
                     | JUST(BoxingExpr("naive-p-to-s"))
                     | JUST(BoxingExpr("naive-s-to-p"))
                     | JUST(BoxingExpr("nd-sbp-dim-reduce"))
                     | JUST(SymmetricNDimToNDimBoxingExpr())
                     | JUST(BoxingExpr("generic-symmetric-nd-sbp-to-nd-sbp"))
                     | JUST(SymmetricOneDimToNDimBoxingExpr())
                     | JUST(SymmetricNDimToOneDimBoxingExpr())
                     | JUST(GenericBoxingExpr());
  // clang-format on
  return core | JUST(OptionalCudaCopy(core)) | JUST(OptionalCpuCopy(core));
}

}  // namespace

static constexpr auto* MainBoxingExpr = DECORATE(&RawMainBoxingExpr, ThreadLocalCached);

Maybe<EagerBoxingInterpreter> GetBoxingInterpreter(Symbol<NdSbp> in_nd_sbp,
                                                   Symbol<NdSbp> out_nd_sbp,
                                                   Symbol<ParallelDesc> in_parallel_desc,
                                                   Symbol<ParallelDesc> out_parallel_desc,
                                                   const Shape& logical_shape) {
  const auto& in = JUST(PlacedNdSbp::New(in_nd_sbp, in_parallel_desc));
  const auto& out = JUST(PlacedNdSbp::New(out_nd_sbp, out_parallel_desc));
  const auto& main_boxing_expr = JUST(MainBoxingExpr());
  const auto& status = TRY(main_boxing_expr->Check(in, out, logical_shape));
  if (status.IsOk()) {
    const auto& boxing_func = JUST(main_boxing_expr->GetBoxingFunction(in, out, logical_shape));
    return std::shared_ptr<EagerBoxingInterpreter>(
        new NaiveEagerBoxingInterpreter(boxing_func, JUST(status)));
  }

  UNIMPLEMENTED_THEN_RETURN() << Error::RuntimeError() << "global-to-global not supported"
                              << ". from_nd_sbp: " << NdSbpToString(in_nd_sbp)
                              << ", to_nd_sbp: " << NdSbpToString(out_nd_sbp)
                              << ", from_placement: " << *JUST(PlacementToString(in_parallel_desc))
                              << ", to_placement: " << *JUST(PlacementToString(out_parallel_desc));
}

static constexpr auto* CachedGetBoxingInterpreter =
    DECORATE(&GetBoxingInterpreter, ThreadLocalCachedCopiable);

}  // namespace

Maybe<EagerBoxingInterpreter> EagerBoxingInterpreterManager::GetEagerBoxingInterpreter(
    Symbol<NdSbp> in_nd_sbp, Symbol<NdSbp> out_nd_sbp, Symbol<ParallelDesc> in_parallel_desc,
    Symbol<ParallelDesc> out_parallel_desc, const Shape& logical_shape) const {
  return JUST(CachedGetBoxingInterpreter(in_nd_sbp, out_nd_sbp, in_parallel_desc, out_parallel_desc,
                                         logical_shape));
}

COMMAND(
    Singleton<EagerBoxingInterpreterManager>::SetAllocated(new EagerBoxingInterpreterManager()));

}  // namespace oneflow
