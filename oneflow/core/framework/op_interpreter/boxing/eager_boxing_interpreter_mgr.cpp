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
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/framework/op_interpreter/boxing/eager_boxing_interpreter_mgr.h"
#include "oneflow/core/framework/op_interpreter/boxing/eager_boxing_interpreter_util.h"
#include "oneflow/core/framework/op_interpreter/boxing/collective_boxing_interpreter.h"
#include "oneflow/core/framework/op_interpreter/boxing/identity_boxing_interpreter.h"
#include "oneflow/core/framework/op_interpreter/boxing/naive_b2p_boxing_interpreter.h"
#include "oneflow/core/framework/op_interpreter/boxing/naive_s2p_boxing_interpreter.h"
#include "oneflow/core/framework/op_interpreter/boxing/cuda_copy_boxing_interpreter.h"
#include "oneflow/core/framework/op_interpreter/boxing/cuda_based_cpu_mpi_boxing_interpreter.h"
#include "oneflow/core/framework/op_interpreter/boxing/boxing_dividor_util.h"

namespace oneflow {

namespace {
using SbpPair2EagerBoxingInterpreter =
    HashMap<std::pair<cfg::SbpParallel, cfg::SbpParallel>, std::shared_ptr<EagerBoxingInterpreter>>;

Maybe<EagerBoxingInterpreter> GetOneDimNcclCollectiveEagerBoxingInterpreter(
    Symbol<cfg::NdSbp> in_nd_sbp, Symbol<cfg::NdSbp> out_nd_sbp) {
  static SbpPair2EagerBoxingInterpreter sbp_pair2eager_boxing_interpreter = {
      {{*JUST(MakeSplitSbpParallel(0)), *JUST(MakeBroadcastSbpParallel())},  // S(0) -> B
       std::make_shared<NcclCollectiveAllGatherBoxingInterpreter>()},
      {{*JUST(MakeBroadcastSbpParallel()), *JUST(MakeSplitSbpParallel(0))},  // B -> S(0)
       std::make_shared<NcclCollectiveReduceScatterBoxingInterpreter>("max")},
      {{*JUST(MakePartialSumSbpParallel()), *JUST(MakeBroadcastSbpParallel())},  // P -> B
       std::make_shared<NcclCollectiveAllReduceBoxingInterpreter>()},
      {{*JUST(MakePartialSumSbpParallel()), *JUST(MakeSplitSbpParallel(0))},  // P -> S(0)
       std::make_shared<NcclCollectiveReduceScatterBoxingInterpreter>("sum")},
      {{*JUST(MakeSplitSbpParallel(0)), *JUST(MakePartialSumSbpParallel())},  // S(0) -> P
       std::make_shared<NcclS2PBoxingInterpreter>()},
  };
  const auto& key = std::make_pair(in_nd_sbp->sbp_parallel(0), out_nd_sbp->sbp_parallel(0));
  return JUST(MapAt(sbp_pair2eager_boxing_interpreter, key));
}

Maybe<EagerBoxingInterpreter> GetCudaBasedCpuMpiBoxingInterpreter(
    Symbol<cfg::NdSbp> in_nd_sbp, Symbol<cfg::NdSbp> out_nd_sbp,
    Symbol<ParallelDesc> in_parallel_desc, Symbol<ParallelDesc> out_parallel_desc) {
  CHECK_OR_RETURN(in_nd_sbp != out_nd_sbp);
  const auto& gpu_in_parallel_desc = JUST(ReplaceDeviceType(in_parallel_desc, DeviceType::kGPU));
  const auto& gpu_out_parallel_desc = JUST(ReplaceDeviceType(out_parallel_desc, DeviceType::kGPU));
  CHECK_OR_RETURN(gpu_in_parallel_desc == gpu_out_parallel_desc);
  const auto& gpu_boxing_interpreter =
      JUST(GetOneDimNcclCollectiveEagerBoxingInterpreter(in_nd_sbp, out_nd_sbp));
  return std::shared_ptr<EagerBoxingInterpreter>(new CudaBasedCpuMpiBoxingInterpreter());
}

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

Maybe<BoxingExprIf> RawMainBoxingExpr() {
  const auto& core = JUST(BoxingExpr("identity")) | JUST(BoxingExpr("flatten-hierarchy"))
                     | JUST(BoxingExpr("asymmetric-x-to-b"));
  return core | JUST(OptionalCudaCopy(core));
}

}  // namespace

static constexpr auto* MainBoxingExpr = DECORATE(&RawMainBoxingExpr, ThreadLocal);

Maybe<EagerBoxingInterpreter> GetBoxingInterpreter(Symbol<cfg::NdSbp> in_nd_sbp,
                                                   Symbol<cfg::NdSbp> out_nd_sbp,
                                                   Symbol<ParallelDesc> in_parallel_desc,
                                                   Symbol<ParallelDesc> out_parallel_desc) {
  if (in_parallel_desc == out_parallel_desc
      && (in_parallel_desc->parallel_num() == 1 || in_nd_sbp == out_nd_sbp)) {
    return std::shared_ptr<EagerBoxingInterpreter>(new IdentityBoxingInterpreter());
  }
  if (in_nd_sbp->sbp_parallel_size() == 1 && out_nd_sbp->sbp_parallel_size() == 1
      && in_parallel_desc == out_parallel_desc
      && EagerBoxingInterpreterUtil::IsBoxingB2P(in_nd_sbp->sbp_parallel(0),
                                                 out_nd_sbp->sbp_parallel(0))) {
    return std::shared_ptr<EagerBoxingInterpreter>(new NaiveB2PBoxingInterpreter());
  }
  if (in_nd_sbp->sbp_parallel_size() == 1 && out_nd_sbp->sbp_parallel_size() == 1
      && in_parallel_desc == out_parallel_desc
      && in_parallel_desc->device_type() == DeviceType::kGPU
      && EagerBoxingInterpreterUtil::IsBoxingS2S(in_nd_sbp->sbp_parallel(0),
                                                 out_nd_sbp->sbp_parallel(0))) {
    return std::shared_ptr<EagerBoxingInterpreter>(new NcclCollectiveS2SBoxingInterpreter());
  }
  if (in_nd_sbp->sbp_parallel_size() == 1 && out_nd_sbp->sbp_parallel_size() == 1
      && in_parallel_desc == out_parallel_desc
      && in_parallel_desc->device_type() == DeviceType::kGPU) {
    const auto& gpu_boxing_interpreter =
        TRY(GetOneDimNcclCollectiveEagerBoxingInterpreter(in_nd_sbp, out_nd_sbp));
    if (gpu_boxing_interpreter.IsOk()) { return JUST(gpu_boxing_interpreter); }
  }
  if (in_nd_sbp->sbp_parallel_size() == 1 && out_nd_sbp->sbp_parallel_size() == 1
      && in_parallel_desc == out_parallel_desc
      && in_parallel_desc->device_type() == DeviceType::kCPU) {
    const auto& interpreter = TRY(GetCudaBasedCpuMpiBoxingInterpreter(
        in_nd_sbp, out_nd_sbp, in_parallel_desc, out_parallel_desc));
    if (interpreter.IsOk()) { return JUST(interpreter); }
  }
  if (in_nd_sbp->sbp_parallel_size() == 1 && out_nd_sbp->sbp_parallel_size() == 1
      && JUST(IgnoringDeviceTypeEqual(in_parallel_desc, out_parallel_desc))
      && ((in_parallel_desc->device_type() == DeviceType::kGPU
           && out_parallel_desc->device_type() == DeviceType::kCPU)
          || (in_parallel_desc->device_type() == DeviceType::kCPU
              && out_parallel_desc->device_type() == DeviceType::kGPU))
      && in_nd_sbp == out_nd_sbp) {
    return std::shared_ptr<EagerBoxingInterpreter>(new CudaCopyBoxingInterpreter());
  }
  if (in_nd_sbp->sbp_parallel_size() == 1 && out_nd_sbp->sbp_parallel_size() == 1
      && JUST(IgnoringDeviceTypeEqual(in_parallel_desc, out_parallel_desc))
      && ((in_parallel_desc->device_type() == DeviceType::kGPU
           && out_parallel_desc->device_type() == DeviceType::kCPU)
          || (in_parallel_desc->device_type() == DeviceType::kCPU
              && out_parallel_desc->device_type() == DeviceType::kGPU))
      && in_nd_sbp != out_nd_sbp) {
    const auto& interpreter = TRY(GetCudaBasedCpuMpiBoxingInterpreter(
        in_nd_sbp, out_nd_sbp, in_parallel_desc, out_parallel_desc));
    if (interpreter.IsOk()) { return JUST(interpreter); }
  }
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
