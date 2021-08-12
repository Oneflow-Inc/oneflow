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
#include "oneflow/core/job/sbp_parallel.h"
#include "oneflow/core/framework/op_interpreter/boxing/eager_boxing_interpreter_mgr.h"
#include "oneflow/core/framework/op_interpreter/boxing/eager_boxing_interpreter_util.h"
#include "oneflow/core/framework/op_interpreter/boxing/collective_boxing_interpreter.h"
#include "oneflow/core/framework/op_interpreter/boxing/identity_boxing_interpreter.h"
#include "oneflow/core/framework/op_interpreter/boxing/naive_b2p_boxing_interpreter.h"
#include "oneflow/core/framework/op_interpreter/boxing/naive_s2p_boxing_interpreter.h"

namespace oneflow {

namespace {
using SbpPair2EagerBoxingInterpreter =
    HashMap<std::pair<cfg::SbpParallel, cfg::SbpParallel>, std::shared_ptr<EagerBoxingInterpreter>>;

Maybe<Symbol<cfg::SbpParallel>> GetSplitSbpParallel(int axis) {
  CHECK_LT_OR_RETURN(axis, kMaxSplitAxis);
  cfg::SbpParallel split_sbp_parallel;
  split_sbp_parallel.mutable_split_parallel()->set_axis(axis);
  return SymbolOf(split_sbp_parallel);
}

Maybe<Symbol<cfg::SbpParallel>> MakeBroadcastSbpParallel() {
  cfg::SbpParallel broadcast_sbp;
  broadcast_sbp.mutable_broadcast_parallel();
  return SymbolOf(broadcast_sbp);
}

Maybe<Symbol<cfg::SbpParallel>> MakePartialSumSbpParallel() {
  cfg::SbpParallel partial_sum_sbp;
  partial_sum_sbp.mutable_partial_sum_parallel();
  return SymbolOf(partial_sum_sbp);
}

std::string GetSupportedBoxingTypeInfo() {
  static std::string supported_boxing_type_info =
      "============ Supported eager boxing type============\n"
      "\'[S(0)] -> [B]\' on GPU\n"
      "\'[S(0)] -> [P]\' on GPU\n"
      "\'[P] -> [B]\' on GPU\n"
      "\'[P] -> [S(0)]\' on GPU\n"
      "\'[B] -> [S(0)]\' on GPU\n"
      "\'[B] -> [P]\' on GPU or CPU";
  return supported_boxing_type_info;
}

Maybe<EagerBoxingInterpreter> GetOneDimNcclCollectiveEagerBoxingInterpreter(
    Symbol<cfg::ParallelDistribution> in_nd_sbp, Symbol<cfg::ParallelDistribution> out_nd_sbp) {
  static SbpPair2EagerBoxingInterpreter sbp_pair2eager_boxing_interpreter = {
      {{*JUST(GetSplitSbpParallel(0)), *JUST(MakeBroadcastSbpParallel())},  // S(0) -> B
       std::make_shared<NcclCollectiveAllGatherBoxingInterpreter>()},
      {{*JUST(MakeBroadcastSbpParallel()), *JUST(GetSplitSbpParallel(0))},  // B -> S(0)
       std::make_shared<NcclCollectiveReduceScatterBoxingInterpreter>("max")},
      {{*JUST(MakePartialSumSbpParallel()), *JUST(MakeBroadcastSbpParallel())},  // P -> B
       std::make_shared<NcclCollectiveAllReduceBoxingInterpreter>()},
      {{*JUST(MakePartialSumSbpParallel()), *JUST(GetSplitSbpParallel(0))},  // P -> S(0)
       std::make_shared<NcclCollectiveReduceScatterBoxingInterpreter>("sum")},
      {{*JUST(GetSplitSbpParallel(0)), *JUST(MakePartialSumSbpParallel())},  // S(0) -> P
       std::make_shared<NcclS2PBoxingInterpreter>()},
  };
  const auto& key = std::make_pair(in_nd_sbp->sbp_parallel(0), out_nd_sbp->sbp_parallel(0));
  CHECK_OR_RETURN(sbp_pair2eager_boxing_interpreter.find(key)
                  != sbp_pair2eager_boxing_interpreter.end())
      << "Eager boxing type \'" << ParallelDistributionToString(in_nd_sbp) << " -> "
      << ParallelDistributionToString(out_nd_sbp) << "\'"
      << " not support yet\n"
      << GetSupportedBoxingTypeInfo();

  return JUST(MapAt(sbp_pair2eager_boxing_interpreter, key));
}

Maybe<EagerBoxingInterpreter> GetBoxingInterpreter(Symbol<cfg::ParallelDistribution> in_nd_sbp,
                                                   Symbol<cfg::ParallelDistribution> out_nd_sbp,
                                                   Symbol<ParallelDesc> in_parallel_desc,
                                                   Symbol<ParallelDesc> out_parallel_desc) {
  if (in_nd_sbp == out_nd_sbp && in_parallel_desc == out_parallel_desc) {
    static std::shared_ptr<EagerBoxingInterpreter> identity_boxing_interpreter =
        std::make_shared<IdentityBoxingInterpreter>();
    return identity_boxing_interpreter;
  }
  if (in_nd_sbp->sbp_parallel_size() == 1 && out_nd_sbp->sbp_parallel_size() == 1) {
    if (in_parallel_desc == out_parallel_desc) {
      if (EagerBoxingInterpreterUtil::IsBoxingB2P(in_nd_sbp->sbp_parallel(0),
                                                  out_nd_sbp->sbp_parallel(0))) {
        std::shared_ptr<EagerBoxingInterpreter> naive_bp_boxing_interpreter =
            std::make_shared<NaiveB2PBoxingInterpreter>();
        return naive_bp_boxing_interpreter;
      } else if (in_parallel_desc->device_type() == DeviceType::kGPU) {
        return GetOneDimNcclCollectiveEagerBoxingInterpreter(in_nd_sbp, out_nd_sbp);
      } else {
        UNIMPLEMENTED_THEN_RETURN()
            << "Eager boxing type \'" << ParallelDistributionToString(in_nd_sbp) << " -> "
            << ParallelDistributionToString(out_nd_sbp) << "\'"
            << " not support yet\n"
            << GetSupportedBoxingTypeInfo();
      }
    } else {
      UNIMPLEMENTED_THEN_RETURN() << "Eager boxing with different placement not support yet\n"
                                  << GetSupportedBoxingTypeInfo();
    }
  } else {
    UNIMPLEMENTED_THEN_RETURN() << "N-dim eager boxing type \'"
                                << ParallelDistributionToString(in_nd_sbp) << " -> "
                                << ParallelDistributionToString(out_nd_sbp) << "\'"
                                << " not support yet\n"
                                << GetSupportedBoxingTypeInfo();
  }
}

auto* CachedGetBoxingInterpreter = DECORATE(&GetBoxingInterpreter, ThreadLocal);

}  // namespace

Maybe<EagerBoxingInterpreter> EagerBoxingInterpreterManager::GetEagerBoxingInterpreter(
    Symbol<cfg::ParallelDistribution> in_nd_sbp, Symbol<cfg::ParallelDistribution> out_nd_sbp,
    Symbol<ParallelDesc> in_parallel_desc, Symbol<ParallelDesc> out_parallel_desc) const {
  return CachedGetBoxingInterpreter(in_nd_sbp, out_nd_sbp, in_parallel_desc, out_parallel_desc);
}

COMMAND(Global<EagerBoxingInterpreterManager>::SetAllocated(new EagerBoxingInterpreterManager()));

}  // namespace oneflow
