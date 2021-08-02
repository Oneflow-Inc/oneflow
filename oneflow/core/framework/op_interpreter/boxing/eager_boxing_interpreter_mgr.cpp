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

Maybe<EagerBoxingInterpreter> GetOneDimNcclCollectiveEagerBoxingInterpreter(
    Symbol<cfg::ParallelDistribution> in_parallel_distribution,
    Symbol<cfg::ParallelDistribution> out_parallel_distribution) {
  // static SbpPair2EagerBoxingInterpreter sbp_pair2eager_boxing_interpreter;
  static SbpPair2EagerBoxingInterpreter sbp_pair2eager_boxing_interpreter = {
      {{*JUST(GetSplitSbpParallel(0)), *JUST(MakeBroadcastSbpParallel())},
       std::make_shared<NcclCollectiveAllGatherBoxingInterpreter>()},
      {{*JUST(MakeBroadcastSbpParallel()), *JUST(GetSplitSbpParallel(0))},
       std::make_shared<NcclCollectiveReduceScatterBoxingInterpreter>("max")},
      {{*JUST(MakePartialSumSbpParallel()), *JUST(MakeBroadcastSbpParallel())},
       std::make_shared<NcclCollectiveAllReduceBoxingInterpreter>()},
      {{*JUST(MakePartialSumSbpParallel()), *JUST(GetSplitSbpParallel(0))},
       std::make_shared<NcclCollectiveReduceScatterBoxingInterpreter>("sum")},
      {{*JUST(GetSplitSbpParallel(0)), *JUST(MakePartialSumSbpParallel())},
       std::make_shared<NcclS2PBoxingInterpreter>()},
  };
  return sbp_pair2eager_boxing_interpreter.at(
      {in_parallel_distribution->sbp_parallel(0), out_parallel_distribution->sbp_parallel(0)});
}

}  // namespace

Maybe<EagerBoxingInterpreter> EagerBoxingInterpreterManager::GetEagerBoxingInterpreter(
    Symbol<cfg::ParallelDistribution> in_parallel_distribution,
    Symbol<cfg::ParallelDistribution> out_parallel_distribution,
    Symbol<ParallelDesc> in_parallel_desc, Symbol<ParallelDesc> out_parallel_desc) const {
  if (in_parallel_distribution == out_parallel_distribution
      && in_parallel_desc == out_parallel_desc) {
    std::shared_ptr<EagerBoxingInterpreter> identity_boxing_interpreter =
        std::make_shared<IdentityBoxingInterpreter>();
    return identity_boxing_interpreter;
  }
  if (in_parallel_distribution->sbp_parallel_size() == 1
      && out_parallel_distribution->sbp_parallel_size() == 1) {
    if (EagerBoxingInterpreterUtil::IsPlacementSymmetrical(in_parallel_desc, out_parallel_desc)) {
      if (in_parallel_desc == out_parallel_desc
          && EagerBoxingInterpreterUtil::IsBoxingB2P(in_parallel_distribution->sbp_parallel(0),
                                                     out_parallel_distribution->sbp_parallel(0))) {
        std::shared_ptr<EagerBoxingInterpreter> naive_bp_boxing_interpreter =
            std::make_shared<NaiveB2PBoxingInterpreter>();
        return naive_bp_boxing_interpreter;
      }
      if (EagerBoxingInterpreterUtil::IsDeviceTypeGPU(in_parallel_desc)) {
        return GetOneDimNcclCollectiveEagerBoxingInterpreter(in_parallel_distribution,
                                                             out_parallel_distribution);
      } else {
        OF_UNIMPLEMENTED();
      }
    } else {
      OF_UNIMPLEMENTED();
    }
  } else {
    OF_UNIMPLEMENTED();
  }
}

COMMAND(Global<EagerBoxingInterpreterManager>::SetAllocated(new EagerBoxingInterpreterManager()));

}  // namespace oneflow
