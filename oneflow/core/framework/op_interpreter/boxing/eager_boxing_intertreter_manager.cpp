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
#include "oneflow/core/framework/op_interpreter/boxing/eager_boxing_interpreter_manager.h"
#include "oneflow/core/framework/op_interpreter/boxing/eager_boxing_interpreter_util.h"
#include "oneflow/core/framework/op_interpreter/boxing/collective_boxing_interpreter.h"

namespace oneflow {

namespace {
using SbpPair2EagerBoxingInterpreter = HashMap<std::pair<cfg::SbpParallel, cfg::SbpParallel>, std::shared_ptr<EagerBoxingInterpreter>>;

Maybe<EagerBoxingInterpreter> GetOneDimNcclCollectiveEagerBoxingInterpreter(Symbol<cfg::ParallelDistribution> in_parallel_distribution,
    Symbol<cfg::ParallelDistribution> out_parallel_distribution) {
  static SbpPair2EagerBoxingInterpreter sbp_pair2eager_boxing_interpreter = {
    {std::make_pair(), std::make_shared<>},
  }; 
}

}  // namespace

Maybe<EagerBoxingInterpreter> EagerBoxingInterpreterManager::GetEagerBoxingInterpreter(
    Symbol<cfg::ParallelDistribution> in_parallel_distribution,
    Symbol<cfg::ParallelDistribution> out_parallel_distribution,
    Symbol<ParallelDesc> in_parallel_desc, Symbol<ParallelDesc> out_parallel_desc) {
  if (in_parallel_distribution->sbp_parallel_size() == 1
      && out_parallel_distribution->sbp_parallel_size() == 1) {
    if (EagerBoxingInterpreterUtil::IsPlacementSymmetrical(in_parallel_desc, out_parallel_desc)) {
      if (EagerBoxingInterpreterUtil::IsDeviceTypeGPU(in_parallel_desc)) {
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

}  // namespace oneflow

namespace std {
template<>
struct hash<std::pair<oneflow::cfg::SbpParallel, oneflow::cfg::SbpParallel>> {
  size_t operator()(const std::pair<oneflow::cfg::SbpParallel, oneflow::cfg::SbpParallel>& sbp_pair) const {
    return std::hash<oneflow::cfg::SbpParallel>()(sbp_pair.first) ^ std::hash<oneflow::cfg::SbpParallel>()(sbp_pair.second);
  }
};
}
