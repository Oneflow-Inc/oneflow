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
#include "oneflow/core/framework/op_interpreter/boxing/cuda_copy_boxing_interpreter.h"
#include "oneflow/core/framework/op_interpreter/boxing/eager_boxing_interpreter_mgr.h"
#include "oneflow/core/framework/op_interpreter/boxing/eager_boxing_interpreter_util.h"
#include "oneflow/core/framework/op_interpreter/boxing/collective_boxing_interpreter.h"
#include "oneflow/core/framework/op_interpreter/boxing/identity_boxing_interpreter.h"
#include "oneflow/core/framework/op_interpreter/boxing/naive_b2p_boxing_interpreter.h"
#include "oneflow/core/framework/op_interpreter/boxing/naive_s2p_boxing_interpreter.h"
#include "oneflow/core/framework/op_interpreter/boxing/cuda_based_cpu_mpi_boxing_interpreter.h"

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
  return JUST(MapAt(sbp_pair2eager_boxing_interpreter,
                    std::make_pair(in_nd_sbp->sbp_parallel(0), out_nd_sbp->sbp_parallel(0))));
}

Maybe<Optional<EagerBoxingInterpreter>> GetCudaBasedCpuMpiBoxingInterpreter(
    Symbol<cfg::ParallelDistribution> in_nd_sbp, Symbol<cfg::ParallelDistribution> out_nd_sbp,
    Symbol<ParallelDesc> in_parallel_desc, Symbol<ParallelDesc> out_parallel_desc) {
  CHECK_OR_RETURN(in_nd_sbp != out_nd_sbp);
  const auto& gpu_in_parallel_desc = JUST(ReplaceDeviceType(in_parallel_desc, DeviceType::kGPU));
  const auto& gpu_out_parallel_desc = JUST(ReplaceDeviceType(out_parallel_desc, DeviceType::kGPU));
  CHECK_OR_RETURN(gpu_in_parallel_desc == gpu_out_parallel_desc);
  const auto& gpu_boxing_interpreter =
      TRY(GetOneDimNcclCollectiveEagerBoxingInterpreter(in_nd_sbp, out_nd_sbp));
  if (gpu_boxing_interpreter.IsOk()) {
    return Optional<EagerBoxingInterpreter>(
        std::shared_ptr<EagerBoxingInterpreter>(new CudaBasedCpuMpiBoxingInterpreter()));
  } else {
    return Optional<EagerBoxingInterpreter>();
  }
}

Maybe<bool> IgnoringDeviceTypeEqual(Symbol<ParallelDesc> lhs, Symbol<ParallelDesc> rhs) {
  if (lhs == rhs) { return true; }
  return lhs == JUST(ReplaceDeviceType(rhs, lhs->device_type()));
}

Maybe<EagerBoxingInterpreter> GetBoxingInterpreter(Symbol<cfg::ParallelDistribution> in_nd_sbp,
                                                   Symbol<cfg::ParallelDistribution> out_nd_sbp,
                                                   Symbol<ParallelDesc> in_parallel_desc,
                                                   Symbol<ParallelDesc> out_parallel_desc) {
  if (in_nd_sbp == out_nd_sbp && in_parallel_desc == out_parallel_desc) {
    return std::shared_ptr<EagerBoxingInterpreter>(new IdentityBoxingInterpreter());
  } else if (in_nd_sbp->sbp_parallel_size() == 1 && out_nd_sbp->sbp_parallel_size() == 1) {
    if (in_parallel_desc == out_parallel_desc) {
      if (EagerBoxingInterpreterUtil::IsBoxingB2P(in_nd_sbp->sbp_parallel(0),
                                                  out_nd_sbp->sbp_parallel(0))) {
        return std::shared_ptr<EagerBoxingInterpreter>(new NaiveB2PBoxingInterpreter());
      } else if (in_parallel_desc->device_type() == DeviceType::kGPU) {
        return GetOneDimNcclCollectiveEagerBoxingInterpreter(in_nd_sbp, out_nd_sbp);
      } else if (in_parallel_desc->device_type() == DeviceType::kCPU) {
        const auto& opt_interpreter = JUST(GetCudaBasedCpuMpiBoxingInterpreter(
            in_nd_sbp, out_nd_sbp, in_parallel_desc, out_parallel_desc));
        if (opt_interpreter->has_value()) { return opt_interpreter->value(); }
      }
    } else if (JUST(IgnoringDeviceTypeEqual(in_parallel_desc, out_parallel_desc))) {
      if ((in_parallel_desc->device_type() == DeviceType::kGPU
           && out_parallel_desc->device_type() == DeviceType::kCPU)
          || (in_parallel_desc->device_type() == DeviceType::kCPU
              && out_parallel_desc->device_type() == DeviceType::kGPU)) {
        if (in_nd_sbp == out_nd_sbp) {
          return std::shared_ptr<EagerBoxingInterpreter>(new CudaCopyBoxingInterpreter());
        } else {
          const auto& opt_interpreter = JUST(GetCudaBasedCpuMpiBoxingInterpreter(
              in_nd_sbp, out_nd_sbp, in_parallel_desc, out_parallel_desc));
          if (opt_interpreter->has_value()) { return opt_interpreter->value(); }
        }
      }
    }
  }
  UNIMPLEMENTED_THEN_RETURN();
}

static constexpr auto* CachedGetBoxingInterpreter = DECORATE(&GetBoxingInterpreter, ThreadLocal);

}  // namespace

Maybe<EagerBoxingInterpreter> EagerBoxingInterpreterManager::GetEagerBoxingInterpreter(
    Symbol<cfg::ParallelDistribution> in_nd_sbp, Symbol<cfg::ParallelDistribution> out_nd_sbp,
    Symbol<ParallelDesc> in_parallel_desc, Symbol<ParallelDesc> out_parallel_desc) const {
  return CachedGetBoxingInterpreter(in_nd_sbp, out_nd_sbp, in_parallel_desc, out_parallel_desc);
}

COMMAND(Global<EagerBoxingInterpreterManager>::SetAllocated(new EagerBoxingInterpreterManager()));

}  // namespace oneflow
