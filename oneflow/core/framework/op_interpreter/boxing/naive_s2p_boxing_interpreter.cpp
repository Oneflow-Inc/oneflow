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
#include "oneflow/core/framework/op_interpreter/boxing/naive_s2p_boxing_interpreter.h"
#include "oneflow/core/framework/op_interpreter/boxing/naive_b2p_boxing_interpreter.h"
#include "oneflow/core/framework/op_interpreter/boxing/collective_boxing_interpreter.h"
#include "oneflow/core/framework/op_interpreter/boxing/eager_boxing_interpreter_util.h"

namespace oneflow {

namespace {

Maybe<Symbol<cfg::ParallelDistribution>> GetBroadcastParallelDistribution() {
  cfg::ParallelDistribution broadcast_parallel_distribution;
  broadcast_parallel_distribution.mutable_sbp_parallel()->Add()->mutable_broadcast_parallel();
  return SymbolOf(broadcast_parallel_distribution);
}

}  // namespace

Maybe<one::Tensor> NcclS2PBoxingInterpreter::InterpretImpl(
    const std::shared_ptr<one::Tensor>& input,
    Symbol<cfg::ParallelDistribution> in_parallel_distribution,
    Symbol<cfg::ParallelDistribution> out_parallel_distribution,
    Symbol<ParallelDesc> in_parallel_desc, Symbol<ParallelDesc> out_parallel_desc) const {
  CHECK_OR_RETURN(EagerBoxingInterpreterUtil::IsBoxingS2P(
      in_parallel_distribution->sbp_parallel(0), out_parallel_distribution->sbp_parallel(0)));
  CHECK_EQ_OR_RETURN(in_parallel_desc, out_parallel_desc);
  static thread_local Symbol<cfg::ParallelDistribution> mid_parallel_distribution =
      JUST(GetBroadcastParallelDistribution());
  static thread_local std::shared_ptr<NcclCollectiveAllGatherBoxingInterpreter> s2b_interpreter =
      std::make_shared<NcclCollectiveAllGatherBoxingInterpreter>();
  static thread_local std::shared_ptr<NaiveB2PBoxingInterpreter> b2p_interpreter =
      std::make_shared<NaiveB2PBoxingInterpreter>();
  const auto& mid_tesnor =
      JUST(s2b_interpreter->Interpret(input, in_parallel_distribution, mid_parallel_distribution,
                                      in_parallel_desc, out_parallel_desc));
  return JUST(b2p_interpreter->Interpret(mid_tesnor, mid_parallel_distribution,
                                         out_parallel_distribution, in_parallel_desc,
                                         out_parallel_desc));
}

}  // namespace oneflow
