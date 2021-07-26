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
#include "oneflow/core/framework/op_interpreter/boxing/naive_b2p_boxing_interpreter.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/functional/functional.h"

namespace oneflow {

Maybe<void> NaiveB2PBoxingInterpreter::Interpret(
    const one::TensorTuple& inputs, one::TensorTuple* outputs,
    Symbol<cfg::ParallelDistribution> in_parallel_distribution,
    Symbol<cfg::ParallelDistribution> out_parallel_distribution,
    Symbol<ParallelDesc> in_parallel_desc, Symbol<ParallelDesc> out_parallel_desc) const {
  CHECK_EQ_OR_RETURN(in_parallel_desc, out_parallel_desc);
  int64_t root = JUST(in_parallel_desc->DeviceId4ParallelId(0));
  if (root == GlobalProcessCtx::LocalRank() % Global<ResourceDesc, ForEnv>::Get()->GpuDeviceNum()) {
    std::string device_type = in_parallel_desc->device_tag() == "gpu" ? "cuda" : "cpu";
    outputs->at(0) = JUST(one::functional::Copy(inputs.at(0), device_type, root));
  } else {
    outputs->at(0) = JUST(one::functional::ZerosLike(inputs.at(0)));
  }
  *outputs = inputs;
  return Maybe<void>::Ok();
}

}  // namespace oneflow
