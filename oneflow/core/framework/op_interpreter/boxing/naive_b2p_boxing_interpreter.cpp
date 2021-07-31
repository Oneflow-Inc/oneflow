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

Maybe<one::Tensor> NaiveB2PBoxingInterpreter::Interpret(
    const std::shared_ptr<one::Tensor>& input,
    Symbol<cfg::ParallelDistribution> in_parallel_distribution,
    Symbol<cfg::ParallelDistribution> out_parallel_distribution,
    Symbol<ParallelDesc> in_parallel_desc, Symbol<ParallelDesc> out_parallel_desc) const {
  CHECK_EQ_OR_RETURN(in_parallel_desc, out_parallel_desc);
  int64_t root = JUST(in_parallel_desc->DeviceId4ParallelId(0));
  if (root == GlobalProcessCtx::LocalRank() % Global<ResourceDesc, ForEnv>::Get()->GpuDeviceNum()) {
    std::string device_type = in_parallel_desc->device_tag() == "gpu" ? "cuda" : "cpu";
    return JUST(one::functional::Copy(input, device_type, root));
  } else {
    // use SliceUpdate because ZerosLike does not have backward
    int64_t num_axes = input->shape()->NumAxes();
    std::vector<int64_t> start(num_axes, 0);
    std::vector<int64_t> stop(num_axes);
    std::vector<int64_t> step(num_axes, 1);
    {
      for (int64_t i = 0; i < num_axes; ++i) { stop.at(i) = input->shape()->At(i); }
    }
    return JUST(one::functional::SliceUpdate(input, JUST(one::functional::ZerosLike(input)), start,
                                             stop, step));
  }
}

}  // namespace oneflow
