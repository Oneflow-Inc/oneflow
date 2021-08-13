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
#include "oneflow/core/framework/op_interpreter/boxing/eager_boxing_interpreter.h"
#include "oneflow/core/framework/op_interpreter/boxing/eager_boxing_interpreter_mgr.h"

namespace oneflow {

Maybe<EagerBoxingCall> EagerBoxingCall::New(Symbol<cfg::ParallelDistribution> in_nd_sbp,
                                            Symbol<cfg::ParallelDistribution> out_nd_sbp,
                                            Symbol<ParallelDesc> in_parallel_desc,
                                            Symbol<ParallelDesc> out_parallel_desc) {
  const auto* mgr = Global<EagerBoxingInterpreterManager>::Get();
  const auto& boxing_interpreter = JUST(
      mgr->GetEagerBoxingInterpreter(in_nd_sbp, out_nd_sbp, in_parallel_desc, out_parallel_desc));
  return std::shared_ptr<EagerBoxingCall>(new EagerBoxingCall{
      .boxing_interpreter = boxing_interpreter,
      .in_nd_sbp = in_nd_sbp,
      .out_nd_sbp = out_nd_sbp,
      .in_parallel_desc = in_parallel_desc,
      .out_parallel_desc = out_parallel_desc,
  });
}

Maybe<one::Tensor> EagerBoxingCall::Apply(const std::shared_ptr<one::Tensor>& input) const {
  CHECK_OR_RETURN(JUST(input->nd_sbp()) == this->in_nd_sbp);
  CHECK_OR_RETURN(JUST(input->parallel_desc()) == this->in_parallel_desc);
  return this->boxing_interpreter->Interpret(input, this->in_nd_sbp, this->out_nd_sbp,
                                             this->in_parallel_desc, this->out_parallel_desc);
}

}  // namespace oneflow
