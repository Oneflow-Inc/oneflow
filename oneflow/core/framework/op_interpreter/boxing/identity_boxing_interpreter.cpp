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
#include "oneflow/core/framework/op_interpreter/boxing/identity_boxing_interpreter.h"

namespace oneflow {

Maybe<one::Tensor> IdentityBoxingInterpreter::InterpretImpl(
    const std::shared_ptr<one::Tensor>& input, Symbol<cfg::ParallelDistribution> in_nd_sbp,
    Symbol<cfg::ParallelDistribution> out_nd_sbp, Symbol<ParallelDesc> in_parallel_desc,
    Symbol<ParallelDesc> out_parallel_desc) const {
  CHECK_EQ_OR_RETURN(in_nd_sbp, out_nd_sbp);
  CHECK_EQ_OR_RETURN(in_parallel_desc, out_parallel_desc);
  return input;
}

}  // namespace oneflow
