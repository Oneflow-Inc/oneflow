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
#ifndef ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_BOXING_EAGER_BOXING_INTERPRETER_H_
#define ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_BOXING_EAGER_BOXING_INTERPRETER_H_

#include "oneflow/core/common/symbol.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/sbp_parallel.cfg.h"

namespace oneflow {

class EagerBoxingInterpreter {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EagerBoxingInterpreter);
  EagerBoxingInterpreter() = default;
  virtual ~EagerBoxingInterpreter() = default;

  virtual Maybe<one::Tensor> Interpret(const std::shared_ptr<one::Tensor>& input,
                                       Symbol<cfg::ParallelDistribution> in_parallel_distribution,
                                       Symbol<cfg::ParallelDistribution> out_parallel_distribution,
                                       Symbol<ParallelDesc> in_parallel_desc,
                                       Symbol<ParallelDesc> out_parallel_desc) const = 0;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_OP_INTERPRETER_BOXING_EAGER_BOXING_INTERPRETER_H_
