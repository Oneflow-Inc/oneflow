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
#ifndef ONEFLOW_CORE_FRAMEWORK_BOXING_EAGER_BOXING_INTERPRETER_MANAGER_H_
#define ONEFLOW_CORE_FRAMEWORK_BOXING_EAGER_BOXING_INTERPRETER_MANAGER_H_

#include "oneflow/core/framework/op_interpreter/boxing/eager_boxing_interpreter.h"

namespace oneflow {

class EagerBoxingInterpreterManager final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EagerBoxingInterpreterManager);
  EagerBoxingInterpreterManager() = default;
  virtual ~EagerBoxingInterpreterManager() = default;

  Maybe<EagerBoxingInterpreter> GetEagerBoxingInterpreter(
      Symbol<cfg::ParallelDistribution> in_parallel_distribution,
      Symbol<cfg::ParallelDistribution> out_parallel_distribution,
      Symbol<ParallelDesc> in_parallel_desc, Symbol<ParallelDesc> out_parallel_desc) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_BOXING_EAGER_BOXING_INTERPRETER_MANAGER_H_
