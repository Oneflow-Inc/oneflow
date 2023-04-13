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
#ifndef ONEFLOW_CORE_VM_INSTRUCTION_POLICY_UTIL_H_
#define ONEFLOW_CORE_VM_INSTRUCTION_POLICY_UTIL_H_

#include <functional>
#include <set>
#include "oneflow/core/vm/vm_object.h"

namespace oneflow {
namespace vm {

struct InstructionPolicyUtil {
  static std::function<void(Dependence*)> SetInserter(DependenceVector* dependences) {
    auto existed =
        std::make_shared<std::set<Dependence*>>(dependences->begin(), dependences->end());
    return [dependences, existed](Dependence* object) {
      if (existed->insert(object).second) { dependences->push_back(object); }
    };
  }
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_INSTRUCTION_POLICY_UTIL_H_
