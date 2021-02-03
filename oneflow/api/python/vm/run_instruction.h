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
#ifndef ONEFLOW_API_PYTHON_VM_RUN_INSTRUCTION_H_
#define ONEFLOW_API_PYTHON_VM_RUN_INSTRUCTION_H_

#include "oneflow/core/common/global.h"
#include "oneflow/core/eager/eager_oneflow.h"
#include "oneflow/core/eager/eager_symbol.cfg.h"

namespace oneflow {

inline void RunLogicalInstruction(
    const std::shared_ptr<vm::cfg::InstructionListProto>& cfg_instruction_list,
    const std::shared_ptr<eager::cfg::EagerSymbolList>& cfg_eager_symbol_list) {
  return Global<eager::EagerOneflow>::Get()
      ->RunLogicalInstruction(cfg_instruction_list, cfg_eager_symbol_list)
      .GetOrThrow();
}

inline void RunPhysicalInstruction(
    const std::shared_ptr<vm::cfg::InstructionListProto>& cfg_instruction_list,
    const std::shared_ptr<eager::cfg::EagerSymbolList>& cfg_eager_symbol_list) {
  return Global<eager::EagerOneflow>::Get()
      ->RunPhysicalInstruction(cfg_instruction_list, cfg_eager_symbol_list)
      .GetOrThrow();
}

}  // namespace oneflow

#endif  // ONEFLOW_API_PYTHON_VM_RUN_INSTRUCTION_H_
