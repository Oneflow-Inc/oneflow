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
#ifndef ONEFLOW_CORE_VM_INIT_SYMBOL_INSTRUCTION_TYPE_H_
#define ONEFLOW_CORE_VM_INIT_SYMBOL_INSTRUCTION_TYPE_H_

#include "oneflow/core/object_msg/flat_msg_view.h"
#include "oneflow/core/vm/host_stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/instruction_operand.msg.h"
#include "oneflow/core/vm/symbol_storage.h"
#include "oneflow/core/vm/object_wrapper.h"

namespace oneflow {
namespace vm {

// clang-format off
FLAT_MSG_VIEW_BEGIN(SymbolInstrOperand);
  FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(InitSymbolOperand, serialized_logical_object_id);
FLAT_MSG_VIEW_END(SymbolInstrOperand);
// clang-format on

template<typename T>
class InitSymbolInstructionType final : public InstructionType {
 public:
  InitSymbolInstructionType() = default;
  ~InitSymbolInstructionType() override = default;

  using stream_type = HostStreamType;

  void Infer(Instruction* instruction) const override {
    FlatMsgView<SymbolInstrOperand> args(instruction->instr_msg().operand());
    FOR_RANGE(int, i, 0, args->serialized_logical_object_id_size()) {
      const auto& operand = args->serialized_logical_object_id(i);
      int64_t logical_object_id = operand.logical_object_id();
      const auto& symbol = Global<symbol::Storage<T>>::Get()->GetPtr(logical_object_id);
      auto* rw_mutexed_object = instruction->mut_operand_type(operand);
      rw_mutexed_object->Init<ObjectWrapper<T>>(symbol);
    }
  }
  void Compute(Instruction* instruction) const override {
    // do nothing
  }
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_INIT_SYMBOL_INSTRUCTION_TYPE_H_
