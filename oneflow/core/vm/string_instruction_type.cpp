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
#include "oneflow/core/object_msg/flat_msg_view.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/vm/host_stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/instruction_operand.msg.h"
#include "oneflow/core/vm/string_object.h"
#include "oneflow/core/vm/string_symbol.h"
#include "oneflow/core/vm/symbol_storage.h"

namespace oneflow {
namespace vm {

COMMAND(Global<symbol::Storage<StringSymbol>>::SetAllocated(new symbol::Storage<StringSymbol>()));

namespace {

// clang-format off
FLAT_MSG_VIEW_BEGIN(StringObjectInstrOperand);
  FLAT_MSG_VIEW_DEFINE_OPERAND_LIST(vm::InitSymbolOperand, string);
FLAT_MSG_VIEW_END(StringObjectInstrOperand);
// clang-format on

}  // namespace

class InitStringSymbolInstructionType final : public InstructionType {
 public:
  InitStringSymbolInstructionType() = default;
  ~InitStringSymbolInstructionType() override = default;

  using stream_type = HostStreamType;

  void Infer(Instruction* instruction) const override {
    FlatMsgView<StringObjectInstrOperand> args(instruction->instr_msg().operand());
    FOR_RANGE(int, i, 0, args->string_size()) {
      int64_t logical_object_id = args->string(i).logical_object_id();
      const auto& str_sym = Global<symbol::Storage<StringSymbol>>::Get()->Get(logical_object_id);
      auto* rw_mutexed_object = instruction->mut_operand_type(args->string(i));
      rw_mutexed_object->Init<StringObject>(str_sym.data());
    }
  }
  void Compute(Instruction* instruction) const override {
    // do nothing
  }
};
COMMAND(RegisterInstructionType<InitStringSymbolInstructionType>("InitStringSymbol"));

}  // namespace vm
}  // namespace oneflow
