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
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/host_stream_type.h"
#include "oneflow/core/eager/opkernel_instruction.msg.h"
#include "oneflow/core/job/foreign_callback.h"

namespace oneflow {

namespace eager {

class RemoveForeignCallbackInstructionType : public vm::InstructionType {
 public:
  RemoveForeignCallbackInstructionType() = default;
  ~RemoveForeignCallbackInstructionType() override = default;

  using stream_type = vm::HostStreamType;

  void Infer(vm::Instruction* instruction) const override {
    // do nothing
  }

  void Compute(vm::Instruction* instruction) const override {
    FlatMsgView<RemoveForeignCallbackInstrOperand> args(instruction->instr_msg().operand());
    (*Global<std::shared_ptr<ForeignCallback>>::Get())
        ->RemoveForeignCallback(args->unique_callback_id());
  }
};

COMMAND(vm::RegisterInstructionType<RemoveForeignCallbackInstructionType>("RemoveForeignCallback"));

}  // namespace eager

}  // namespace oneflow
