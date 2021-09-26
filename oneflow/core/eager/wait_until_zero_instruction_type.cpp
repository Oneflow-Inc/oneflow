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
#include "oneflow/core/eager/wait_until_zero_stream_type.h"
#include "oneflow/core/eager/wait_until_zero_phy_instr_operand.h"
#include "oneflow/core/vm/stream.msg.h"
#include "oneflow/core/vm/thread_ctx.msg.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/ref_cnt_instruction_status_querier.h"

namespace oneflow {

namespace vm {

class WaitUntilZeroInstructionType final : public InstructionType {
 public:
  WaitUntilZeroInstructionType(const WaitUntilZeroInstructionType&) = delete;
  WaitUntilZeroInstructionType(WaitUntilZeroInstructionType&&) = delete;
  WaitUntilZeroInstructionType() = default;
  ~WaitUntilZeroInstructionType() = default;

  WaitUntilZeroInstructionType& operator=(const WaitUntilZeroInstructionType&) = delete;
  WaitUntilZeroInstructionType& operator=(WaitUntilZeroInstructionType&&) = delete;

  using stream_type = WaitUntilZeroStreamType;
  void Infer(vm::Instruction* instruction) const override { UNIMPLEMENTED(); }

  void Compute(vm::Instruction* instruction) const override {
    const auto* ptr = instruction->instr_msg().phy_instr_operand().get();
    const auto* phy_instr_operand = dynamic_cast<const WaitUntilZeroPhyInstrOperand*>(ptr);
    CHECK_NOTNULL(phy_instr_operand);
    auto* status_buffer_data = instruction->mut_status_buffer()->mut_buffer()->mut_data();
    auto* status_querier = RefCntInstrStatusQuerier::MutCast(status_buffer_data);
    status_querier->SetRefCntAndSetLaunched(phy_instr_operand->ref_cnt());
  }
};

COMMAND(RegisterInstructionType<WaitUntilZeroInstructionType>("WaitUntilZero"));
}  // namespace vm

}  // namespace oneflow
