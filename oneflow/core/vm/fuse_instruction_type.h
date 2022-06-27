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
#ifndef ONEFLOW_CORE_VM_FUSE_INSTRUCTION_TYPE_H_
#define ONEFLOW_CORE_VM_FUSE_INSTRUCTION_TYPE_H_

#include "oneflow/core/vm/instruction.h"
#include "oneflow/core/vm/fuse_phy_instr_operand.h"
#include "oneflow/core/profiler/profiler.h"

namespace oneflow {

namespace vm {

class FuseInstructionType : public vm::InstructionType {
 public:
  FuseInstructionType() = default;
  ~FuseInstructionType() override = default;

  std::string DebugName(const Instruction&) const override { return "Fuse"; }

  void InitInstructionStatus(Instruction* instruction) const override {
    const auto& phy_instr_operand = instruction->phy_instr_operand();
    auto* ptr = dynamic_cast<vm::FusePhyInstrOperand*>(phy_instr_operand.get());
    auto* instruction_list = CHECK_NOTNULL(ptr)->mut_instruction_list();
    auto* last_instruction = CHECK_NOTNULL(instruction_list->Last());
    last_instruction->instruction_type().InitInstructionStatusIf(instruction);
  }

  Maybe<void> Prepare(vm::Instruction* instruction) const override {
    const auto& phy_instr_operand = instruction->phy_instr_operand();
    auto* ptr = dynamic_cast<vm::FusePhyInstrOperand*>(phy_instr_operand.get());
    CHECK_NOTNULL_OR_RETURN(ptr);
    auto* instruction_list = ptr->mut_instruction_list();
    INTRUSIVE_UNSAFE_FOR_EACH_PTR(instruction, instruction_list) {
      JUST(instruction->instruction_type().PrepareIf(instruction));
    }
    return Maybe<void>::Ok();
  }
  void Compute(vm::Instruction* instruction) const override {
    const auto& phy_instr_operand = instruction->phy_instr_operand();
    auto* ptr = dynamic_cast<vm::FusePhyInstrOperand*>(phy_instr_operand.get());
    auto* instruction_list = CHECK_NOTNULL(ptr)->mut_instruction_list();
    OF_PROFILER_RANGE_GUARD("F:" + instruction->DebugName());
    INTRUSIVE_UNSAFE_FOR_EACH_PTR(instruction, instruction_list) { instruction->Compute(); }
  }
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_FUSE_INSTRUCTION_TYPE_H_
