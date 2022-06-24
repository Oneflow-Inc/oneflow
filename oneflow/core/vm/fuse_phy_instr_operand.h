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
#ifndef ONEFLOW_CORE_VM_FUSE_PHY_INSTR_OPERAND_H_
#define ONEFLOW_CORE_VM_FUSE_PHY_INSTR_OPERAND_H_

#include <functional>
#include "oneflow/core/vm/phy_instr_operand.h"
#include "oneflow/core/vm/instruction.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/eager/local_dep_object.h"

namespace oneflow {
namespace vm {

class FusePhyInstrOperand : public PhyInstrOperand {
 public:
  explicit FusePhyInstrOperand(InstructionList&& instruction_list)
      : instruction_list_(), input_dependences_(), output_dependences_() {
    instruction_list.MoveTo(&instruction_list_);
    auto ReadOnlyDepsInserter = SetInserter(&input_dependences_);
    auto WritableDepsInserter = SetInserter(&output_dependences_);
    auto* last_instruction = instruction_list_.Last();
    INTRUSIVE_UNSAFE_FOR_EACH_PTR(instruction, &instruction_list_) {
      if (instruction == last_instruction) {
        CHECK(instruction->instruction_type().fuse_type() == kEnableInstructionFuseAsTailOnly
              || instruction->instruction_type().fuse_type()
                     == kEnableInstructionFuseAtAnyPosition);
      } else {
        CHECK(instruction->instruction_type().fuse_type() == kEnableInstructionFuseAtAnyPosition);
      }
      if (unlikely(stream_sequential_dependence_ == nullptr)) {
        stream_sequential_dependence_ =
            instruction->phy_instr_operand()->stream_sequential_dependence();
      } else {
        CHECK_EQ(stream_sequential_dependence_,
                 instruction->phy_instr_operand()->stream_sequential_dependence());
      }
      for (auto* dep : instruction->phy_instr_operand()->input_dependences()) {
        ReadOnlyDepsInserter(dep);
      }
      for (auto* dep : instruction->phy_instr_operand()->output_dependences()) {
        WritableDepsInserter(dep);
      }
    }
  }
  ~FusePhyInstrOperand() override = default;

  const DependenceVector& input_dependences() const override { return input_dependences_; }
  const DependenceVector& output_dependences() const override { return output_dependences_; }

  InstructionList* mut_instruction_list() { return &instruction_list_; }

 private:
  InstructionList instruction_list_;
  DependenceVector input_dependences_;
  DependenceVector output_dependences_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_FUSE_PHY_INSTR_OPERAND_H_
