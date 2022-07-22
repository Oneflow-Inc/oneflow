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
#ifndef ONEFLOW_CORE_VM_NAIVE_INSTRUCTION_POLICY_H_
#define ONEFLOW_CORE_VM_NAIVE_INSTRUCTION_POLICY_H_

#include "oneflow/core/vm/instruction_policy.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/phy_instr_operand.h"

namespace oneflow {
namespace vm {

class NaiveInstructionPolicy final : public InstructionPolicy {
 public:
  NaiveInstructionPolicy(const InstructionType* instruction_type,
                         const std::shared_ptr<PhyInstrOperand>& phy_instr_operand)
      : instruction_type_(instruction_type), phy_instr_operand_(phy_instr_operand) {}

  ~NaiveInstructionPolicy() override = default;

  const DependenceVector& input_dependences() const override {
    return phy_instr_operand_->input_dependences();
  }
  const DependenceVector& output_dependences() const override {
    return phy_instr_operand_->output_dependences();
  }
  Dependence* stream_sequential_dependence() const override {
    return phy_instr_operand_->stream_sequential_dependence();
  }
  void ForEachInputEagerBlobObjects(void (*DoEach)(EagerBlobObject*)) const override {
    return phy_instr_operand_->ForEachInputEagerBlobObjects(DoEach);
  }

  bool IsBarrier() const override { return instruction_type_->IsBarrier(); }
  InstructionFuseType fuse_type() const override { return instruction_type_->fuse_type(); }
  std::string DebugName(const Instruction& instruction) const override {
    return instruction_type_->DebugName(instruction);
  }

  const std::shared_ptr<PhyInstrOperand>& phy_instr_operand() const override {
    return phy_instr_operand_;
  }

 private:
  Maybe<void> Prepare(Instruction* instruction) override {
    return instruction_type_->PrepareIf(instruction);
  }
  void Compute(Instruction* instruction) override {
    return instruction_type_->ComputeIf(instruction);
  }
  void InitInstructionStatus(Instruction* instruction) override {
    return instruction_type_->InitInstructionStatusIf(instruction);
  }
  void DeleteInstructionStatus(Instruction* instruction) override {
    return instruction_type_->DeleteInstructionStatusIf(instruction);
  }

  const InstructionType* instruction_type_;
  std::shared_ptr<PhyInstrOperand> phy_instr_operand_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_NAIVE_INSTRUCTION_POLICY_H_
