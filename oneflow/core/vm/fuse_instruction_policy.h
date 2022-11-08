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
#ifndef ONEFLOW_CORE_VM_FUSE_INSTRUCTION_POLICY_H_
#define ONEFLOW_CORE_VM_FUSE_INSTRUCTION_POLICY_H_

#include <functional>
#include "oneflow/core/vm/instruction.h"
#include "oneflow/core/vm/instruction_policy_util.h"
#include "oneflow/core/vm/vm_object.h"

namespace oneflow {
namespace vm {

class FuseInstructionPolicy final : public InstructionPolicy {
 public:
  explicit FuseInstructionPolicy(InstructionList&& instruction_list)
      : instruction_list_(), input_dependences_(), output_dependences_() {
    instruction_list.MoveTo(&instruction_list_);
    auto ReadOnlyDepsInserter = InstructionPolicyUtil::SetInserter(&input_dependences_);
    auto WritableDepsInserter = InstructionPolicyUtil::SetInserter(&output_dependences_);
    auto* last_instruction = instruction_list_.Last();
    INTRUSIVE_UNSAFE_FOR_EACH_PTR(instruction, &instruction_list_) {
      if (instruction == last_instruction) {
        CHECK(instruction->instruction_policy().fuse_type() == kEnableInstructionFuseAsTailOnly
              || instruction->instruction_policy().fuse_type()
                     == kEnableInstructionFuseAtAnyPosition);
      } else {
        CHECK(instruction->instruction_policy().fuse_type() == kEnableInstructionFuseAtAnyPosition);
      }
      if (unlikely(stream_sequential_dependence_ == nullptr)) {
        stream_sequential_dependence_ =
            instruction->instruction_policy().stream_sequential_dependence();
      } else {
        CHECK_EQ(stream_sequential_dependence_,
                 instruction->instruction_policy().stream_sequential_dependence());
      }
      for (auto* dep : instruction->instruction_policy().input_dependences()) {
        ReadOnlyDepsInserter(dep);
      }
      for (auto* dep : instruction->instruction_policy().output_dependences()) {
        WritableDepsInserter(dep);
      }
    }
  }

  ~FuseInstructionPolicy() override = default;

  const DependenceVector& input_dependences() const override { return input_dependences_; }
  const DependenceVector& output_dependences() const override { return output_dependences_; }

  InstructionList* mut_instruction_list() { return &instruction_list_; }

 private:
  Maybe<void> Prepare(Instruction* instruction) override {
    INTRUSIVE_UNSAFE_FOR_EACH_PTR(instruction, mut_instruction_list()) {
      JUST(instruction->Prepare());
    }
    return Maybe<void>::Ok();
  }
  void Compute(Instruction* instruction) override {
    OF_PROFILER_RANGE_GUARD("F:" + instruction->DebugName());
    INTRUSIVE_UNSAFE_FOR_EACH_PTR(instruction, mut_instruction_list()) { instruction->Compute(); }
  }
  void InitInstructionStatus(Instruction* instruction) override {
    auto* last_instruction = CHECK_NOTNULL(mut_instruction_list()->Last());
    last_instruction->mut_instruction_policy()->InitInstructionStatusIf(instruction);
  }

  std::string DebugName(const Instruction&) const override { return "Fuse"; }

  InstructionList instruction_list_;
  DependenceVector input_dependences_;
  DependenceVector output_dependences_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_FUSE_INSTRUCTION_POLICY_H_
