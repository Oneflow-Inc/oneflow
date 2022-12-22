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
#ifndef ONEFLOW_CORE_VM_BARRIER_INSTRUCTION_POLICY_H_
#define ONEFLOW_CORE_VM_BARRIER_INSTRUCTION_POLICY_H_

#include "oneflow/core/vm/instruction_policy.h"
namespace oneflow {
namespace vm {

class BarrierInstructionPolicy final : public InstructionPolicy {
 public:
  BarrierInstructionPolicy(const std::function<void()>& callback) : callback_(callback) {
    stream_sequential_dependence_ = nullptr;
  }
  ~BarrierInstructionPolicy() override = default;

  const DependenceVector& input_dependences() const override {
    static DependenceVector dependences{};
    return dependences;
  }
  const DependenceVector& output_dependences() const override {
    static DependenceVector dependences{};
    return dependences;
  }

  bool IsBarrier() const override { return true; }

  std::string DebugName(const vm::Instruction& instruction) const override { return "Barrier"; }
  Maybe<void> Prepare(Instruction* instruction) override { return Maybe<void>::Ok(); }
  void Compute(Instruction* instruction) override { return callback_(); }

 private:
  std::function<void()> callback_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_BARRIER_INSTRUCTION_POLICY_H_
