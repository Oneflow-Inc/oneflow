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
#ifndef ONEFLOW_CORE_VM_TOUCH_TENSORS_INSTRUCTION_POLICY_H_
#define ONEFLOW_CORE_VM_TOUCH_TENSORS_INSTRUCTION_POLICY_H_

#include "oneflow/core/vm/instruction_policy.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/vm/instruction_policy_util.h"

namespace oneflow {
namespace vm {

class TouchTensorsInstructionPolicy final : public InstructionPolicy {
 public:
  explicit TouchTensorsInstructionPolicy(const vm::EagerBlobObjectList& eager_blob_objects)
      : eager_blob_objects_(eager_blob_objects) {
    const auto& Insert = InstructionPolicyUtil::SetInserter(&input_dependences_);
    for (const auto& eager_blob_object : eager_blob_objects_) {
      Insert(CHECK_JUST(eager_blob_object->compute_local_dep_object()));
    }
  }
  ~TouchTensorsInstructionPolicy() = default;

  const DependenceVector& input_dependences() const override { return input_dependences_; }
  const DependenceVector& output_dependences() const override {
    static DependenceVector empty{};
    return empty;
  }

  std::string DebugName(const vm::Instruction& instruction) const override {
    return "TouchTensors";
  }
  Maybe<void> Prepare(vm::Instruction* instruction) override { return Maybe<void>::Ok(); }
  void Compute(vm::Instruction* instruction) override {}

 private:
  vm::EagerBlobObjectList eager_blob_objects_;
  DependenceVector input_dependences_;
};

}  // namespace vm
}  // namespace oneflow
#endif  // ONEFLOW_CORE_VM_TOUCH_TENSORS_INSTRUCTION_POLICY_H_
