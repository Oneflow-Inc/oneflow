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
#ifndef ONEFLOW_CORE_VM_ALLOCATE_INSTRUCTION_POLICY_H_
#define ONEFLOW_CORE_VM_ALLOCATE_INSTRUCTION_POLICY_H_

#include <memory>
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/vm/instruction_policy.h"
#include "oneflow/core/vm/stream.h"

namespace oneflow {

namespace vm {

class AllocateTensorInstructionPolicy final : public InstructionPolicy {
 public:
  AllocateTensorInstructionPolicy(const EagerBlobObjectList& eager_blob_objects,
                                  vm::Stream* vm_stream);
  AllocateTensorInstructionPolicy(const AllocateTensorInstructionPolicy&) = delete;
  AllocateTensorInstructionPolicy(AllocateTensorInstructionPolicy&&) = delete;

  ~AllocateTensorInstructionPolicy() override = default;

  const DependenceVector& input_dependences() const override {
    static thread_local DependenceVector input_dependences{};
    return input_dependences;
  }
  const DependenceVector& output_dependences() const override { return output_dependences_; }

  InstructionFuseType fuse_type() const override { return kEnableInstructionFuseAtAnyPosition; }

  std::string DebugName(const vm::Instruction& instruction) const override;

 private:
  Maybe<void> Prepare(Instruction* instruction) override { return Maybe<void>::Ok(); }
  void Compute(Instruction* instruction) override;

  EagerBlobObjectList eager_blob_objects_;
  DependenceVector output_dependences_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_ALLOCATE_INSTRUCTION_POLICY_H_
