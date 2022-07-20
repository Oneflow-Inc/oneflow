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
#ifndef ONEFLOW_CORE_EAGER_TOUCH_TENSORS_INSTRUCTION_TYPE_H_
#define ONEFLOW_CORE_EAGER_TOUCH_TENSORS_INSTRUCTION_TYPE_H_

#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/phy_instr_operand.h"
#include "oneflow/core/eager/eager_blob_object.h"

namespace oneflow {
namespace vm {

class Instruction;

class TouchTensorsPhyInstrOperand final : public PhyInstrOperand {
 public:
  TouchTensorsPhyInstrOperand(const vm::EagerBlobObjectList& eager_blob_objects);

  const DependenceVector& input_dependences() const override { return input_dependences_; }
  const DependenceVector& output_dependences() const override {
    static DependenceVector empty{};
    return empty;
  }

  void ForEachInputEagerBlobObjects(void (*DoEach)(EagerBlobObject*)) const override {
    for (const auto& eager_blob_object : eager_blob_objects_) { DoEach(eager_blob_object.get()); }
  }

 private:
  vm::EagerBlobObjectList eager_blob_objects_;
  DependenceVector input_dependences_;
};

class TouchTensorsInstructionType final : public InstructionType {
 public:
  TouchTensorsInstructionType() = default;
  ~TouchTensorsInstructionType() override = default;

  std::string DebugName(const vm::Instruction& instruction) const override {
    return "TouchTensors";
  }
  Maybe<void> Prepare(vm::Instruction* instruction) const override { return Maybe<void>::Ok(); }
  void Compute(vm::Instruction* instruction) const override {}
};

}  // namespace vm
}  // namespace oneflow
#endif  // ONEFLOW_CORE_EAGER_TOUCH_TENSORS_INSTRUCTION_TYPE_H_
