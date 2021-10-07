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
#include "oneflow/core/intrusive/flat_msg_view.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/instruction_operand.h"

namespace oneflow {
namespace vm {

class LazyReferenceInstructionType : public vm::InstructionType {
 public:
  LazyReferenceInstructionType() = default;
  virtual ~LazyReferenceInstructionType() override = default;

  // clang-format off
  FLAT_MSG_VIEW_BEGIN(LazyReferenceInstruction);
    FLAT_MSG_VIEW_DEFINE_PATTERN(vm::MutOperand, eager_blob);
    FLAT_MSG_VIEW_DEFINE_PATTERN(vm::SymbolOperand, lbn_sym_id);
  FLAT_MSG_VIEW_END(LazyReferenceInstruction);
  // clang-format on

  void Infer(vm::Instruction* instruction) const override { CHECK_OK(Run(instruction)); }
  void Compute(vm::Instruction* instruction) const override{
      // do nothing
  };

 private:
  Maybe<void> Run(vm::Instruction* instruction) const;
};

class AccessBlobByCallbackInstructionType : public vm::InstructionType {
 public:
  AccessBlobByCallbackInstructionType() = default;
  ~AccessBlobByCallbackInstructionType() override = default;

  void Compute(vm::Instruction* instruction) const override;

  void Infer(vm::Instruction* instruction) const override { UNIMPLEMENTED(); }
};

class RecordEventInstructionType : public vm::InstructionType {
 public:
  RecordEventInstructionType() = default;
  ~RecordEventInstructionType() override = default;

  void Compute(vm::Instruction* instruction) const override {
    instruction->set_has_event_record(true);
  }

  void Infer(vm::Instruction* instruction) const override { UNIMPLEMENTED(); }
};

}  // namespace vm
}  // namespace oneflow
