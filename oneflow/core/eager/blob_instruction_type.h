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

namespace oneflow {
namespace vm {

class AccessBlobByCallbackInstructionType : public vm::InstructionType {
 public:
  AccessBlobByCallbackInstructionType() = default;
  ~AccessBlobByCallbackInstructionType() override = default;

  void Compute(vm::Instruction* instruction) const override;
  void ComputeInFuseMode(vm::InstructionMsg* instruction_msg) const override;

 private:
  void ComputeInstrMsg(const vm::InstructionMsg& instruction_msg) const;
};

class RecordEventInstructionType : public vm::InstructionType {
 public:
  RecordEventInstructionType() = default;
  ~RecordEventInstructionType() override = default;

  void Compute(vm::Instruction* instruction) const override {}
};

}  // namespace vm
}  // namespace oneflow
