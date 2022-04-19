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
#ifndef ONEFLOW_CORE_EAGER_CALL_OPKERNEL_INSTRUCTION_H_
#define ONEFLOW_CORE_EAGER_CALL_OPKERNEL_INSTRUCTION_H_

#include "oneflow/core/vm/instr_type_id.h"
#include "oneflow/core/vm/instruction.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/memory/memory_case.pb.h"

namespace oneflow {
namespace vm {

class LocalCallOpKernelInstructionType : public vm::InstructionType {
 public:
  void Compute(vm::Instruction* instruction) const override;
  void ComputeInFuseMode(vm::InstructionMsg* instr_msg) const override;

  InstructionFuseType fuse_type() const override { return kEnableInstructionFuseAtAnyPosition; }

  std::string DebugOpTypeName(const vm::InstructionMsg& instr_msg) const override;

 protected:
  LocalCallOpKernelInstructionType() = default;
  virtual ~LocalCallOpKernelInstructionType() = default;

 private:
  Maybe<void> MaybeCompute(vm::Instruction* instruction) const;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_CALL_OPKERNEL_INSTRUCTION_H_
