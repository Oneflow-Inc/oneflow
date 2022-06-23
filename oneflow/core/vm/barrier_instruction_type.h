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
#ifndef ONEFLOW_CORE_VM_BARRIER_INSTRUCTION_TYPE_H_
#define ONEFLOW_CORE_VM_BARRIER_INSTRUCTION_TYPE_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/intrusive/flat_msg_view.h"
#include "oneflow/core/rpc/include/base.h"
#include "oneflow/core/vm/control_stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/instruction.h"
#include "oneflow/core/vm/virtual_machine_engine.h"
#include "oneflow/core/vm/barrier_phy_instr_operand.h"
#include "oneflow/core/control/global_process_ctx.h"

namespace oneflow {
namespace vm {

class BarrierInstructionType : public InstructionType {
 public:
  BarrierInstructionType() = default;
  virtual ~BarrierInstructionType() override = default;

  bool IsBarrier() const override { return true; }

  std::string DebugName(const vm::InstructionMsg& instr_msg) const override { return "Barrier"; }
  void Compute(Instruction* instruction) const override { Run(instruction->instr_msg()); }
  void ComputeInFuseMode(InstructionMsg* instr_msg) const override { Run(*instr_msg); }

 protected:
  void Run(const InstructionMsg& instr_msg) const {
    const auto* operand =
        dynamic_cast<const BarrierPhyInstrOperand*>(instr_msg.phy_instr_operand().get());
    CHECK_NOTNULL(operand)->callback();
  }
};

class GlobalSyncInstructionType : public InstructionType {
 public:
  GlobalSyncInstructionType() = default;
  virtual ~GlobalSyncInstructionType() override = default;

  bool IsBarrier() const override { return true; }

  std::string DebugName(const vm::InstructionMsg& instr_msg) const override { return "GlobalSync"; }
  void Compute(Instruction* instruction) const override { OF_ENV_BARRIER(); }
  void ComputeInFuseMode(InstructionMsg* instr_msg) const override { OF_ENV_BARRIER(); }
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_BARRIER_INSTRUCTION_TYPE_H_
