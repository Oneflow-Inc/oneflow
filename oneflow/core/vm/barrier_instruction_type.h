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
#include "oneflow/core/rpc/include/base.h"
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

  std::string DebugName(const vm::Instruction& instruction) const override { return "Barrier"; }
  Maybe<void> Prepare(Instruction* instruction) const override { return Maybe<void>::Ok(); }
  void Compute(Instruction* instruction) const override { Run(*instruction); }

 protected:
  void Run(const Instruction& instruction) const {
    const auto& phy_instr_operand = instruction.phy_instr_operand();
    const auto* operand =
        CHECK_NOTNULL(dynamic_cast<const BarrierPhyInstrOperand*>(phy_instr_operand.get()));
    operand->callback();
  }
};

class GlobalSyncInstructionType : public InstructionType {
 public:
  GlobalSyncInstructionType() = default;
  virtual ~GlobalSyncInstructionType() override = default;

  bool IsBarrier() const override { return true; }

  std::string DebugName(const Instruction& instruction) const override { return "GlobalSync"; }
  Maybe<void> Prepare(Instruction* instruction) const override { return Maybe<void>::Ok(); }
  void Compute(Instruction* instruction) const override { OF_ENV_BARRIER(); }
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_BARRIER_INSTRUCTION_TYPE_H_
