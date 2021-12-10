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

#include "oneflow/core/eager/critical_section_stream_type.h"
#include "oneflow/core/eager/critical_section_status_querier.h"
#include "oneflow/core/eager/critical_section_phy_instr_operand.h"
#include "oneflow/core/framework/nn_graph_if.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/vm/instruction.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/job/job_instance.h"
#include "oneflow/core/common/buffer_manager.h"
#include "oneflow/core/common/global.h"
#include "oneflow/core/vm/stream.h"
#include "oneflow/core/vm/thread_ctx.h"
#include "oneflow/core/register/ofblob.h"
#include "oneflow/core/vm/ref_cnt_instruction_status_querier.h"
#include "oneflow/core/profiler/profiler.h"

namespace oneflow {

namespace vm {

class CriticalSectionBeginInstructionType final : public InstructionType {
 public:
  CriticalSectionBeginInstructionType(const CriticalSectionBeginInstructionType&) = delete;
  CriticalSectionBeginInstructionType(CriticalSectionBeginInstructionType&&) = delete;
  CriticalSectionBeginInstructionType& operator=(const CriticalSectionBeginInstructionType&) =
      delete;
  CriticalSectionBeginInstructionType& operator=(CriticalSectionBeginInstructionType&&) = delete;
  CriticalSectionBeginInstructionType() = default;
  ~CriticalSectionBeginInstructionType() = default;

  using stream_type = CriticalSectionStreamType;

  void Infer(vm::Instruction* instruction) const override { UNIMPLEMENTED(); }

  void Compute(vm::Instruction* instruction) const override {
    auto* status_buffer_data = instruction->mut_status_buffer()->mut_buffer()->mut_data();
    auto* status_querier = CriticalSectionStatusQuerier::MutCast(status_buffer_data);
    status_querier->SetLaunched(std::make_shared<NaiveEventRecord>());
  }
};

COMMAND(RegisterInstructionType<CriticalSectionBeginInstructionType>("CriticalSectionBegin"));

class CriticalSectionEndInstructionType final : public InstructionType {
 public:
  CriticalSectionEndInstructionType(const CriticalSectionEndInstructionType&) = delete;
  CriticalSectionEndInstructionType(CriticalSectionEndInstructionType&&) = delete;
  CriticalSectionEndInstructionType& operator=(const CriticalSectionEndInstructionType&) = delete;
  CriticalSectionEndInstructionType& operator=(CriticalSectionEndInstructionType&&) = delete;
  CriticalSectionEndInstructionType() = default;
  ~CriticalSectionEndInstructionType() = default;

  using stream_type = CriticalSectionStreamType;

  void Infer(vm::Instruction* instruction) const override { UNIMPLEMENTED(); }

  void Compute(vm::Instruction* instruction) const override {
    const auto* ptr = instruction->instr_msg().phy_instr_operand().get();
    const auto* phy_instr_operand = dynamic_cast<const CriticalSectionEndPhyInstrOperand*>(ptr);
    CHECK_NOTNULL(phy_instr_operand);
    auto* status_buffer_data = instruction->mut_status_buffer()->mut_buffer()->mut_data();
    auto* status_querier = CriticalSectionStatusQuerier::MutCast(status_buffer_data);
    status_querier->SetLaunched(phy_instr_operand->event_record());
  }
};

COMMAND(RegisterInstructionType<CriticalSectionEndInstructionType>("CriticalSectionEnd"));

}  // namespace vm
}  // namespace oneflow
