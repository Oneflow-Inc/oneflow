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
#include "oneflow/core/eager/critical_section_phy_instr_operand.h"
#include "oneflow/core/framework/nn_graph_if.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/job/job_instance.h"
#include "oneflow/core/common/buffer_manager.h"
#include "oneflow/core/common/global.h"
#include "oneflow/core/vm/stream.msg.h"
#include "oneflow/core/vm/thread_ctx.msg.h"
#include "oneflow/core/register/ofblob.h"
#include "oneflow/core/vm/ref_cnt_instruction_status_querier.h"
#include "oneflow/core/profiler/profiler.h"

namespace oneflow {

namespace vm {

template<typename PhyInstrOperandT>
class CriticalSectionInstructionType final : public InstructionType { // NOLINT
 public:
  CriticalSectionInstructionType(const CriticalSectionInstructionType&) = delete;
  CriticalSectionInstructionType(CriticalSectionInstructionType&&) = delete;
  CriticalSectionInstructionType() = default;
  ~CriticalSectionInstructionType() = default;

  using stream_type = CriticalSectionStreamType;

  void Infer(vm::Instruction* instruction) const override { UNIMPLEMENTED(); }

  void Compute(vm::Instruction* instruction) const override {
    const auto* ptr = instruction->instr_msg().phy_instr_operand().get();
    const auto* phy_instr_operand = dynamic_cast<const PhyInstrOperandT*>(ptr);
    CHECK_NOTNULL(phy_instr_operand);
    phy_instr_operand->ProducerNotifiesConsumer();
    auto* status_buffer_data = instruction->mut_status_buffer()->mut_buffer()->mut_data();
    auto* status_querier = RefCntInstrStatusQuerier::MutCast(status_buffer_data);
    status_querier->SetRefCntAndSetLaunched(phy_instr_operand->consumer_ref_cnt());
  }
};

COMMAND(
    RegisterInstructionType<CriticalSectionInstructionType<InputCriticalSectionPhyInstrOperand>>(
        "InputCriticalSection"));

COMMAND(
    RegisterInstructionType<CriticalSectionInstructionType<OutputCriticalSectionPhyInstrOperand>>(
        "OutputCriticalSection"));

COMMAND(RegisterInstructionType<
        CriticalSectionInstructionType<ParameterCriticalSectionPhyInstrOperand>>(
    "ParameterCriticalSection"));

COMMAND(RegisterInstructionType<CriticalSectionInstructionType<NcclCriticalSectionPhyInstrOperand>>(
    "NcclCriticalSection"));

}  // namespace vm
}  // namespace oneflow
