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
#include "oneflow/core/job/critical_section_instance.h"
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

class CriticalSectionBeginInstructionType final : public InstructionType {
 public:
  CriticalSectionBeginInstructionType(const CriticalSectionBeginInstructionType&) = delete;
  CriticalSectionBeginInstructionType(CriticalSectionBeginInstructionType&&) = delete;
  CriticalSectionBeginInstructionType& operator=(const CriticalSectionBeginInstructionType&) = delete;
  CriticalSectionBeginInstructionType& operator=(CriticalSectionBeginInstructionType&&) = delete;
  CriticalSectionBeginInstructionType() = default;
  ~CriticalSectionBeginInstructionType() = default;

  using stream_type = CriticalSectionStreamType;

  void Infer(vm::Instruction* instruction) const override { UNIMPLEMENTED(); }

  void Compute(vm::Instruction* instruction) const override {
    {
      OF_PROFILER_RANGE_PUSH("MakeCriticalSectionInstance");
      const auto& critical_section_instance = MakeCriticalSectionInstance(instruction);
      OF_PROFILER_RANGE_POP();  // MakeCriticalSectionInstance
      const auto& cur_nn_graph = GetCurNNGraph(instruction);
      const auto& job_name = job_instance->job_name();
      auto* buffer_mgr = Global<BufferMgr<std::shared_ptr<CriticalSectionInstance>>>::Get();
      buffer_mgr->Get(GetCriticalSectionWaitBufferName(job_name))->Push(job_instance);
      for (int i = 0; i < cur_nn_graph->inputs_op_names().size(); ++i) {
        if (cur_nn_graph->inputs_valid().at(i)) {
          const std::string& input_op_name = cur_nn_graph->inputs_op_names().at(i);
          buffer_mgr->Get(GetInputBufferName(job_name, input_op_name))->Push(critical_section_instance);
        }
      }
      buffer_mgr->Get(GetCriticalSectionCallbackBufferName(job_name))->Push(job_instance);
    }
    {
      auto* status_buffer_data = instruction->mut_status_buffer()->mut_buffer()->mut_data();
      auto* status_querier = CriticalSectionStatusQuerier::MutCast(status_buffer_data);
      status_querier->SetLaunched(std::make_shared<NaiveEventRecord>());
    }
  }

 private:
  std::shared_ptr<NNGraphIf> GetCurNNGraph(Instruction* instruction) const {
    const auto* ptr = instruction->instr_msg().phy_instr_operand().get();
    const auto* phy_instr_operand = dynamic_cast<const CriticalSectionBeginPhyInstrOperand*>(ptr);
    CHECK_NOTNULL(phy_instr_operand);
    return phy_instr_operand->nn_graph();
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
