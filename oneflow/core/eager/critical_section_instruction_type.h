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
#ifndef ONEFLOW_CORE_EAGER_CRITICAL_SECTION_INSTRUCTION_TYPE_H_
#define ONEFLOW_CORE_EAGER_CRITICAL_SECTION_INSTRUCTION_TYPE_H_

#include "oneflow/core/vm/critical_section_status_querier.h"
#include "oneflow/core/eager/critical_section_phy_instr_operand.h"
#include "oneflow/core/job/critical_section_instance.h"
#include "oneflow/core/framework/nn_graph_if.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/vm/instruction.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/common/buffer_manager.h"
#include "oneflow/core/common/singleton.h"
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

  std::string DebugName(const vm::Instruction& instruction) const override {
    return "CriticalSectionBegin";
  }
  Maybe<void> Prepare(vm::Instruction* instruction) const override { return Maybe<void>::Ok(); }
  void Compute(vm::Instruction* instruction) const override {
    OF_PROFILER_RANGE_GUARD("CriticalSectionBegin");
    {
      auto ptr = instruction->phy_instr_operand();
      auto phy_instr_operand = std::dynamic_pointer_cast<CriticalSectionBeginPhyInstrOperand>(ptr);
      CHECK_NOTNULL(phy_instr_operand);
      const auto& critical_section_instance = MakeCriticalSectionInstance(phy_instr_operand);
      const auto& job_name = critical_section_instance->job_name();
      auto* buffer_mgr = Singleton<BufferMgr<std::shared_ptr<CriticalSectionInstance>>>::Get();
      for (int i = 0; i < phy_instr_operand->interfaces_op_names().size(); ++i) {
        if (phy_instr_operand->interfaces_valid().at(i)) {
          const std::string& interface_op_name = phy_instr_operand->interfaces_op_names().at(i);
          const auto& buffer_name =
              phy_instr_operand->GetInterfaceBufferName(job_name, interface_op_name);
          buffer_mgr->Get(buffer_name)->Push(critical_section_instance);
        }
      }
      const auto& callback_buffer_name =
          phy_instr_operand->GetInterfaceCriticalSectionCallbackBufferName(job_name);
      buffer_mgr->Get(callback_buffer_name)->Push(critical_section_instance);
      const auto& wait_buffer_name =
          phy_instr_operand->GetInterfaceCriticalSectionWaitBufferName(job_name);
      buffer_mgr->Get(wait_buffer_name)->Push(critical_section_instance);
    }
    {
      auto* status_buffer_data = instruction->mut_status_buffer()->mut_buffer();
      auto* status_querier = CriticalSectionStatusQuerier::MutCast(status_buffer_data);
      status_querier->SetLaunched(std::make_shared<NaiveEventRecord>());
    }
  }

 private:
  class NaiveCriticalSectionInstance final : public CriticalSectionInstance {
   public:
    NaiveCriticalSectionInstance(
        const std::shared_ptr<CriticalSectionBeginPhyInstrOperand>& phy_instr_operand,
        const std::string& job_name)
        : CriticalSectionInstance(), phy_instr_operand_(phy_instr_operand), job_name_(job_name) {}

    ~NaiveCriticalSectionInstance() override = default;

    const std::string& job_name() const override { return job_name_; }

    void AccessBlobByOpName(uint64_t ofblob_ptr, const std::string& op_name) const override {
      phy_instr_operand_->AccessBlobByOpName(ofblob_ptr, op_name);
    }
    void Finish() const override { phy_instr_operand_->Finish(); }

   private:
    std::shared_ptr<CriticalSectionBeginPhyInstrOperand> phy_instr_operand_;
    std::string job_name_;
  };

  std::shared_ptr<CriticalSectionInstance> MakeCriticalSectionInstance(
      const std::shared_ptr<CriticalSectionBeginPhyInstrOperand>& phy_instr_operand) const {
    phy_instr_operand->FinishInvalidInterfaceEventRecords();
    const auto& job_name = phy_instr_operand->nn_graph()->job_name();
    return std::make_shared<NaiveCriticalSectionInstance>(phy_instr_operand, job_name);
  }
};

class CriticalSectionEndInstructionType final : public InstructionType {
 public:
  CriticalSectionEndInstructionType(const CriticalSectionEndInstructionType&) = delete;
  CriticalSectionEndInstructionType(CriticalSectionEndInstructionType&&) = delete;
  CriticalSectionEndInstructionType& operator=(const CriticalSectionEndInstructionType&) = delete;
  CriticalSectionEndInstructionType& operator=(CriticalSectionEndInstructionType&&) = delete;
  CriticalSectionEndInstructionType() = default;
  ~CriticalSectionEndInstructionType() = default;

  std::string DebugName(const vm::Instruction& instruction) const override {
    return "CriticalSectionEnd";
  }
  Maybe<void> Prepare(vm::Instruction* instruction) const override { return Maybe<void>::Ok(); }
  void Compute(vm::Instruction* instruction) const override {
    const auto* ptr = instruction->phy_instr_operand().get();
    const auto* phy_instr_operand = dynamic_cast<const CriticalSectionEndPhyInstrOperand*>(ptr);
    CHECK_NOTNULL(phy_instr_operand);
    auto* status_buffer_data = instruction->mut_status_buffer()->mut_buffer();
    auto* status_querier = CriticalSectionStatusQuerier::MutCast(status_buffer_data);
    status_querier->SetLaunched(phy_instr_operand->event_record());
  }
};

}  // namespace vm
}  // namespace oneflow
#endif  // ONEFLOW_CORE_EAGER_CRITICAL_SECTION_INSTRUCTION_TYPE_H_
