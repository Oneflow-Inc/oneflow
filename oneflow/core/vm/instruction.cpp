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
#include "oneflow/core/vm/instruction.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/stream.h"
#include "oneflow/core/vm/thread_ctx.h"
#include "oneflow/core/vm/virtual_machine_engine.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/cpp_attribute.h"
#include "oneflow/core/profiler/profiler.h"

namespace oneflow {
namespace vm {

std::string InstructionMsg::DebugName() const {
  std::string op_type_name = instr_type_id().instruction_type().DebugOpTypeName(*this);
  return op_type_name + ":" + instr_type_name();
}

void InstructionMsg::__Init__() { *mut_instr_type_name() = ""; }

void InstructionMsg::__Init__(const std::string& instr_type_name) {
  __Init__();
  mut_instr_type_id()->CopyFrom(LookupInstrTypeId(instr_type_name));
  *mut_instr_type_name() = instr_type_name;
}

void InstructionMsg::__Init__(VirtualMachineEngine* vm, const std::string& instr_type_name,
                              const std::shared_ptr<const ParallelDesc>& phy_instr_parallel_desc,
                              const std::shared_ptr<PhyInstrOperand>& phy_instr_operand) {
  __Init__();
  // There are instructions without concept of ParallelDesc, like LaunchLazyJob,
  // ComputeGlobalFrontSeqBarrier. If phy_instr_parallel_desc is empty, Instructions are run on the
  // sole stream within the StreamRtDesc.
  if (likely(phy_instr_parallel_desc)) {
    int device_id = phy_instr_parallel_desc->parallel_id2device_id().at(0);
    vm->GetCachedInstrTypeIdAndPhyInstrStream(instr_type_name, device_id, mut_instr_type_id(),
                                              &phy_instr_stream_);
  } else {
    vm->GetInstrTypeIdAndSoleStream(instr_type_name, mut_instr_type_id(), &phy_instr_stream_);
  }
  *mut_instr_type_name() = instr_type_name;
  phy_instr_parallel_desc_ = phy_instr_parallel_desc;
  phy_instr_operand_ = phy_instr_operand;
}

void InstructionMsg::__Init__(const InstructionMsg& instr_msg) {
  __Init__();
  mut_instr_type_id()->CopyFrom(instr_msg.instr_type_id());
  *mut_instr_type_name() = instr_msg.instr_type_name();
  const auto& parallel_desc = instr_msg.phy_instr_parallel_desc();
  if (parallel_desc) { phy_instr_parallel_desc_ = parallel_desc; }
  phy_instr_operand_ = instr_msg.phy_instr_operand();
  if (instr_msg.phy_instr_stream() != nullptr) { phy_instr_stream_ = instr_msg.phy_instr_stream(); }
}

intrusive::shared_ptr<InstructionMsg> InstructionMsg::Clone() const {
  return intrusive::make_shared<InstructionMsg>(*this);
}

void Instruction::Init(InstructionMsg* instr_msg, Stream* stream,
                       const std::shared_ptr<const ParallelDesc>& parallel_desc) {
  __Init__();
  reset_instr_msg(instr_msg);
  set_stream(stream);
  instr_msg->instr_type_id().instruction_type().InitInstructionStatusIf(this);
  *mut_parallel_desc() = parallel_desc;
}

void Instruction::Delete() {
  OF_PROFILER_RANGE_GUARD("Instruction::Delete");
  instr_msg().instr_type_id().instruction_type().DeleteInstructionStatusIf(this);
  OF_PROFILER_RANGE_PUSH("ClearInstrMsg");
  clear_instr_msg();
  OF_PROFILER_RANGE_POP();
  mut_in_edges()->Clear();
  mut_out_edges()->Clear();
}

bool Instruction::Done() const {
  return stream_type().QueryInstructionStatusDone(stream(), status_buffer());
}

const StreamType& Instruction::stream_type() const { return stream().stream_type(); }

}  // namespace vm
}  // namespace oneflow
