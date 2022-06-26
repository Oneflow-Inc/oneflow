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
#include "oneflow/core/framework/stream_get_stream_role_name.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/cpp_attribute.h"
#include "oneflow/core/profiler/profiler.h"

namespace oneflow {
namespace vm {

std::string Instruction::DebugName() const {
  std::string instr_name = instruction_type().DebugName(*this);
  return instr_name + ":" + GetStreamRoleName::Visit(stream().stream_role());
}

void Instruction::__Init__(Stream* stream, const InstructionType* instruction_type,
                           const std::shared_ptr<PhyInstrOperand>& phy_instr_operand) {
  stream_ = stream;
  instruction_type_ = instruction_type;
  phy_instr_operand_ = phy_instr_operand;
}

void Instruction::InitStatus() { instruction_type().InitInstructionStatusIf(this); }

void Instruction::DeleteStatusAndClearEdges() {
  OF_PROFILER_RANGE_GUARD("Instruction::DeleteStatusAndClearEdges");
  instruction_type().DeleteInstructionStatusIf(this);
  mut_in_edges()->Clear();
  mut_out_edges()->Clear();
}

bool Instruction::Done() const {
  return stream_type().QueryInstructionStatusDone(stream(), status_buffer());
}

const StreamType& Instruction::stream_type() const { return stream().stream_type(); }

}  // namespace vm
}  // namespace oneflow
