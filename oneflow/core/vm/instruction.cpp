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
#include "oneflow/core/vm/stream.h"
#include "oneflow/core/vm/thread_ctx.h"
#include "oneflow/core/vm/virtual_machine_engine.h"
#include "oneflow/core/framework/stream_get_stream_type_name.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/cpp_attribute.h"
#include "oneflow/core/profiler/profiler.h"

namespace oneflow {
namespace vm {

std::string Instruction::DebugName() const {
  std::string instr_name = instruction_policy().DebugName(*this);
  return instr_name + ":" + GetStreamTypeName::Visit(stream().stream_type());
}

void Instruction::__Init__(Stream* stream,
                           std::shared_ptr<InstructionPolicy>&& instruction_policy) {
  stream_ = stream;
  instruction_policy_ = instruction_policy;
}

void Instruction::InitStatus() { instruction_policy_->InitInstructionStatusIf(this); }

Maybe<void> Instruction::Prepare() { return instruction_policy_->PrepareIf(this); }
void Instruction::Compute() { return instruction_policy_->ComputeIf(this); }

void Instruction::DeleteStatusAndClearEdges() {
  OF_PROFILER_RANGE_GUARD("Instruction::DeleteStatusAndClearEdges");
  instruction_policy_->DeleteInstructionStatusIf(this);
  INTRUSIVE_FOR_EACH_PTR(edge, mut_in_edges()) {
    Instruction* in_instruction = edge->mut_src_instruction();
    CHECK(in_instruction->Done());
    in_instruction->mut_out_edges()->Erase(edge);
    mut_in_edges()->Erase(edge);
  }
  CHECK_EQ(out_edges().size(), 0);
}

bool Instruction::Done() const {
  return stream_policy().QueryInstructionStatusDone(stream(), status_buffer());
}

StreamPolicy* Instruction::mut_stream_policy() { return mut_stream()->mut_stream_policy(); }

const StreamPolicy& Instruction::stream_policy() const { return stream().stream_policy(); }

}  // namespace vm
}  // namespace oneflow
