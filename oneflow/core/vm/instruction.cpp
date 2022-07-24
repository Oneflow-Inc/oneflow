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
  std::string instr_name = instruction_policy().DebugName(*this);
  return instr_name + ":" + GetStreamRoleName::Visit(stream().stream_role());
}

void Instruction::__Init__(Stream* stream,
                           std::unique_ptr<InstructionPolicy>&& instruction_policy) {
  stream_ = stream;
  instruction_policy_ = std::move(instruction_policy);
}

void Instruction::InitStatus() { instruction_policy_->InitInstructionStatusIf(this); }

Maybe<void> Instruction::Prepare() { return instruction_policy_->PrepareIf(this); }
void Instruction::Compute() { return instruction_policy_->ComputeIf(this); }

void Instruction::DeleteStatusAndClearEdges() {
  OF_PROFILER_RANGE_GUARD("Instruction::DeleteStatusAndClearEdges");
  instruction_policy_->DeleteInstructionStatusIf(this);
  mut_in_edges()->Clear();
  mut_out_edges()->Clear();
}

bool Instruction::Done() const {
  return stream_policy().QueryInstructionStatusDone(stream(), status_buffer())
         && in_edges().empty();
}

StreamPolicy* Instruction::mut_stream_policy() { return mut_stream()->mut_stream_policy(); }

const StreamPolicy& Instruction::stream_policy() const { return stream().stream_policy(); }

}  // namespace vm
}  // namespace oneflow
