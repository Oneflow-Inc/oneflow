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
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/core/vm/stream.h"
#include "oneflow/core/vm/thread_ctx.h"
#include "oneflow/core/vm/virtual_machine_engine.h"
#include "oneflow/core/framework/stream_get_stream_type_name.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/cpp_attribute.h"
#include "oneflow/extension/stack/foreign_stack_getter.h"
#include "oneflow/core/profiler/profiler.h"

namespace oneflow {
namespace vm {

std::string Instruction::DebugName() const {
  std::string instr_name = instruction_policy().DebugName(*this);
  return instr_name + ":s_" + GetStreamTypeName::Visit(stream().stream_type());
}

void Instruction::__Init__(Stream* stream,
                           std::shared_ptr<InstructionPolicy>&& instruction_policy) {
  stream_ = stream;
  instruction_policy_ = std::move(instruction_policy);
  if (IsMainThread()) {
    if (auto* stack_getter = Singleton<ForeignStackGetter>::Get()) {
      foreign_frame_ = stack_getter->GetCurrentFrame();
    }
  }
}

void Instruction::InitStatus() { instruction_policy_->InitInstructionStatusIf(this); }

Maybe<void> Instruction::Prepare() {
  ForeignFrameThreadLocalGuard guard(foreign_frame_);
  return instruction_policy_->PrepareIf(this);
}
void Instruction::Compute() {
  ForeignFrameThreadLocalGuard guard(foreign_frame_);
  instruction_policy_->ComputeIf(this);
}

void Instruction::DeleteStatusAndCheckEdges() {
  OF_PROFILER_RANGE_GUARD("Instruction::DeleteStatusAndCheckEdges");
  instruction_policy_->DeleteInstructionStatusIf(this);
  INTRUSIVE_FOR_EACH_PTR(edge, mut_in_edges()) {
    Instruction* in_instruction = edge->mut_src_instruction();
    LOG(FATAL) << "unerased edge: " << in_instruction->DebugName() << " -> " << this->DebugName();
  }
  INTRUSIVE_FOR_EACH_PTR(edge, mut_out_edges()) {
    Instruction* out_instruction = edge->mut_dst_instruction();
    LOG(FATAL) << "unerased edge: " << this->DebugName() << " -> " << out_instruction->DebugName();
  }
}

bool Instruction::Launched() const {
  return stream_policy().QueryInstructionStatusLaunched(stream(), status_buffer());
}

bool Instruction::Done() const {
  return stream_policy().QueryInstructionStatusDone(stream(), status_buffer());
}

StreamPolicy* Instruction::mut_stream_policy() { return mut_stream()->mut_stream_policy(); }

const StreamPolicy& Instruction::stream_policy() const { return stream().stream_policy(); }

}  // namespace vm
}  // namespace oneflow
