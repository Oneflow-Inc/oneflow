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

#include "oneflow/core/vm/critical_section_stream_policy.h"
#include "oneflow/core/vm/instruction.h"
#include "oneflow/core/vm/thread_ctx.h"
#include "oneflow/core/vm/critical_section_status_querier.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace vm {

void CriticalSectionStreamPolicy::InitInstructionStatus(
    const Stream& stream, InstructionStatusBuffer* status_buffer) const {
  static_assert(sizeof(CriticalSectionStatusQuerier) < kInstructionStatusBufferBytes, "");
  CriticalSectionStatusQuerier::PlacementNew(status_buffer->mut_buffer());
}

void CriticalSectionStreamPolicy::DeleteInstructionStatus(
    const Stream& stream, InstructionStatusBuffer* status_buffer) const {
  auto* ptr = CriticalSectionStatusQuerier::MutCast(status_buffer->mut_buffer());
  ptr->~CriticalSectionStatusQuerier();
}

bool CriticalSectionStreamPolicy::QueryInstructionStatusLaunched(
    const Stream& stream, const InstructionStatusBuffer& status_buffer) const {
  return CriticalSectionStatusQuerier::Cast(status_buffer.buffer())->QueryLaunched();
}

bool CriticalSectionStreamPolicy::QueryInstructionStatusDone(
    const Stream& stream, const InstructionStatusBuffer& status_buffer) const {
  return CriticalSectionStatusQuerier::Cast(status_buffer.buffer())->QueryDone();
}

void CriticalSectionStreamPolicy::Run(Instruction* instruction) const { instruction->Compute(); }

}  // namespace vm
}  // namespace oneflow
