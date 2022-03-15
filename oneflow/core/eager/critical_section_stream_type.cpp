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
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/instruction.h"
#include "oneflow/core/vm/thread_ctx.h"
#include "oneflow/core/eager/critical_section_status_querier.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace vm {

void CriticalSectionStreamType::InitDeviceCtx(std::unique_ptr<DeviceCtx>* device_ctx,
                                              Stream* stream) const {
  device_ctx->reset();
}

void CriticalSectionStreamType::InitInstructionStatus(
    const Stream& stream, InstructionStatusBuffer* status_buffer) const {
  static_assert(sizeof(CriticalSectionStatusQuerier) < kInstructionStatusBufferBytes, "");
  CriticalSectionStatusQuerier::PlacementNew(status_buffer->mut_buffer()->mut_data());
}

void CriticalSectionStreamType::DeleteInstructionStatus(
    const Stream& stream, InstructionStatusBuffer* status_buffer) const {
  auto* ptr = CriticalSectionStatusQuerier::MutCast(status_buffer->mut_buffer()->mut_data());
  ptr->~CriticalSectionStatusQuerier();
}

bool CriticalSectionStreamType::QueryInstructionStatusDone(
    const Stream& stream, const InstructionStatusBuffer& status_buffer) const {
  return CriticalSectionStatusQuerier::Cast(status_buffer.buffer().data())->QueryDone();
}

void CriticalSectionStreamType::Compute(Instruction* instruction) const {
  {
    const auto& instr_type_id = instruction->mut_instr_msg()->instr_type_id();
    instr_type_id.instruction_type().Compute(instruction);
  }
}

intrusive::shared_ptr<StreamDesc> CriticalSectionStreamType::MakeStreamDesc(
    const Resource& resource, int64_t this_machine_id) const {
  auto ret = intrusive::make_shared<StreamDesc>();
  ret->set_stream_type(StaticGlobalStreamType<CriticalSectionStreamType>());
  ret->set_num_streams_per_machine(1);
  ret->set_num_streams_per_thread(1);
  return ret;
}

}  // namespace vm
}  // namespace oneflow
