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
#include "oneflow/core/intrusive/flat_msg_view.h"
#include "oneflow/core/vm/cpu_stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/instruction.h"
#include "oneflow/core/vm/thread_ctx.h"
#include "oneflow/core/vm/naive_instruction_status_querier.h"
#include "oneflow/core/device/cpu_device_context.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/profiler/profiler.h"

namespace oneflow {
namespace vm {

void CpuStreamType::InitDeviceCtx(std::unique_ptr<DeviceCtx>* device_ctx, Stream* stream) const {
  device_ctx->reset(new CpuDeviceCtx());
}

void CpuStreamType::InitInstructionStatus(const Stream& stream,
                                          InstructionStatusBuffer* status_buffer) const {
  static_assert(sizeof(NaiveInstrStatusQuerier) < kInstructionStatusBufferBytes, "");
  NaiveInstrStatusQuerier::PlacementNew(status_buffer->mut_buffer()->mut_data());
}

void CpuStreamType::DeleteInstructionStatus(const Stream& stream,
                                            InstructionStatusBuffer* status_buffer) const {
  auto* ptr = NaiveInstrStatusQuerier::MutCast(status_buffer->mut_buffer()->mut_data());
  ptr->~NaiveInstrStatusQuerier();
}

bool CpuStreamType::QueryInstructionStatusDone(const Stream& stream,
                                               const InstructionStatusBuffer& status_buffer) const {
  return NaiveInstrStatusQuerier::Cast(status_buffer.buffer().data())->done();
}

void CpuStreamType::Compute(Instruction* instruction) const {
  OF_PROFILER_RANGE_GUARD("S:" + instruction->instr_msg().DebugName());
  {
    const auto& instr_type_id = instruction->mut_instr_msg()->instr_type_id();
    instr_type_id.instruction_type().Compute(instruction);
  }
  auto* status_buffer = instruction->mut_status_buffer();
  NaiveInstrStatusQuerier::MutCast(status_buffer->mut_buffer()->mut_data())->set_done();
}

intrusive::shared_ptr<StreamDesc> CpuStreamType::MakeStreamDesc(const Resource& resource,
                                                                int64_t this_machine_id) const {
  if (!resource.has_cpu_device_num()) { return intrusive::shared_ptr<StreamDesc>(); }
  std::size_t device_num = resource.cpu_device_num();
  auto ret = intrusive::make_shared<StreamDesc>();
  ret->set_stream_type(StaticGlobalStreamType<CpuStreamType>());
  ret->set_num_streams_per_machine(device_num);
  ret->set_num_streams_per_thread(device_num);
  return ret;
}

}  // namespace vm
}  // namespace oneflow
