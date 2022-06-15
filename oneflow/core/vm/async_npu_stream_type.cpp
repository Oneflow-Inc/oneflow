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
#ifdef WITH_NPU

#include "oneflow/core/vm/async_npu_stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/stream.h"
#include "oneflow/core/vm/npu_stream_handle_device_context.h"
#include "oneflow/core/vm/npu_optional_event_record_status_querier.h"
#include "oneflow/core/profiler/profiler.h"

namespace oneflow {
namespace vm {

void AsyncNpuStreamType::InitDeviceCtx(std::unique_ptr<DeviceCtx>* device_ctx,
                                        Stream* stream) const {
  device_ctx->reset(new NpuStreamHandleDeviceCtx(stream->device_id()));
}

void AsyncNpuStreamType::InitInstructionStatus(const Stream& stream,
                                                InstructionStatusBuffer* status_buffer) const {
  static_assert(sizeof(NpuOptionalEventRecordStatusQuerier) < kInstructionStatusBufferBytes, "");
  auto* event_provider = dynamic_cast<QueryNpuEventProvider*>(stream.device_ctx().get());
  auto* data_ptr = status_buffer->mut_buffer()->mut_data();
  const auto& npu_event = CHECK_NOTNULL(event_provider)->GetNpuEvent();
  NpuOptionalEventRecordStatusQuerier::PlacementNew(data_ptr, npu_event);
}

void AsyncNpuStreamType::DeleteInstructionStatus(const Stream& stream,
                                                  InstructionStatusBuffer* status_buffer) const {
  auto* ptr =
      NpuOptionalEventRecordStatusQuerier::MutCast(status_buffer->mut_buffer()->mut_data());
  ptr->~NpuOptionalEventRecordStatusQuerier();
}

bool AsyncNpuStreamType::QueryInstructionStatusDone(
    const Stream& stream, const InstructionStatusBuffer& status_buffer) const {
  return NpuOptionalEventRecordStatusQuerier::Cast(status_buffer.buffer().data())->done();
}

void AsyncNpuStreamType::Compute(Instruction* instruction) const {
  OF_PROFILER_RANGE_PUSH("S:" + instruction->instr_msg().DebugName());
  auto* stream = instruction->mut_stream();
  aclrtSetDevice(stream->device_id());
  {
    const auto& instr_type_id = instruction->mut_instr_msg()->instr_type_id();
    instr_type_id.instruction_type().Compute(instruction);
    //OF_NPU_CHECK(cudaGetLastError()); dck_caution_here
  }
  char* data_ptr = instruction->mut_status_buffer()->mut_buffer()->mut_data();
  NpuOptionalEventRecordStatusQuerier::MutCast(data_ptr)->SetLaunched(stream->device_ctx().get());
  OF_PROFILER_RANGE_POP();
}

intrusive::shared_ptr<StreamDesc> AsyncNpuStreamType::MakeStreamDesc(
    const Resource& resource, int64_t this_machine_id) const {
  //if (!resource.has_gpu_device_num()) { return intrusive::shared_ptr<StreamDesc>(); }
  std::size_t device_num = 1; //dck_caution_here
  auto ret = intrusive::make_shared<StreamDesc>();
  ret->set_stream_type(StaticGlobalStreamType<AsyncNpuStreamType>());
  ret->set_num_streams_per_machine(device_num);
  ret->set_num_streams_per_thread(device_num);
  return ret;
}

}  // namespace vm
}  // namespace oneflow

#endif
