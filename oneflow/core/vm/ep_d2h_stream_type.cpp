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

#include "oneflow/core/vm/ep_d2h_stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/stream.h"
#include "oneflow/core/vm/thread_ctx.h"
#include "oneflow/core/vm/ep_optional_event_record_status_querier.h"
#include "oneflow/core/vm/ep_device_context.h"
#include "oneflow/core/vm/bin_allocator.h"
#include "oneflow/core/vm/ep_backend_host_allocator.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/profiler/profiler.h"
#include "oneflow/core/ep/include/device_manager_registry.h"
#include "oneflow/core/ep/include/allocation_options.h"

namespace oneflow {
namespace vm {

void EpD2HStreamType::InitDeviceCtx(std::unique_ptr<DeviceCtx>* device_ctx, Stream* stream) const {
  DeviceType device_type = stream->device()->enum_type();
  size_t device_index = stream->device()->device_id();
  auto ep_device = Global<ep::DeviceManagerRegistry>::Get()->GetDevice(device_type, device_index);
  auto ep_backend_allocator =
      std::make_unique<EpBackendHostAllocator>(ep_device, ep::AllocationOptions{});
  device_ctx->reset(new EpDeviceCtx(stream->device(), std::move(ep_backend_allocator)));
}

void EpD2HStreamType::InitInstructionStatus(const Stream& stream,
                                            InstructionStatusBuffer* status_buffer) const {
  static_assert(sizeof(EpOptionalEventRecordStatusQuerier) < kInstructionStatusBufferBytes, "");
  auto* ep_device_ctx = static_cast<EpDeviceCtx*>(stream.device_ctx().get());  // NOLINT
  auto* ep_event_provider = ep_device_ctx->ep_event_provider();
  auto* data_ptr = status_buffer->mut_buffer()->mut_data();
  const auto& ep_event = CHECK_NOTNULL(ep_event_provider)->GetReusedEpEvent();
  EpOptionalEventRecordStatusQuerier::PlacementNew(data_ptr, ep_event);
}

void EpD2HStreamType::DeleteInstructionStatus(const Stream& stream,
                                              InstructionStatusBuffer* status_buffer) const {
  auto* ptr = EpOptionalEventRecordStatusQuerier::MutCast(status_buffer->mut_buffer()->mut_data());
  ptr->~EpOptionalEventRecordStatusQuerier();
}

bool EpD2HStreamType::QueryInstructionStatusDone(
    const Stream& stream, const InstructionStatusBuffer& status_buffer) const {
  return EpOptionalEventRecordStatusQuerier::Cast(status_buffer.buffer().data())->done();
}

void EpD2HStreamType::Compute(Instruction* instruction) const {
  OF_PROFILER_RANGE_PUSH("S:" + instruction->instr_msg().DebugName());
  auto* stream = instruction->mut_stream();
  auto* ep_device_ctx = static_cast<EpDeviceCtx*>(stream->device_ctx().get());  // NOLINT
  auto* ep_device = ep_device_ctx->GetOrCreateEpDevice();
  ep_device->SetAsActiveDevice();
  instruction->instr_msg().instruction_type().Compute(instruction);
  char* data_ptr = instruction->mut_status_buffer()->mut_buffer()->mut_data();
  EpOptionalEventRecordStatusQuerier::MutCast(data_ptr)->SetLaunched(ep_device_ctx);
  OF_PROFILER_RANGE_POP();
}

}  // namespace vm
}  // namespace oneflow
