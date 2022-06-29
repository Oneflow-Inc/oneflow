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

#include "oneflow/core/vm/ep_stream_type.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/stream_role.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/stream.h"
#include "oneflow/core/vm/thread_ctx.h"
#include "oneflow/core/vm/ep_optional_event_record_status_querier.h"
#include "oneflow/core/vm/ep_device_context.h"
#include "oneflow/core/vm/bin_allocator.h"
#include "oneflow/core/vm/ep_backend_allocator.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/profiler/profiler.h"
#include "oneflow/core/ep/include/device_manager_registry.h"

namespace oneflow {
namespace vm {

void EpStreamType::InitDeviceCtx(std::unique_ptr<DeviceCtx>* device_ctx, Stream* stream) const {
  DeviceType device_type = stream->device()->enum_type();
  size_t device_index = stream->device()->device_id();
  auto ep_device =
      Singleton<ep::DeviceManagerRegistry>::Get()->GetDevice(device_type, device_index);
  auto ep_backend_allocator =
      std::make_unique<EpBackendAllocator>(ep_device, ep::AllocationOptions{});
  device_ctx->reset(new EpDeviceCtx(stream->device(), std::move(ep_backend_allocator)));
}

void EpStreamType::InitInstructionStatus(const Stream& stream,
                                         InstructionStatusBuffer* status_buffer) const {
  static_assert(sizeof(EpOptionalEventRecordStatusQuerier) < kInstructionStatusBufferBytes, "");
  auto* data_ptr = status_buffer->mut_buffer();
  EpOptionalEventRecordStatusQuerier::PlacementNew(data_ptr, nullptr);
}

void EpStreamType::DeleteInstructionStatus(const Stream& stream,
                                           InstructionStatusBuffer* status_buffer) const {
  auto* ptr = EpOptionalEventRecordStatusQuerier::MutCast(status_buffer->mut_buffer());
  ptr->~EpOptionalEventRecordStatusQuerier();
}

bool EpStreamType::QueryInstructionStatusDone(const Stream& stream,
                                              const InstructionStatusBuffer& status_buffer) const {
  return EpOptionalEventRecordStatusQuerier::Cast(status_buffer.buffer())->done();
}

void EpStreamType::Compute(Instruction* instruction) const {
  OF_PROFILER_RANGE_GUARD("S:" + instruction->DebugName());
  auto* stream = instruction->mut_stream();
  auto* ep_device_ctx = static_cast<EpDeviceCtx*>(stream->device_ctx().get());  // NOLINT
  auto* ep_device = ep_device_ctx->GetOrCreateEpDevice();
  ep_device->SetAsActiveDevice();
  instruction->instruction_type().Compute(instruction);
  char* data_ptr = instruction->mut_status_buffer()->mut_buffer();
  EpOptionalEventRecordStatusQuerier::MutCast(data_ptr)->SetLaunched(ep_device_ctx);
}

}  // namespace vm
}  // namespace oneflow
