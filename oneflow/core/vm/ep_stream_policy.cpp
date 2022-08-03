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

#include "oneflow/core/vm/ep_stream_policy.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/stream_type.h"
#include "oneflow/core/vm/stream.h"
#include "oneflow/core/vm/thread_ctx.h"
#include "oneflow/core/vm/ep_optional_event_record_status_querier.h"
#include "oneflow/core/vm/ep_backend_allocator.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace vm {

namespace {

std::unique_ptr<BinAllocator<ThreadSafeLock>> CreateEpBackendDeviceAllocator(
    Symbol<Device> device) {
  DeviceType device_type = device->enum_type();
  size_t device_index = device->device_id();
  auto ep_device =
      Singleton<ep::DeviceManagerRegistry>::Get()->GetDevice(device_type, device_index);
  auto ep_backend_allocator =
      std::make_unique<EpBackendAllocator>(ep_device, ep::AllocationOptions{});
  return std::make_unique<BinAllocator<ThreadSafeLock>>(ep::kMaxAlignmentRequirement,
                                                        std::move(ep_backend_allocator));
}

}  // namespace

EpStreamPolicy::EpStreamPolicy(Symbol<Device> device)
    : EpStreamPolicyBase(device, CreateEpBackendDeviceAllocator(device)) {}

void EpStreamPolicy::InitInstructionStatus(const Stream& stream,
                                           InstructionStatusBuffer* status_buffer) const {
  static_assert(sizeof(EpOptionalEventRecordStatusQuerier) < kInstructionStatusBufferBytes, "");
  auto* data_ptr = status_buffer->mut_buffer();
  EpOptionalEventRecordStatusQuerier::PlacementNew(data_ptr, nullptr);
}

}  // namespace vm
}  // namespace oneflow
