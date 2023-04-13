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

#include "oneflow/core/vm/pinned_ep_stream_policy.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/stream_type.h"
#include "oneflow/core/vm/stream.h"
#include "oneflow/core/vm/thread_ctx.h"
#include "oneflow/core/vm/ep_optional_event_record_status_querier.h"
#include "oneflow/core/vm/ep_backend_host_allocator.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace vm {

namespace {

std::unique_ptr<BinAllocator<ThreadSafeLock>> CreatePinedEpBackendHostAllocator(
    Symbol<Device> device) {
  // TODO:(zhaoluyang) empty/cast/copy op support pin_memory_device
  DeviceType device_type = device->enum_type();
  size_t device_index = device->device_id();
  auto ep_device =
      Singleton<ep::DeviceManagerRegistry>::Get()->GetDevice(device_type, device_index);
  ep::AllocationOptions options{};
  options.SetPinnedDevice(device_type, device_index);
  auto ep_backend_allocator = std::make_unique<EpBackendHostAllocator>(ep_device, options);
  return std::make_unique<BinAllocator<ThreadSafeLock>>(ep::kMaxAlignmentRequirement,
                                                        std::move(ep_backend_allocator));
}

}  // namespace

PinnedEpStreamPolicy::PinnedEpStreamPolicy(Symbol<Device> device)
    : EpStreamPolicyBase(device, CreatePinedEpBackendHostAllocator(device)) {}

void PinnedEpStreamPolicy::InitInstructionStatus(const Stream& stream,
                                                 InstructionStatusBuffer* status_buffer) const {
  static_assert(sizeof(EpOptionalEventRecordStatusQuerier) < kInstructionStatusBufferBytes, "");
  auto* data_ptr = status_buffer->mut_buffer();
  EpOptionalEventRecordStatusQuerier::PlacementNew(data_ptr, nullptr);
}

}  // namespace vm
}  // namespace oneflow
