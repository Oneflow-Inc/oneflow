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
#ifndef ONEFLOW_CORE_VM_EVENT_RECORDED_EP_STREAM_POLICY_H_
#define ONEFLOW_CORE_VM_EVENT_RECORDED_EP_STREAM_POLICY_H_

#include "oneflow/core/vm/ep_stream_policy_base.h"

namespace oneflow {
namespace vm {

class EventRecordedEpStreamPolicy final : public EpStreamPolicyBase {
 public:
  EventRecordedEpStreamPolicy(Symbol<Device> device, std::unique_ptr<vm::Allocator>&& allocator);
  ~EventRecordedEpStreamPolicy() override = default;

  void InitInstructionStatus(const Stream& stream,
                             InstructionStatusBuffer* status_buffer) const override;

  static std::unique_ptr<BinAllocator<ThreadSafeLock>> CreateEpBackendDeviceAllocator(
      Symbol<Device> device);
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_EVENT_RECORDED_EP_STREAM_POLICY_H_
