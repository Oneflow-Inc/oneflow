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
#ifndef ONEFLOW_CORE_VM_STREAM_CREATE_STREAM_POLICY_H_
#define ONEFLOW_CORE_VM_STREAM_CREATE_STREAM_POLICY_H_

#include "oneflow/core/common/symbol.h"
#include "oneflow/core/common/stream_type.h"
#include "oneflow/core/vm/control_stream_policy.h"
#include "oneflow/core/vm/event_recorded_ep_stream_policy.h"
#include "oneflow/core/vm/critical_section_stream_policy.h"
#include "oneflow/core/vm/ep_d2h_stream_policy.h"
#include "oneflow/core/vm/ep_stream_policy.h"
#include "oneflow/core/vm/pinned_ep_stream_policy.h"
#include "oneflow/core/vm/lazy_job_stream_policy.h"

namespace oneflow {

class Device;

struct CreateStreamPolicy final : public StreamTypeVisitor<CreateStreamPolicy> {
  static Maybe<vm::StreamPolicy> VisitCompute(Symbol<Device> device) {
    return std::shared_ptr<vm::StreamPolicy>(new vm::EpStreamPolicy(device));
  }
  static Maybe<vm::StreamPolicy> VisitHost2Device(Symbol<Device> device) {
    std::unique_ptr<vm::Allocator> allocator{};
    if (device->enum_type() == DeviceType::kCPU) {
      allocator = vm::EventRecordedEpStreamPolicy::CreateEpBackendDeviceAllocator(device);
    } else {
      allocator =
          std::make_unique<vm::UnimplementedAllocator>("allocator is not supported on h2d stream.");
    }
    return std::shared_ptr<vm::StreamPolicy>(
        new vm::EventRecordedEpStreamPolicy(device, std::move(allocator)));
  }
  static Maybe<vm::StreamPolicy> VisitDevice2Host(Symbol<Device> device) {
    return std::shared_ptr<vm::StreamPolicy>(new vm::EpD2HStreamPolicy(device));
  }
  static Maybe<vm::StreamPolicy> VisitCcl(Symbol<Device> device) {
    auto allocator = vm::EventRecordedEpStreamPolicy::CreateEpBackendDeviceAllocator(device);
    return std::shared_ptr<vm::StreamPolicy>(
        new vm::EventRecordedEpStreamPolicy(device, std::move(allocator)));
  }
  static Maybe<vm::StreamPolicy> VisitBarrier(Symbol<Device> device) {
    return std::shared_ptr<vm::StreamPolicy>(new vm::ControlStreamPolicy());
  }
  static Maybe<vm::StreamPolicy> VisitCriticalSection(Symbol<Device> device) {
    return std::shared_ptr<vm::StreamPolicy>(new vm::CriticalSectionStreamPolicy());
  }
  static Maybe<vm::StreamPolicy> VisitLazyJobLauncher(Symbol<Device> device) {
    return std::shared_ptr<vm::StreamPolicy>(new vm::LazyJobStreamPolicy());
  }
  static Maybe<vm::StreamPolicy> VisitPinnedCompute(Symbol<Device> device) {
    return std::shared_ptr<vm::StreamPolicy>(new vm::PinnedEpStreamPolicy(device));
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_STREAM_CREATE_STREAM_POLICY_H_
