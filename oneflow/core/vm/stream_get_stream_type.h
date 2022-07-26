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
#ifndef ONEFLOW_CORE_VM_STREAM_GET_STREAM_TYPE_H_
#define ONEFLOW_CORE_VM_STREAM_GET_STREAM_TYPE_H_

#include "oneflow/core/common/symbol.h"
#include "oneflow/core/common/stream_role.h"
#include "oneflow/core/common/singleton_ptr.h"
#include "oneflow/core/vm/control_stream_policy.h"
#include "oneflow/core/vm/event_recorded_ep_stream_type.h"
#include "oneflow/core/vm/critical_section_stream_type.h"
#include "oneflow/core/vm/ep_d2h_stream_type.h"
#include "oneflow/core/vm/ep_stream_type.h"
#include "oneflow/core/vm/pinned_ep_stream_type.h"
#include "oneflow/core/vm/lazy_job_stream_type.h"
#include "oneflow/core/vm/naive_stream_policy.h"
#include "oneflow/core/device/device_context.h"

namespace oneflow {

class Device;

struct CreateStreamPolicy final : public StreamRoleVisitor<CreateStreamPolicy> {
  static Maybe<vm::StreamPolicy> VisitCompute(Symbol<Device> device) {
    const auto* stream_type = SingletonPtr<vm::EpStreamType>();
    return Create(stream_type, device);
  }
  static Maybe<vm::StreamPolicy> VisitHost2Device(Symbol<Device> device) {
    const auto* stream_type = SingletonPtr<vm::EventRecordedEpStreamType>();
    return Create(stream_type, device);
  }
  static Maybe<vm::StreamPolicy> VisitDevice2Host(Symbol<Device> device) {
    const auto* stream_type = SingletonPtr<vm::EpD2HStreamType>();
    return Create(stream_type, device);
  }
  static Maybe<vm::StreamPolicy> VisitSyncedLaunchedCommNet(Symbol<Device> device) {
    const auto* stream_type = SingletonPtr<vm::EventRecordedEpStreamType>();
    return Create(stream_type, device);
  }
  static Maybe<vm::StreamPolicy> VisitAsyncedLaunchedCommNet(Symbol<Device> device) {
    const auto* stream_type = SingletonPtr<vm::EventRecordedEpStreamType>();
    return Create(stream_type, device);
  }
  static Maybe<vm::StreamPolicy> VisitBarrier(Symbol<Device> device) {
    return std::shared_ptr<vm::StreamPolicy>(new vm::ControlStreamPolicy());
  }
  static Maybe<vm::StreamPolicy> VisitCriticalSection(Symbol<Device> device) {
    const auto* stream_type = SingletonPtr<vm::CriticalSectionStreamType>();
    return Create(stream_type, device);
  }
  static Maybe<vm::StreamPolicy> VisitLazyJobLauncher(Symbol<Device> device) {
    const auto* stream_type = SingletonPtr<vm::LazyJobStreamType>();
    return Create(stream_type, device);
  }
  static Maybe<vm::StreamPolicy> VisitPinnedCompute(Symbol<Device> device) {
    const auto* stream_type = SingletonPtr<vm::PinnedEpStreamType>();
    return Create(stream_type, device);
  }
  static Maybe<vm::StreamPolicy> VisitTmpCompute(Symbol<Device> device) {
    const auto* stream_type = SingletonPtr<vm::EventRecordedEpStreamType>();
    return Create(stream_type, device);
  }

 private:
  static Maybe<vm::StreamPolicy> Create(const vm::StreamType* stream_type, Symbol<Device> device) {
    std::unique_ptr<DeviceCtx> device_ctx{};
    stream_type->InitDeviceCtx(&device_ctx, device);
    return std::shared_ptr<vm::StreamPolicy>(
        new vm::NaiveStreamPolicy(stream_type, std::move(device_ctx)));
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_STREAM_GET_STREAM_TYPE_H_
