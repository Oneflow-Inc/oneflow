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

#include "oneflow/core/common/stream_role.h"
#include "oneflow/core/common/singleton_ptr.h"
#include "oneflow/core/vm/event_recorded_ep_stream_type.h"
#include "oneflow/core/vm/control_stream_type.h"
#include "oneflow/core/vm/critical_section_stream_type.h"
#include "oneflow/core/vm/ep_d2h_stream_type.h"
#include "oneflow/core/vm/ep_stream_type.h"
#include "oneflow/core/vm/lazy_job_stream_type.h"
#include "oneflow/core/vm/stream_get_stream_type.h"

namespace oneflow {

struct GetStreamType final : public StreamRoleVisitor<GetStreamType> {
  static Maybe<const vm::StreamType*> VisitCompute(DeviceType device_type) {
    return SingletonPtr<vm::EpStreamType>();
  }
  static Maybe<const vm::StreamType*> VisitHost2Device(DeviceType device_type) {
    return SingletonPtr<vm::EventRecordedEpStreamType>();
  }
  static Maybe<const vm::StreamType*> VisitDevice2Host(DeviceType device_type) {
    return SingletonPtr<vm::EpD2HStreamType>();
  }
  static Maybe<const vm::StreamType*> VisitSyncedLaunchedCommNet(DeviceType device_type) {
    return SingletonPtr<vm::EventRecordedEpStreamType>();
  }
  static Maybe<const vm::StreamType*> VisitAsyncedLaunchedCommNet(DeviceType device_type) {
    return SingletonPtr<vm::EventRecordedEpStreamType>();
  }
  static Maybe<const vm::StreamType*> VisitBarrier(DeviceType device_type) {
    return SingletonPtr<vm::ControlStreamType>();
  }
  static Maybe<const vm::StreamType*> VisitCriticalSection(DeviceType device_type) {
    return SingletonPtr<vm::CriticalSectionStreamType>();
  }
  static Maybe<const vm::StreamType*> VisitLazyJobLauncher(DeviceType device_type) {
    return SingletonPtr<vm::LazyJobStreamType>();
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_STREAM_GET_STREAM_TYPE_H_
