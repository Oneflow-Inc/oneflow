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
#ifndef ONEFLOW_CORE_VM_STREAM_OUTPUT_MEM_IS_HOST_H_
#define ONEFLOW_CORE_VM_STREAM_OUTPUT_MEM_IS_HOST_H_

#include "oneflow/core/common/stream_type.h"
#include "oneflow/core/common/device_type.h"
#include "oneflow/core/common/device_type_is_host.h"

namespace oneflow {

struct StreamOutputMemIsHost final : public StreamTypeVisitor<StreamOutputMemIsHost> {
  static bool VisitCompute(DeviceType device_type) { return DeviceTypeIsHost::Visit(device_type); }
  static bool VisitHost2Device(DeviceType device_type) {
    return DeviceTypeIsHost::Visit(device_type);
  }
  static bool VisitDevice2Host(DeviceType) { return true; }
  static bool VisitAsyncedDevice2Host(DeviceType device_type) { return true; }
  static bool VisitSyncedLaunchedCommNet(DeviceType device_type) {
    return DeviceTypeIsHost::Visit(device_type);
  }
  static bool VisitAsyncedLaunchedCommNet(DeviceType device_type) {
    return DeviceTypeIsHost::Visit(device_type);
  }
  static bool VisitBarrier(DeviceType) { return true; }
  static bool VisitCriticalSection(DeviceType) { return false; }
  static bool VisitLazyJobLauncher(DeviceType) { return false; }
  static bool VisitPinnedCompute(DeviceType device_type) { return true; }
  static bool VisitTmpCompute(DeviceType device_type) { return VisitCompute(device_type); }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_STREAM_OUTPUT_MEM_IS_HOST_H_
