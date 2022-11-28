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
#ifndef ONEFLOW_CORE_FRAMEWORK_STREAM_SUPPORT_STREAM_WAIT_H_
#define ONEFLOW_CORE_FRAMEWORK_STREAM_SUPPORT_STREAM_WAIT_H_

#include <glog/logging.h>
#include "oneflow/core/common/stream_type.h"

namespace oneflow {

struct StreamSupportStreamWait : public StreamTypeVisitor<StreamSupportStreamWait> {
  static bool VisitCompute(DeviceType device_type) { return Supported(device_type); }
  static bool VisitHost2Device(DeviceType device_type) { return Supported(device_type); }
  static bool VisitDevice2Host(DeviceType device_type) { return Supported(device_type); }
  static bool VisitCcl(DeviceType device_type) { return Supported(device_type); }
  static bool VisitBarrier(DeviceType device_type) { return false; }
  static bool VisitCriticalSection(DeviceType device_type) { return false; }
  static bool VisitLazyJobLauncher(DeviceType device_type) { return false; }
  static bool VisitPinnedCompute(DeviceType device_type) { return VisitCompute(device_type); }

 private:
  static bool Supported(DeviceType device_type) { return device_type == kCUDA; }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_STREAM_SUPPORT_STREAM_WAIT_H_
