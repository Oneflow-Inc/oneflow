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
#ifndef ONEFLOW_CORE_EP_ACTIVE_DEVICE_GUARD_H_
#define ONEFLOW_CORE_EP_ACTIVE_DEVICE_GUARD_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/ep/include/device.h"

namespace oneflow {

namespace ep {

class DeviceManager;

class ActiveDeviceGuard {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ActiveDeviceGuard);
  explicit ActiveDeviceGuard(Device* device);
  ~ActiveDeviceGuard();

 private:
  size_t saved_active_device_;
  DeviceManager* device_manager_;
};

}  // namespace ep

}  // namespace oneflow

#endif  // ONEFLOW_CORE_EP_ACTIVE_DEVICE_GUARD_H_
