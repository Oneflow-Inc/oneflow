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
#include "oneflow/core/ep/include/active_device_guard.h"
#include "oneflow/core/ep/include/device_manager_registry.h"

namespace oneflow {

namespace ep {

ActiveDeviceGuard::ActiveDeviceGuard(Device* device) {
  device_manager_ =                                                                       // NOLINT
      Global<ep::DeviceManagerRegistry>::Get()->GetDeviceManager(device->device_type());  // NOLINT
  CHECK_NOTNULL(device_manager_);
  saved_active_device_ = device_manager_->GetActiveDeviceIndex();
  device->SetAsActiveDevice();
}

ActiveDeviceGuard::~ActiveDeviceGuard() {
  device_manager_->SetActiveDeviceByIndex(saved_active_device_);
}

}  // namespace ep

}  // namespace oneflow
