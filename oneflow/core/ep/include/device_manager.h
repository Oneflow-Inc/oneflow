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
#ifndef ONEFLOW_CORE_EP_DEVICE_MANAGER_H_
#define ONEFLOW_CORE_EP_DEVICE_MANAGER_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/ep/include/device.h"
#include "oneflow/core/common/auto_registration_factory.h"
#include "oneflow/core/common/device_type.h"

namespace oneflow {

namespace ep {

class DeviceManagerRegistry;

class DeviceManager {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DeviceManager);
  DeviceManager() = default;
  virtual ~DeviceManager() = default;

  virtual DeviceManagerRegistry* registry() const = 0;
  virtual std::shared_ptr<Device> GetDevice(size_t device_index) = 0;
  virtual size_t GetDeviceCount(size_t primary_device_index) = 0;
  virtual size_t GetDeviceCount() = 0;
  virtual size_t GetActiveDeviceIndex() = 0;
  virtual void SetActiveDeviceByIndex(size_t device_index) = 0;
};

}  // namespace ep

}  // namespace oneflow

#endif  // ONEFLOW_CORE_EP_DEVICE_MANAGER_H_
