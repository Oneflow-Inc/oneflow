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
#ifndef ONEFLOW_CORE_EP_DEVICE_MANAGER_REGISTRY_H_
#define ONEFLOW_CORE_EP_DEVICE_MANAGER_REGISTRY_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/ep/include/device_manager.h"
#include "oneflow/core/ep/include/device_manager_factory.h"

namespace oneflow {

namespace ep {

class DeviceManagerRegistry {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DeviceManagerRegistry);
  DeviceManagerRegistry();
  ~DeviceManagerRegistry();

  DeviceManager* GetDeviceManager(DeviceType device_type);
  DeviceManager* GetDeviceManagerOrNull(DeviceType device_type);
  std::shared_ptr<Device> GetDevice(DeviceType device_type, size_t device_index);
  size_t GetDeviceCount(DeviceType device_type);
  size_t GetDeviceCount(const std::string& device_type_name);

  static void RegisterDeviceManagerFactory(std::unique_ptr<DeviceManagerFactory>&& factory);
  static void DumpVersionInfo();
  static std::string GetDeviceTypeNameByDeviceType(DeviceType device_type);
  static DeviceType GetDeviceTypeByDeviceTypeName(const std::string& device_type_name);
  static std::set<DeviceType> GetRegisteredDeviceTypes();
  static bool IsDeviceTypeRegistered(DeviceType device_type);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace ep

}  // namespace oneflow

#endif  // ONEFLOW_CORE_EP_DEVICE_MANAGER_REGISTRY_H_
