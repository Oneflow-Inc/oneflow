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
#ifndef ONEFLOW_CORE_HARDWARE_DEVICE_DESCRIPTOR_CLASS_H_
#define ONEFLOW_CORE_HARDWARE_DEVICE_DESCRIPTOR_CLASS_H_

#include "oneflow/core/hardware/device_descriptor_list.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace hardware {

class DeviceDescriptorClass {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DeviceDescriptorClass);
  DeviceDescriptorClass() = default;
  virtual ~DeviceDescriptorClass() = default;

  virtual std::shared_ptr<const DeviceDescriptorList> QueryDeviceDescriptorList() const = 0;
  virtual std::string Name() const = 0;
  virtual void SerializeDeviceDescriptorList(
      const std::shared_ptr<const DeviceDescriptorList>& list, std::string* serialized) const = 0;
  virtual std::shared_ptr<const DeviceDescriptorList> DeserializeDeviceDescriptorList(
      const std::string& serialized) const = 0;
  virtual void DumpDeviceDescriptorListSummary(
      const std::shared_ptr<const DeviceDescriptorList>& list, const std::string& path) const = 0;

  static void RegisterClass(std::shared_ptr<const DeviceDescriptorClass> descriptor_class);
  static size_t GetRegisteredClassesCount();
  static std::shared_ptr<const DeviceDescriptorClass> GetRegisteredClass(size_t index);
  static std::shared_ptr<const DeviceDescriptorClass> GetRegisteredClass(
      const std::string& class_name);
};

}  // namespace hardware

}  // namespace oneflow

#endif  // ONEFLOW_CORE_HARDWARE_DEVICE_DESCRIPTOR_CLASS_H_
