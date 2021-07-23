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
#ifndef ONEFLOW_CORE_DEVICE_BASIC_DEVICE_DESCRIPTOR_LIST_H_
#define ONEFLOW_CORE_DEVICE_BASIC_DEVICE_DESCRIPTOR_LIST_H_

#include "oneflow/core/device/device_descriptor_list.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace oneflow {

namespace device {

class BasicDeviceDescriptorList : public DeviceDescriptorList {
 public:
  explicit BasicDeviceDescriptorList(
      std::vector<std::shared_ptr<const DeviceDescriptor>> device_descriptor_list);
  BasicDeviceDescriptorList();
  ~BasicDeviceDescriptorList() override;

  size_t DeviceCount() const override;
  std::shared_ptr<const DeviceDescriptor> GetDevice(size_t ordinal) const override;

 private:
  std::vector<std::shared_ptr<const DeviceDescriptor>> device_descriptor_list_;
};

}  // namespace device

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_BASIC_DEVICE_DESCRIPTOR_LIST_H_
