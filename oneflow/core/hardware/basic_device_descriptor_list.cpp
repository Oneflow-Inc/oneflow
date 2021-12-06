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
#include "oneflow/core/hardware/basic_device_descriptor_list.h"

namespace oneflow {

namespace hardware {

BasicDeviceDescriptorList::BasicDeviceDescriptorList(
    std::vector<std::shared_ptr<const DeviceDescriptor>> device_descriptor_list)
    : device_descriptor_list_(std::move(device_descriptor_list)) {}

BasicDeviceDescriptorList::BasicDeviceDescriptorList()
    : BasicDeviceDescriptorList(std::vector<std::shared_ptr<const DeviceDescriptor>>()) {}

BasicDeviceDescriptorList::~BasicDeviceDescriptorList() = default;

size_t BasicDeviceDescriptorList::DeviceCount() const { return device_descriptor_list_.size(); }

std::shared_ptr<const DeviceDescriptor> BasicDeviceDescriptorList::GetDevice(size_t ordinal) const {
  if (ordinal < device_descriptor_list_.size()) {
    return device_descriptor_list_.at(ordinal);
  } else {
    return nullptr;
  }
}

}  // namespace hardware

}  // namespace oneflow
