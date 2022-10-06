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
#ifndef ONEFLOW_CORE_HARDWARE_DEVICE_DESCRIPTOR_LIST_H_
#define ONEFLOW_CORE_HARDWARE_DEVICE_DESCRIPTOR_LIST_H_

#include "oneflow/core/hardware/device_descriptor.h"
#include "oneflow/core/common/util.h"
#include <cstdint>
#include <memory>

namespace oneflow {

namespace hardware {

class DeviceDescriptorList {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DeviceDescriptorList);
  DeviceDescriptorList() = default;
  virtual ~DeviceDescriptorList() = default;

  virtual size_t DeviceCount() const = 0;
  virtual std::shared_ptr<const DeviceDescriptor> GetDevice(size_t ordinal) const = 0;
};

}  // namespace hardware

}  // namespace oneflow

#endif  // ONEFLOW_CORE_HARDWARE_DEVICE_DESCRIPTOR_LIST_H_
