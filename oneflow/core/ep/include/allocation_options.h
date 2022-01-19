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
#ifndef ONEFLOW_CORE_EP_ALLOCATION_ATTRIBUTE_H_
#define ONEFLOW_CORE_EP_ALLOCATION_ATTRIBUTE_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/device_type.h"

namespace oneflow {

namespace ep {

class AllocationOptions {
 public:
  AllocationOptions()
      : pinned_device_type_(DeviceType::kInvalidDevice),
        pinned_device_index_{},
        numa_node_affinity_(-1) {}
  ~AllocationOptions() = default;

  bool HasPinnedDevice() const { return pinned_device_type_ != DeviceType::kInvalidDevice; }

  DeviceType GetPinnedDeviceType() const {
    CHECK(HasPinnedDevice());
    return pinned_device_type_;
  }

  size_t GetPinnedDeviceIndex() const {
    CHECK(HasPinnedDevice());
    return pinned_device_index_;
  }

  void SetPinnedDevice(DeviceType device_type, size_t device_index) {
    CHECK(!HasPinnedDevice());
    CHECK_NE(device_type, DeviceType::kInvalidDevice);
    pinned_device_type_ = device_type;
    pinned_device_index_ = device_index;
  }

  void ClearPinnedDevice() { pinned_device_type_ = DeviceType::kInvalidDevice; }

  bool HasNumaNodeAffinity() const { return numa_node_affinity_ >= 0; }

  size_t GetNumaNodeAffinity() const {
    CHECK(HasNumaNodeAffinity());
    return numa_node_affinity_;
  }

  void SetNumaNodeAffinity(size_t numa_node) { numa_node_affinity_ = numa_node; }

  void ClearNumaNodeAffinity() { numa_node_affinity_ = -1; }

 private:
  DeviceType pinned_device_type_;
  size_t pinned_device_index_;
  int32_t numa_node_affinity_;
};

}  // namespace ep

}  // namespace oneflow

#endif  // ONEFLOW_CORE_EP_ALLOCATION_ATTRIBUTE_H_
