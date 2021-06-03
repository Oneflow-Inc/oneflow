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
#ifndef ONEFLOW_CORE_DEVICE_NODE_DEVICE_DESCRIPTOR_MANAGER_H_
#define ONEFLOW_CORE_DEVICE_NODE_DEVICE_DESCRIPTOR_MANAGER_H_

#include "oneflow/core/device/node_device_descriptor.h"

namespace oneflow {

namespace device {

class NodeDeviceDescriptorManager {
 public:
  NodeDeviceDescriptorManager();
  ~NodeDeviceDescriptorManager();

  std::shared_ptr<const NodeDeviceDescriptor> GetNodeDeviceDescriptor(int64_t rank) const;
  std::shared_ptr<const NodeDeviceDescriptor> GetLocalNodeDeviceDescriptor() const;

  void DumpSummary(const std::string& path) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace device

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_NODE_DEVICE_DESCRIPTOR_MANAGER_H_
