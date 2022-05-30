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
#ifdef WITH_NPU
#include "oneflow/core/ep/npu/npu_device_manager.h"
#include "oneflow/core/ep/npu/npu_device.h"
#include "oneflow/core/device/npu_util.h"


namespace oneflow {

namespace ep {

NpuDeviceManager::NpuDeviceManager(DeviceManagerRegistry* registry) : registry_(registry) {}
NpuDeviceManager::~NpuDeviceManager() = default;

DeviceManagerRegistry* NpuDeviceManager::registry() const { return registry_; }

std::shared_ptr<Device> NpuDeviceManager::GetDevice(size_t device_index) {
  std::lock_guard<std::mutex> lock(devices_mutex_);
  if (device_index < devices_.size() && devices_.at(device_index)) {
    return devices_.at(device_index);
  }
  auto device = std::make_shared<NpuDevice>(device_index, this);
  if (device_index >= devices_.size()) { devices_.resize(device_index + 1); }
  devices_.at(device_index) = device;
  return device;
}

size_t NpuDeviceManager::GetDeviceCount(size_t primary_device_index) {
  std::cout<<"NpuDeviceManager::GetDeviceCount(size_t)"<<std::endl;
  NpuCurrentDeviceGuard guard(primary_device_index);
  return this->GetDeviceCount();
}

size_t NpuDeviceManager::GetDeviceCount() {
  std::cout<<"NpuDeviceManager::GetDeviceCount()"<<std::endl;
  uint32_t count = 0;
  OF_NPU_CHECK(aclrtGetDeviceCount(&count));
  return count;
}

size_t NpuDeviceManager::GetActiveDeviceIndex() {
  std::cout<<"NpuDeviceManager::GetActiveDeviceIndex()"<<std::endl;
  int device = 0;
  OF_NPU_CHECK(aclrtGetDevice(&device));
  return static_cast<size_t>(device);
}

void NpuDeviceManager::SetActiveDeviceByIndex(size_t device_index) {
  std::cout<<"NpuDeviceManager::SetActiveDeviceByIndex(size_t)"<<std::endl;
  OF_NPU_CHECK(aclrtSetDevice(static_cast<int>(device_index)));
}

}  // namespace ep

}  // namespace oneflow

#endif  // WITH_CUDA
