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
#include "oneflow/opencl/ep/cl_device_manager.h"

#include "oneflow/opencl/common/cl_util.h"
#include "oneflow/opencl/common/cl_guard.h"
#include "oneflow/opencl/ep/cl_device.h"

namespace oneflow {
namespace ep {

clDeviceManager::clDeviceManager(DeviceManagerRegistry* registry) : registry_(registry) {}
clDeviceManager::~clDeviceManager() = default;

DeviceManagerRegistry* clDeviceManager::registry() const { return registry_; }

std::shared_ptr<Device> clDeviceManager::GetDevice(size_t device_index) {
  std::lock_guard<std::mutex> lock(devices_mutex_);
  if (device_index < devices_.size() && devices_.at(device_index)) {
    std::shared_ptr<Device> device = devices_.at(device_index);
    return device;
  }
  auto device = std::make_shared<clDevice>(device_index, this);
  if (device_index >= devices_.size()) { devices_.resize(device_index + 1); }
  devices_.at(device_index) = device;
  return device;
}

size_t clDeviceManager::GetDeviceCount(size_t primary_device_index) {
  clCurrentDeviceGuard guard(primary_device_index);
  return this->GetDeviceCount();
}

size_t clDeviceManager::GetDeviceCount() {
  int count = 0;
  OF_CL_CHECK(clGetDeviceCount(&count));
  return count;
}

size_t clDeviceManager::GetActiveDeviceIndex() {
  int device = 0;
  OF_CL_CHECK(clGetDevice(&device));
  return static_cast<size_t>(device);
}

void clDeviceManager::SetActiveDeviceByIndex(size_t device_index) {
  OF_CL_CHECK(clSetDevice(static_cast<int>(device_index)));
}

std::shared_ptr<RandomGenerator> clDeviceManager::CreateRandomGenerator(uint64_t seed,
                                                                        size_t device_index) {
  // TODO
  return nullptr;
}

}  // namespace ep
}  // namespace oneflow
