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
#include "oneflow/cambricon/ep/mlu_device_manager.h"

#include "oneflow/cambricon/common/mlu_util.h"
#include "oneflow/cambricon/common/mlu_guard.h"
#include "oneflow/cambricon/ep/mlu_device.h"
#include "oneflow/cambricon/ep/mlu_random_generator.h"

namespace oneflow {
namespace ep {

MluDeviceManager::MluDeviceManager(DeviceManagerRegistry* registry) : registry_(registry) {}
MluDeviceManager::~MluDeviceManager() = default;

DeviceManagerRegistry* MluDeviceManager::registry() const { return registry_; }

std::shared_ptr<Device> MluDeviceManager::GetDevice(size_t device_index) {
  std::lock_guard<std::mutex> lock(devices_mutex_);
  if (device_index < devices_.size() && devices_.at(device_index)) {
    std::shared_ptr<Device> device = devices_.at(device_index);
    return device;
  }
  auto device = std::make_shared<MluDevice>(device_index, this);
  if (device_index >= devices_.size()) { devices_.resize(device_index + 1); }
  devices_.at(device_index) = device;
  return device;
}

size_t MluDeviceManager::GetDeviceCount(size_t primary_device_index) {
  MluCurrentDeviceGuard guard(primary_device_index);
  return this->GetDeviceCount();
}

size_t MluDeviceManager::GetDeviceCount() {
  uint32_t count = 0;
  OF_MLU_CHECK(cnrtGetDeviceCount(&count));
  return count;
}

size_t MluDeviceManager::GetActiveDeviceIndex() {
  int device = 0;
  OF_MLU_CHECK(cnrtGetDevice(&device));
  return static_cast<size_t>(device);
}

void MluDeviceManager::SetActiveDeviceByIndex(size_t device_index) {
  OF_MLU_CHECK(cnrtSetDevice(static_cast<int>(device_index)));
}

std::shared_ptr<RandomGenerator> MluDeviceManager::CreateRandomGenerator(uint64_t seed,
                                                                         size_t device_index) {
  return std::make_shared<MLUGenerator>(seed, device_index);
}

}  // namespace ep
}  // namespace oneflow
