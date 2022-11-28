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
#include "oneflow/core/ep/cpu/cpu_device_manager.h"
#include "oneflow/core/ep/cpu/cpu_device.h"

namespace oneflow {

namespace ep {

CpuDeviceManager::CpuDeviceManager(DeviceManagerRegistry* registry)
    : device_num_threads_(1), registry_(registry) {}

CpuDeviceManager::~CpuDeviceManager() = default;

DeviceManagerRegistry* CpuDeviceManager::registry() const { return registry_; }

std::shared_ptr<Device> CpuDeviceManager::GetDevice(size_t device_index) {
  std::lock_guard<std::mutex> lock(device_mutex_);
  if (!device_) { device_.reset(new CpuDevice(this)); }
  device_->SetNumThreads(device_num_threads_);
  return device_;
}

size_t CpuDeviceManager::GetDeviceCount(size_t /*primary_device_index*/) { return 1; }

size_t CpuDeviceManager::GetDeviceCount() { return 1; }

size_t CpuDeviceManager::GetActiveDeviceIndex() { return 0; }

void CpuDeviceManager::SetActiveDeviceByIndex(size_t device_index) {}

void CpuDeviceManager::SetDeviceNumThreads(size_t num_threads) {
  device_num_threads_ = num_threads;
}

}  // namespace ep

}  // namespace oneflow
