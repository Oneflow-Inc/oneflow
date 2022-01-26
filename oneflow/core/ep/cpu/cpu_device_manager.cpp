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
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/ep/cpu/cpu_device_manager.h"
#include "oneflow/core/ep/cpu/cpu_device.h"

namespace oneflow {

namespace ep {

constexpr size_t kOtherUsedNumThreads = 2;

CpuDeviceManager::CpuDeviceManager(DeviceManagerRegistry* registry) : registry_(registry) {}

CpuDeviceManager::~CpuDeviceManager() = default;

DeviceManagerRegistry* CpuDeviceManager::registry() const { return registry_; }

std::shared_ptr<Device> CpuDeviceManager::GetDevice(size_t device_index) {
  std::lock_guard<std::mutex> lock(device_mutex_);
  if (!device_) {
    CpuDevice* cpu_device = new CpuDevice(this);
    int64_t cpu_core = std::thread::hardware_concurrency();
    int64_t computing_cores =
        (cpu_core / GlobalProcessCtx::NumOfProcessPerNode()) - kOtherUsedNumThreads;
    if (computing_cores < 1) { computing_cores = 1; }
    computing_cores = ParseIntegerFromEnv("ONEFLOW_EP_CPU_NUM_PARALLELS", computing_cores);
    cpu_device->SetNumParallels(computing_cores);
    device_.reset(cpu_device);
  }
  return device_;
}

size_t CpuDeviceManager::GetDeviceCount(size_t /*primary_device_index*/) { return 1; }

size_t CpuDeviceManager::GetDeviceCount() { return 1; }

size_t CpuDeviceManager::GetActiveDeviceIndex() { return 0; }

void CpuDeviceManager::SetActiveDeviceByIndex(size_t device_index) {}

}  // namespace ep

}  // namespace oneflow
