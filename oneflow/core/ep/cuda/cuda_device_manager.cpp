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
#include "oneflow/core/ep/cuda/cuda_device_manager.h"
#include "oneflow/core/device/cuda_util.h"

#ifdef WITH_CUDA

namespace oneflow {

namespace ep {

CudaDeviceManager::CudaDeviceManager(DeviceManagerRegistry* registry) : registry_(registry) {}
CudaDeviceManager::~CudaDeviceManager() = default;

DeviceManagerRegistry* CudaDeviceManager::registry() const { return registry_; }

std::shared_ptr<Device> CudaDeviceManager::GetDevice(size_t device_index) {
  std::lock_guard<std::mutex> lock(devices_mutex_);
  if (device_index < devices_.size() && devices_.at(device_index)) {
    return devices_.at(device_index);
  }
  auto device = std::make_shared<CudaDevice>(device_index, this);
  if (device_index >= devices_.size()) { devices_.resize(device_index + 1); }
  devices_.at(device_index) = device;
  return device;
}

size_t CudaDeviceManager::GetDeviceCount(size_t primary_device_index) {
  CudaCurrentDeviceGuard guard(primary_device_index);
  return this->GetDeviceCount();
}

size_t CudaDeviceManager::GetDeviceCount() {
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver) { return 0; }
  OF_CUDA_CHECK(err);
  return count;
}

size_t CudaDeviceManager::GetActiveDeviceIndex() {
  int device = 0;
  OF_CUDA_CHECK(cudaGetDevice(&device));
  return static_cast<size_t>(device);
}

void CudaDeviceManager::SetActiveDeviceByIndex(size_t device_index) {
  OF_CUDA_CHECK(cudaSetDevice(static_cast<int>(device_index)));
}

}  // namespace ep

}  // namespace oneflow

#endif  // WITH_CUDA
