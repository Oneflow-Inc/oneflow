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
#include "oneflow/core/ep/include/device_manager_registry.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/ep/include/device_manager.h"

namespace oneflow {

namespace ep {

class DeviceManagerRegistry::Impl {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Impl);
  Impl() { managers_.resize(DeviceType_ARRAYSIZE); }
  ~Impl() = default;

  DeviceManager* GetDeviceManager(DeviceType device_type) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!managers_.at(device_type)) {
      CHECK((IsClassRegistered<int32_t, DeviceManager>(device_type)));
      auto* manager = NewObj<int32_t, DeviceManager>(device_type);
      managers_.at(device_type).reset(manager);
    }
    return managers_.at(device_type).get();
  }
  std::shared_ptr<Device> GetDevice(DeviceType device_type, size_t device_index) {
    return GetDeviceManager(device_type)->GetDevice(device_index);
  }

 private:
  std::mutex mutex_;
  std::vector<std::unique_ptr<DeviceManager>> managers_;
};

DeviceManagerRegistry::DeviceManagerRegistry() { impl_.reset(new Impl()); }

DeviceManagerRegistry::~DeviceManagerRegistry() = default;

DeviceManager* DeviceManagerRegistry::GetDeviceManager(DeviceType device_type) {
  return impl_->GetDeviceManager(device_type);
}

std::shared_ptr<Device> DeviceManagerRegistry::GetDevice(DeviceType device_type,
                                                         size_t device_index) {
  return impl_->GetDevice(device_type, device_index);
}

}  // namespace ep

}  // namespace oneflow
