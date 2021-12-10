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
      std::lock_guard<std::mutex> factories_lock(factories_mutex_);
      auto& factory = factories_.at(device_type);
      CHECK(factory);
      managers_.at(device_type) = factory->NewDeviceManager();
    }
    return managers_.at(device_type).get();
  }

  std::shared_ptr<Device> GetDevice(DeviceType device_type, size_t device_index) {
    return GetDeviceManager(device_type)->GetDevice(device_index);
  }

  static void DumpVersionInfo() {
    std::lock_guard<std::mutex> factories_lock(factories_mutex_);
    for (auto& factory : factories_) {
      if (factory) { factory->DumpVersionInfo(); }
    }
  }

  static std::string GetDeviceTypeNameByDeviceType(DeviceType device_type) {
    std::lock_guard<std::mutex> factories_lock(factories_mutex_);
    if (factories_.size() <= device_type) { return ""; }
    auto& factory = factories_.at(device_type);
    if (!factory) {
      return "";
    } else {
      return factory->device_type_name();
    }
  }

  static DeviceType GetDeviceTypeByDeviceTypeName(const std::string& device_type_name) {
    std::lock_guard<std::mutex> factories_lock(factories_mutex_);
    auto it = device_type_name2device_type_.find(device_type_name);
    if (it == device_type_name2device_type_.end()) {
      return DeviceType::kInvalidDevice;
    } else {
      return it->second;
    }
  }

  static void RegisterDeviceManagerFactory(std::unique_ptr<DeviceManagerFactory>&& factory) {
    CHECK(factory);
    const DeviceType device_type = factory->device_type();
    std::lock_guard<std::mutex> lock(factories_mutex_);
    factories_.resize(DeviceType_ARRAYSIZE);
    CHECK(!factories_.at(device_type));
    const std::string device_type_name = factory->device_type_name();
    CHECK(!device_type_name.empty());
    CHECK(device_type_name2device_type_.emplace(device_type_name, device_type).second);
    factories_.at(device_type) = std::move(factory);
  }

 private:
  std::mutex mutex_;
  std::vector<std::unique_ptr<DeviceManager>> managers_;
  static std::vector<std::unique_ptr<DeviceManagerFactory>> factories_;
  static HashMap<std::string, DeviceType> device_type_name2device_type_;
  static std::mutex factories_mutex_;
};

std::vector<std::unique_ptr<DeviceManagerFactory>> DeviceManagerRegistry::Impl::factories_;
HashMap<std::string, DeviceType> DeviceManagerRegistry::Impl::device_type_name2device_type_;
std::mutex DeviceManagerRegistry::Impl::factories_mutex_;

DeviceManagerRegistry::DeviceManagerRegistry() { impl_.reset(new Impl()); }

DeviceManagerRegistry::~DeviceManagerRegistry() = default;

DeviceManager* DeviceManagerRegistry::GetDeviceManager(DeviceType device_type) {
  return impl_->GetDeviceManager(device_type);
}

std::shared_ptr<Device> DeviceManagerRegistry::GetDevice(DeviceType device_type,
                                                         size_t device_index) {
  return impl_->GetDevice(device_type, device_index);
}

/*static*/ void DeviceManagerRegistry::RegisterDeviceManagerFactory(
    std::unique_ptr<DeviceManagerFactory>&& factory) {
  Impl::RegisterDeviceManagerFactory(std::move(factory));
}

/*static*/ void DeviceManagerRegistry::DumpVersionInfo() { Impl::DumpVersionInfo(); }

/*static*/ std::string DeviceManagerRegistry::GetDeviceTypeNameByDeviceType(
    DeviceType device_type) {
  return Impl::GetDeviceTypeNameByDeviceType(device_type);
}

/*static*/ DeviceType DeviceManagerRegistry::GetDeviceTypeByDeviceTypeName(
    const std::string& device_type_name) {
  return Impl::GetDeviceTypeByDeviceTypeName(device_type_name);
}

}  // namespace ep

}  // namespace oneflow
