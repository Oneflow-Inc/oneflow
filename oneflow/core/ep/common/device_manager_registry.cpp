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
  explicit Impl(DeviceManagerRegistry* registry) : registry_(registry) {
    managers_.resize(DeviceType_ARRAYSIZE);
  }
  ~Impl() = default;

  DeviceManager* GetDeviceManagerOrNull(DeviceType device_type) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!managers_.at(device_type)) {
      std::lock_guard<std::mutex> factories_lock(*factories_mutex());
      auto& factory = factories()->at(device_type);
      if (factory) {
        managers_.at(device_type) = factory->NewDeviceManager(registry_);
      } else {
        return nullptr;
      }
    }
    return managers_.at(device_type).get();
  }

  DeviceManager* GetDeviceManager(DeviceType device_type) {
    return CHECK_NOTNULL(GetDeviceManagerOrNull(device_type));
  }

  std::shared_ptr<Device> GetDevice(DeviceType device_type, size_t device_index) {
    return GetDeviceManager(device_type)->GetDevice(device_index);
  }

  size_t GetDeviceCount(DeviceType device_type) {
    DeviceManager* manager = GetDeviceManagerOrNull(device_type);
    if (manager == nullptr) {
      return 0;
    } else {
      return manager->GetDeviceCount();
    }
  }

  size_t GetDeviceCount(const std::string& device_type_name) {
    return GetDeviceCount(GetDeviceTypeByDeviceTypeName(device_type_name));
  }

  static void DumpVersionInfo() {
    std::lock_guard<std::mutex> factories_lock(*factories_mutex());
    for (auto& factory : *factories()) {
      if (factory) { factory->DumpVersionInfo(); }
    }
  }

  static std::string GetDeviceTypeNameByDeviceType(DeviceType device_type) {
    static thread_local std::vector<std::string> device_type2device_type_name(DeviceType_ARRAYSIZE);
    {
      const std::string& name = device_type2device_type_name.at(device_type);
      if (!name.empty()) { return name; }
    }
    std::lock_guard<std::mutex> factories_lock(*factories_mutex());
    if (factories()->size() <= device_type) { return ""; }
    auto& factory = factories()->at(device_type);
    if (!factory) {
      return "";
    } else {
      std::string name = factory->device_type_name();
      device_type2device_type_name.at(device_type) = name;
      return name;
    }
  }

  static DeviceType GetDeviceTypeByDeviceTypeName(const std::string& device_type_name) {
    static thread_local HashMap<std::string, DeviceType> device_type_name2device_type;
    {
      auto it = device_type_name2device_type.find(device_type_name);
      if (it != device_type_name2device_type.end()) { return it->second; }
    }
    std::lock_guard<std::mutex> factories_lock(*factories_mutex());
    auto it = device_type_name2device_type_map()->find(device_type_name);
    if (it == device_type_name2device_type_map()->end()) {
      return DeviceType::kInvalidDevice;
    } else {
      device_type_name2device_type[device_type_name] = it->second;
      return it->second;
    }
  }

  static void RegisterDeviceManagerFactory(std::unique_ptr<DeviceManagerFactory>&& factory) {
    CHECK(factory);
    const DeviceType device_type = factory->device_type();
    std::lock_guard<std::mutex> lock(*factories_mutex());
    factories()->resize(DeviceType_ARRAYSIZE);
    CHECK(!factories()->at(device_type));
    const std::string device_type_name = factory->device_type_name();
    CHECK(!device_type_name.empty());
    CHECK(device_type_name2device_type_map()->emplace(device_type_name, device_type).second);
    factories()->at(device_type) = std::move(factory);
  }

  static std::set<DeviceType> GetRegisteredDeviceTypes() {
    std::lock_guard<std::mutex> lock(*factories_mutex());
    std::set<DeviceType> types;
    for (auto& factory : *factories()) {
      if (factory) { types.insert(factory->device_type()); }
    }
    return types;
  }

  static bool IsDeviceTypeRegistered(DeviceType device_type) {
    std::lock_guard<std::mutex> lock(*factories_mutex());
    return factories()->at(device_type).operator bool();
  }

 private:
  static HashMap<std::string, DeviceType>* device_type_name2device_type_map() {
    static HashMap<std::string, DeviceType> device_type_name2device_type;
    return &device_type_name2device_type;
  }

  static std::vector<std::unique_ptr<DeviceManagerFactory>>* factories() {
    static std::vector<std::unique_ptr<DeviceManagerFactory>> factories_vec;
    return &factories_vec;
  }

  static std::mutex* factories_mutex() {
    static std::mutex mutex;
    return &mutex;
  }

  std::mutex mutex_;
  std::vector<std::unique_ptr<DeviceManager>> managers_;
  DeviceManagerRegistry* registry_;
};

DeviceManagerRegistry::DeviceManagerRegistry() { impl_.reset(new Impl(this)); }

DeviceManagerRegistry::~DeviceManagerRegistry() = default;

DeviceManager* DeviceManagerRegistry::GetDeviceManager(DeviceType device_type) {
  return impl_->GetDeviceManager(device_type);
}

DeviceManager* DeviceManagerRegistry::GetDeviceManagerOrNull(DeviceType device_type) {
  return impl_->GetDeviceManagerOrNull(device_type);
}

std::shared_ptr<Device> DeviceManagerRegistry::GetDevice(DeviceType device_type,
                                                         size_t device_index) {
  return impl_->GetDevice(device_type, device_index);
}

size_t DeviceManagerRegistry::GetDeviceCount(DeviceType device_type) {
  return impl_->GetDeviceCount(device_type);
}

size_t DeviceManagerRegistry::GetDeviceCount(const std::string& device_type_name) {
  return impl_->GetDeviceCount(device_type_name);
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

/*static*/ std::set<DeviceType> DeviceManagerRegistry::GetRegisteredDeviceTypes() {
  return Impl::GetRegisteredDeviceTypes();
}

/*static*/ bool DeviceManagerRegistry::IsDeviceTypeRegistered(DeviceType device_type) {
  return Impl::IsDeviceTypeRegistered(device_type);
}

}  // namespace ep

}  // namespace oneflow
