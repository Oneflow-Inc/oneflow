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
#include "oneflow/core/hardware/device_descriptor_class.h"
#include <mutex>
#include <utility>
#include <vector>
#include <unordered_map>

namespace oneflow {

namespace hardware {

namespace {

class DeviceClassRegistryStorage {
 public:
  DeviceClassRegistryStorage() = default;
  ~DeviceClassRegistryStorage() = default;
  void Register(std::shared_ptr<const DeviceDescriptorClass> descriptor_class) {
    std::lock_guard<std::mutex> lock(mutex_);
    const std::string name = descriptor_class->Name();
    if (!name2index_.emplace(name, classes_.size()).second) { abort(); }
    classes_.emplace_back(std::make_shared<std::string>(name), std::move(descriptor_class));
  }

  size_t RegisteredCount() {
    std::lock_guard<std::mutex> lock(mutex_);
    return classes_.size();
  }

  const std::string& GetRegisteredClass(size_t index) {
    std::lock_guard<std::mutex> lock(mutex_);
    return *classes_.at(index).first;
  }

  std::shared_ptr<const DeviceDescriptorClass> GetRegistered(size_t index) {
    std::lock_guard<std::mutex> lock(mutex_);
    return classes_.at(index).second;
  }

  std::shared_ptr<const DeviceDescriptorClass> GetRegistered(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = name2index_.find(name);
    if (it == name2index_.end()) { return std::shared_ptr<const DeviceDescriptorClass>(); }
    return classes_.at(it->second).second;
  }

  static DeviceClassRegistryStorage& Instance() {
    static DeviceClassRegistryStorage instance;
    return instance;
  }

 private:
  std::unordered_map<std::string, size_t> name2index_;
  std::vector<std::pair<std::shared_ptr<std::string>, std::shared_ptr<const DeviceDescriptorClass>>>
      classes_;
  std::mutex mutex_;
};

}  // namespace

void DeviceDescriptorClass::RegisterClass(
    std::shared_ptr<const DeviceDescriptorClass> descriptor_class) {
  DeviceClassRegistryStorage::Instance().Register(std::move(descriptor_class));
}

size_t DeviceDescriptorClass::GetRegisteredClassesCount() {
  return DeviceClassRegistryStorage::Instance().RegisteredCount();
}

std::shared_ptr<const DeviceDescriptorClass> DeviceDescriptorClass::GetRegisteredClass(
    size_t index) {
  return DeviceClassRegistryStorage::Instance().GetRegistered(index);
}

std::shared_ptr<const DeviceDescriptorClass> DeviceDescriptorClass::GetRegisteredClass(
    const std::string& class_name) {
  return DeviceClassRegistryStorage::Instance().GetRegistered(class_name);
}

}  // namespace hardware

}  // namespace oneflow
