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

#include "oneflow/core/device/device_descriptor_class.h"
#include "oneflow/core/device/rocm_device_descriptor.h"
#include "oneflow/core/device/basic_device_descriptor_list.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/device/rocm_util.h"
#include "oneflow/core/persistence/tee_persistent_log_stream.h"
#include "oneflow/core/common/str_util.h"
#include <json.hpp>

#ifdef WITH_HIP

namespace oneflow {

namespace device {

namespace {

constexpr char kJsonKeyDevices[] = "devices";

}  // namespace

class RocmDeviceDescriptorClass : public DeviceDescriptorClass {
 public:
  RocmDeviceDescriptorClass() = default;
  ~RocmDeviceDescriptorClass() override = default;

  std::shared_ptr<const DeviceDescriptorList> QueryDeviceDescriptorList() const override {
    int n_dev;
    hipError_t err = hipGetDeviceCount(&n_dev);
    if (err != hipSuccess) {
      LOG(WARNING) << hipGetErrorString(err);
      return std::make_shared<const BasicDeviceDescriptorList>(
          std::vector<std::shared_ptr<const DeviceDescriptor>>());
    }
    OF_ROCM_CHECK(err);
    std::vector<std::shared_ptr<const DeviceDescriptor>> devices(n_dev);
    for (int dev = 0; dev < n_dev; ++dev) { devices.at(dev) = RocmDeviceDescriptor::Query(dev); }
    return std::make_shared<const BasicDeviceDescriptorList>(devices);
  }

  std::string Name() const override { return kRocmDeviceDescriptorClassName; }

  void SerializeDeviceDescriptorList(const std::shared_ptr<const DeviceDescriptorList>& list,
                                     std::string* serialized) const override {
    std::vector<std::string> serialized_devices;
    serialized_devices.reserve(list->DeviceCount());
    for (size_t i = 0; i < list->DeviceCount(); ++i) {
      auto rocm_device = std::dynamic_pointer_cast<const RocmDeviceDescriptor>(list->GetDevice(i));
      CHECK(rocm_device);
      std::string serialized_device;
      rocm_device->Serialize(&serialized_device);
      serialized_devices.push_back(std::move(serialized_device));
    }
    nlohmann::json json_object;
    json_object[kJsonKeyDevices] = serialized_devices;
    *serialized = json_object.dump();
  }

  std::shared_ptr<const DeviceDescriptorList> DeserializeDeviceDescriptorList(
      const std::string& serialized) const override {
    auto json_object = nlohmann::json::parse(serialized);
    std::vector<std::string> serialized_devices = json_object[kJsonKeyDevices];
    std::vector<std::shared_ptr<const DeviceDescriptor>> devices(serialized_devices.size());
    for (int i = 0; i < serialized_devices.size(); ++i) {
      devices.at(i) = RocmDeviceDescriptor::Deserialize(serialized_devices.at(i));
    }
    return std::make_shared<const BasicDeviceDescriptorList>(devices);
  }

  void DumpDeviceDescriptorListSummary(const std::shared_ptr<const DeviceDescriptorList>& list,
                                       const std::string& path) const override {
    for (size_t i = 0; i < list->DeviceCount(); ++i) {
      auto rocm_device = std::dynamic_pointer_cast<const RocmDeviceDescriptor>(list->GetDevice(i));
      CHECK(rocm_device);
      auto stream = TeePersistentLogStream::Create(JoinPath(path, std::to_string(i) + ".json"));
      std::string serialized;
      rocm_device->Serialize(&serialized);
      stream << serialized;
    }
  }
};

COMMAND(DeviceDescriptorClass::RegisterClass(std::make_shared<RocmDeviceDescriptorClass>()));

}  // namespace device

}  // namespace oneflow

#endif  // WITH_HIP
