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
#include "oneflow/core/hardware/cuda_device_descriptor.h"
#include "oneflow/core/hardware/basic_device_descriptor_list.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/persistence/tee_persistent_log_stream.h"
#include "oneflow/core/common/str_util.h"
#include "nlohmann/json.hpp"

#ifdef WITH_CUDA

#include <cuda_runtime.h>

namespace oneflow {

namespace hardware {

namespace {

constexpr char kJsonKeyDevices[] = "devices";

}  // namespace

class CudaDeviceDescriptorClass : public DeviceDescriptorClass {
 public:
  CudaDeviceDescriptorClass() = default;
  ~CudaDeviceDescriptorClass() override = default;

  std::shared_ptr<const DeviceDescriptorList> QueryDeviceDescriptorList() const override {
    int n_dev = 0;
    cudaError_t err = cudaGetDeviceCount(&n_dev);
    if (err != cudaSuccess) {
      LOG(WARNING) << cudaGetErrorString(err);
      return std::make_shared<const BasicDeviceDescriptorList>(
          std::vector<std::shared_ptr<const DeviceDescriptor>>());
    }
    std::vector<std::shared_ptr<const DeviceDescriptor>> devices(n_dev);
    for (int dev = 0; dev < n_dev; ++dev) { devices.at(dev) = CudaDeviceDescriptor::Query(dev); }
    return std::make_shared<const BasicDeviceDescriptorList>(devices);
  }

  std::string Name() const override { return kCudaDeviceDescriptorClassName; }

  void SerializeDeviceDescriptorList(const std::shared_ptr<const DeviceDescriptorList>& list,
                                     std::string* serialized) const override {
    std::vector<std::string> serialized_devices;
    serialized_devices.reserve(list->DeviceCount());
    for (size_t i = 0; i < list->DeviceCount(); ++i) {
      auto cuda_device = std::dynamic_pointer_cast<const CudaDeviceDescriptor>(list->GetDevice(i));
      CHECK(cuda_device);
      std::string serialized_device;
      cuda_device->Serialize(&serialized_device);
      serialized_devices.emplace_back(std::move(serialized_device));
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
      devices.at(i) = CudaDeviceDescriptor::Deserialize(serialized_devices.at(i));
    }
    return std::make_shared<const BasicDeviceDescriptorList>(devices);
  }

  void DumpDeviceDescriptorListSummary(const std::shared_ptr<const DeviceDescriptorList>& list,
                                       const std::string& path) const override {
    for (size_t i = 0; i < list->DeviceCount(); ++i) {
      auto cuda_device = std::dynamic_pointer_cast<const CudaDeviceDescriptor>(list->GetDevice(i));
      CHECK(cuda_device);
      auto stream = TeePersistentLogStream::Create(JoinPath(path, std::to_string(i) + ".json"));
      std::string serialized;
      cuda_device->Serialize(&serialized);
      stream << serialized;
    }
  }
};

COMMAND(DeviceDescriptorClass::RegisterClass(std::make_shared<CudaDeviceDescriptorClass>()));

}  // namespace hardware

}  // namespace oneflow

#endif  // WITH_CUDA
