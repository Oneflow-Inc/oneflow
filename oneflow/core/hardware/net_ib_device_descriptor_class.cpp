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
#include "oneflow/core/hardware/net_ib_device_descriptor.h"
#include "oneflow/core/hardware/basic_device_descriptor_list.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/persistence/tee_persistent_log_stream.h"
#include "oneflow/core/common/str_util.h"
#include "nlohmann/json.hpp"

#ifdef WITH_RDMA

namespace oneflow {

namespace hardware {

namespace {

constexpr char kJsonKeyDevices[] = "devices";

}  // namespace

class NetIBDeviceDescriptorClass : public DeviceDescriptorClass {
 public:
  NetIBDeviceDescriptorClass() = default;
  ~NetIBDeviceDescriptorClass() override = default;

  std::shared_ptr<const DeviceDescriptorList> QueryDeviceDescriptorList() const override {
    std::vector<std::shared_ptr<const DeviceDescriptor>> devices;
    int num_devices;
    if (!ibv::IsAvailable()) { return std::make_shared<const BasicDeviceDescriptorList>(devices); }
    ibv_device** device_list = ibv::wrapper.ibv_get_device_list(&num_devices);
    if (device_list == nullptr) {
      return std::make_shared<const BasicDeviceDescriptorList>(devices);
    }
    for (int i = 0; i < num_devices; ++i) {
      ibv_device* device = device_list[i];
      ibv_context* context = ibv::wrapper.ibv_open_device(device);
      if (context == nullptr) { continue; }
      ibv_device_attr device_attr{};
      if (ibv::wrapper.ibv_query_device(context, &device_attr) != 0) {
        CHECK_EQ(ibv::wrapper.ibv_close_device(context), 0);
      }
      for (int port = 1; port <= device_attr.phys_port_cnt; ++port) {
        auto device_desc =
            NetIBDeviceDescriptor::Query(static_cast<int32_t>(devices.size()), context, port);
        if (device_desc) { devices.emplace_back(device_desc); }
      }
    }
    ibv::wrapper.ibv_free_device_list(device_list);
    return std::make_shared<const BasicDeviceDescriptorList>(devices);
  }

  std::string Name() const override { return kNetIBDeviceDescriptorClassName; }

  void SerializeDeviceDescriptorList(const std::shared_ptr<const DeviceDescriptorList>& list,
                                     std::string* serialized) const override {
    std::vector<std::string> serialized_devices;
    serialized_devices.reserve(list->DeviceCount());
    for (size_t i = 0; i < list->DeviceCount(); ++i) {
      auto ib_device = std::dynamic_pointer_cast<const NetIBDeviceDescriptor>(list->GetDevice(i));
      CHECK(ib_device);
      std::string serialized_device;
      ib_device->Serialize(&serialized_device);
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
      devices.at(i) = NetIBDeviceDescriptor::Deserialize(serialized_devices.at(i));
    }
    return std::make_shared<const BasicDeviceDescriptorList>(devices);
  }

  void DumpDeviceDescriptorListSummary(const std::shared_ptr<const DeviceDescriptorList>& list,
                                       const std::string& path) const override {
    for (size_t i = 0; i < list->DeviceCount(); ++i) {
      auto ib_device = std::dynamic_pointer_cast<const NetIBDeviceDescriptor>(list->GetDevice(i));
      CHECK(ib_device);
      auto stream = TeePersistentLogStream::Create(JoinPath(path, std::to_string(i) + ".json"));
      std::string serialized;
      ib_device->Serialize(&serialized);
      stream << serialized;
    }
  }
};

COMMAND(DeviceDescriptorClass::RegisterClass(std::make_shared<NetIBDeviceDescriptorClass>()));

}  // namespace hardware

}  // namespace oneflow

#endif  // WITH_RDMA
