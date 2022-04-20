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

#ifdef __linux__

#include "oneflow/core/hardware/device_descriptor_class.h"
#include "oneflow/core/hardware/net_socket_device_descriptor.h"
#include "oneflow/core/hardware/basic_device_descriptor_list.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/persistence/tee_persistent_log_stream.h"
#include "oneflow/core/common/str_util.h"
#include "nlohmann/json.hpp"
#include <ifaddrs.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>

namespace oneflow {

namespace hardware {

namespace {

constexpr char kJsonKeyDevices[] = "devices";

}  // namespace

class NetSocketDeviceDescriptorClass : public DeviceDescriptorClass {
 public:
  NetSocketDeviceDescriptorClass() = default;
  ~NetSocketDeviceDescriptorClass() override = default;

  std::shared_ptr<const DeviceDescriptorList> QueryDeviceDescriptorList() const override {
    std::vector<std::shared_ptr<const NetSocketDeviceDescriptor>> devices;
    ifaddrs* interfaces = nullptr;
    if (getifaddrs(&interfaces) != 0) {
      return std::make_shared<const BasicDeviceDescriptorList>();
    }
    ifaddrs* ifa = nullptr;
    for (ifa = interfaces; ifa != nullptr; ifa = ifa->ifa_next) {
      if (ifa->ifa_addr == nullptr) { continue; }
      const std::string name(ifa->ifa_name);
      if (name == "lo") { continue; }
      // TODO(liujuncheng): support ipv6
      if (ifa->ifa_addr->sa_family != AF_INET) { continue; }
      if (std::count_if(devices.cbegin(), devices.cend(),
                        [&](const std::shared_ptr<const NetSocketDeviceDescriptor>& device) {
                          return device->Name() == name;
                        })
          != 0) {
        continue;
      }
      char host[NI_MAXHOST];
      const socklen_t sa_len = (ifa->ifa_addr->sa_family == AF_INET) ? sizeof(struct sockaddr_in)
                                                                     : sizeof(struct sockaddr_in6);
      if (getnameinfo(ifa->ifa_addr, sa_len, host, NI_MAXHOST, nullptr, 0, NI_NUMERICHOST) != 0) {
        continue;
      }
      auto socket_device =
          NetSocketDeviceDescriptor::Query(static_cast<int32_t>(devices.size()), name, host);
      if (socket_device) { devices.emplace_back(socket_device); }
    }
    freeifaddrs(interfaces);
    return std::make_shared<const BasicDeviceDescriptorList>(
        std::vector<std::shared_ptr<const DeviceDescriptor>>{devices.begin(), devices.end()});
  }

  std::string Name() const override { return kNetSocketDeviceDescriptorClassName; }

  void SerializeDeviceDescriptorList(const std::shared_ptr<const DeviceDescriptorList>& list,
                                     std::string* serialized) const override {
    std::vector<std::string> serialized_devices;
    serialized_devices.reserve(list->DeviceCount());
    for (size_t i = 0; i < list->DeviceCount(); ++i) {
      auto socket_device =
          std::dynamic_pointer_cast<const NetSocketDeviceDescriptor>(list->GetDevice(i));
      CHECK(socket_device);
      std::string serialized_device;
      socket_device->Serialize(&serialized_device);
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
      devices.at(i) = NetSocketDeviceDescriptor::Deserialize(serialized_devices.at(i));
    }
    return std::make_shared<const BasicDeviceDescriptorList>(devices);
  }

  void DumpDeviceDescriptorListSummary(const std::shared_ptr<const DeviceDescriptorList>& list,
                                       const std::string& path) const override {
    for (size_t i = 0; i < list->DeviceCount(); ++i) {
      auto socket_device =
          std::dynamic_pointer_cast<const NetSocketDeviceDescriptor>(list->GetDevice(i));
      CHECK(socket_device);
      auto stream = TeePersistentLogStream::Create(JoinPath(path, std::to_string(i) + ".json"));
      std::string serialized;
      socket_device->Serialize(&serialized);
      stream << serialized;
    }
  }
};

COMMAND(DeviceDescriptorClass::RegisterClass(std::make_shared<NetSocketDeviceDescriptorClass>()));

}  // namespace hardware

}  // namespace oneflow

#endif  // __linux__
