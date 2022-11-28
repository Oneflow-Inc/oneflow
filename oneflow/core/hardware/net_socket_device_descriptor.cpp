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

#include "oneflow/core/hardware/net_socket_device_descriptor.h"
#include "nlohmann/json.hpp"

namespace oneflow {

namespace hardware {

namespace {

constexpr char kJsonKeyOrdinal[] = "ordinal";
constexpr char kJsonKeyName[] = "name";
constexpr char kJsonKeyAddress[] = "address";
constexpr char kJsonKeyPCIBusID[] = "pci_bus_id";

void GetPCIBusID(const std::string& name, std::string* pci_bus_id) {
#ifdef __linux__
  const std::string device_path = "/sys/class/net/" + name + "/device";
  char* device_real_path = realpath(device_path.data(), nullptr);
  if (device_real_path == nullptr) { return; }
  const std::string device_real_path_str = device_real_path;
  free(device_real_path);  // NOLINT
  const size_t pos = device_real_path_str.rfind('/');
  if (pos == std::string::npos) { return; }
  *pci_bus_id = device_real_path_str.substr(pos + 1);
#endif
}

}  // namespace

struct NetSocketDeviceDescriptor::Impl {
  int32_t ordinal{};
  std::string name;
  std::string address;
  std::string pci_bus_id;
};

NetSocketDeviceDescriptor::NetSocketDeviceDescriptor() { impl_.reset(new Impl()); }

NetSocketDeviceDescriptor::~NetSocketDeviceDescriptor() = default;

int32_t NetSocketDeviceDescriptor::Ordinal() const { return impl_->ordinal; }

const std::string& NetSocketDeviceDescriptor::Name() const { return impl_->name; }

const std::string& NetSocketDeviceDescriptor::Address() const { return impl_->address; }

const std::string& NetSocketDeviceDescriptor::PCIBusID() const { return impl_->pci_bus_id; }

void NetSocketDeviceDescriptor::Serialize(std::string* serialized) const {
  nlohmann::json json_object;
  json_object[kJsonKeyOrdinal] = impl_->ordinal;
  json_object[kJsonKeyName] = impl_->name;
  json_object[kJsonKeyAddress] = impl_->address;
  json_object[kJsonKeyPCIBusID] = impl_->pci_bus_id;
  *serialized = json_object.dump(2);
}

std::shared_ptr<const NetSocketDeviceDescriptor> NetSocketDeviceDescriptor::Query(
    int32_t ordinal, const std::string& name, const std::string& address) {
  auto* desc = new NetSocketDeviceDescriptor();
  desc->impl_->ordinal = ordinal;
  desc->impl_->name = name;
  desc->impl_->address = address;
  GetPCIBusID(name, &desc->impl_->pci_bus_id);
  return std::shared_ptr<const NetSocketDeviceDescriptor>(desc);
}

std::shared_ptr<const NetSocketDeviceDescriptor> NetSocketDeviceDescriptor::Deserialize(
    const std::string& serialized) {
  auto json_object = nlohmann::json::parse(serialized);
  auto* desc = new NetSocketDeviceDescriptor();
  desc->impl_->ordinal = json_object[kJsonKeyOrdinal];
  desc->impl_->name = json_object[kJsonKeyName];
  desc->impl_->address = json_object[kJsonKeyAddress];
  desc->impl_->pci_bus_id = json_object[kJsonKeyPCIBusID];
  return std::shared_ptr<const NetSocketDeviceDescriptor>(desc);
}

}  // namespace hardware

}  // namespace oneflow

#endif  // __linux__
