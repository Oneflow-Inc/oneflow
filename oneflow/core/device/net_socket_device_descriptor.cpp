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

#include "oneflow/core/device/net_socket_device_descriptor.h"
#include <json.hpp>

namespace oneflow {

namespace device {

namespace {

constexpr char kJsonKeyOrdinal[] = "ordinal";
constexpr char kJsonKeyName[] = "name";
constexpr char kJsonKeyAddress[] = "address";

}  // namespace

struct NetSocketDeviceDescriptor::Impl {
  int32_t ordinal{};
  std::string name;
  std::string address;
};

NetSocketDeviceDescriptor::NetSocketDeviceDescriptor() { impl_.reset(new Impl()); }

NetSocketDeviceDescriptor::~NetSocketDeviceDescriptor() = default;

int32_t NetSocketDeviceDescriptor::Ordinal() const { return impl_->ordinal; }

const std::string& NetSocketDeviceDescriptor::Name() const { return impl_->name; }

const std::string& NetSocketDeviceDescriptor::Address() const { return impl_->address; }

void NetSocketDeviceDescriptor::Serialize(std::string* serialized) const {
  nlohmann::json json_object;
  json_object[kJsonKeyOrdinal] = impl_->ordinal;
  json_object[kJsonKeyName] = impl_->name;
  json_object[kJsonKeyAddress] = impl_->address;
  *serialized = json_object.dump(2);
}

std::shared_ptr<const NetSocketDeviceDescriptor> NetSocketDeviceDescriptor::Query(
    int32_t ordinal, const std::string& name, const std::string& address) {
  auto* desc = new NetSocketDeviceDescriptor();
  desc->impl_->ordinal = ordinal;
  desc->impl_->name = name;
  desc->impl_->address = address;
  return std::shared_ptr<const NetSocketDeviceDescriptor>(desc);
}

std::shared_ptr<const NetSocketDeviceDescriptor> NetSocketDeviceDescriptor::Deserialize(
    const std::string& serialized) {
  auto json_object = nlohmann::json::parse(serialized);
  auto* desc = new NetSocketDeviceDescriptor();
  desc->impl_->ordinal = json_object[kJsonKeyOrdinal];
  desc->impl_->name = json_object[kJsonKeyName];
  desc->impl_->address = json_object[kJsonKeyAddress];
  return std::shared_ptr<const NetSocketDeviceDescriptor>(desc);
}

}  // namespace device

}  // namespace oneflow

#endif  // __linux__
