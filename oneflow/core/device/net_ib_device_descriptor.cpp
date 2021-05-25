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
#include "oneflow/core/device/net_ib_device_descriptor.h"

#ifdef WITH_RDMA

#include <json.hpp>

namespace oneflow {

namespace device {

namespace {

constexpr char kJsonKeyOrdinal[] = "ordinal";
constexpr char kJsonKeyName[] = "name";
constexpr char kJsonKeyGUID[] = "guid";
constexpr char kJsonKeyPort[] = "port";
constexpr char kJsonKeyLankLayer[] = "link_layer";
constexpr char kJsonValueLinkLayerInfiniBand[] = "InfiniBand";

}  // namespace

struct NetIBDeviceDescriptor::Impl {
  int32_t ordinal{};
  std::string name;
  uint64_t guid{};
  uint8_t port{};
  NetIBDeviceDescriptorLinkLayer link_layer{};
};

NetIBDeviceDescriptor::NetIBDeviceDescriptor() { impl_.reset(new Impl()); }

NetIBDeviceDescriptor::~NetIBDeviceDescriptor() = default;

int32_t NetIBDeviceDescriptor::Ordinal() const { return impl_->ordinal; }

const std::string& NetIBDeviceDescriptor::Name() const { return impl_->name; }

uint64_t NetIBDeviceDescriptor::GUID() const { return impl_->guid; }

uint8_t NetIBDeviceDescriptor::Port() const { return impl_->port; }

NetIBDeviceDescriptorLinkLayer NetIBDeviceDescriptor::LinkLayer() const {
  return impl_->link_layer;
}

void NetIBDeviceDescriptor::Serialize(std::string* serialized) const {
  nlohmann::json json_object;
  json_object[kJsonKeyOrdinal] = impl_->ordinal;
  json_object[kJsonKeyName] = impl_->name;
  json_object[kJsonKeyGUID] = impl_->guid;
  json_object[kJsonKeyPort] = impl_->port;
  if (impl_->link_layer == kNetIBDeviceDescriptorLinkLayerInfiniBand) {
    json_object[kJsonKeyLankLayer] = kJsonValueLinkLayerInfiniBand;
  } else {
    UNIMPLEMENTED();
  }
  *serialized = json_object.dump(2);
}

std::shared_ptr<const NetIBDeviceDescriptor> NetIBDeviceDescriptor::Query(int32_t ordinal,
                                                                          ibv_context* context,
                                                                          uint8_t port) {
  CHECK(ibv::IsAvailable());
  ibv_device_attr device_attr{};
  if (ibv::wrapper.ibv_query_device(context, &device_attr) != 0) {
    LOG(WARNING) << "Unable to query device: " << context->device->name;
    return std::shared_ptr<const NetIBDeviceDescriptor>();
  }
  ibv_port_attr port_attr{};
  if (ibv::wrapper.ibv_query_port_wrap(context, port, &port_attr) != 0) {
    LOG(WARNING) << "Unable to query port: device " << context->device->name << " port " << port;
    return std::shared_ptr<const NetIBDeviceDescriptor>();
  }
  if (port_attr.state != IBV_PORT_ACTIVE) {
    LOG(WARNING) << "Inactivate port: device " << context->device->name << " port " << port;
    return std::shared_ptr<const NetIBDeviceDescriptor>();
  }
  // TODO(liujuncheng): Add IBV_LINK_LAYER_ETHERNET support
  if (port_attr.link_layer != IBV_LINK_LAYER_INFINIBAND) {
    LOG(WARNING) << "Link layer is not supported: device " << context->device->name << " port "
                 << port;
    return std::shared_ptr<const NetIBDeviceDescriptor>();
  }
  auto* desc = new NetIBDeviceDescriptor();
  desc->impl_->ordinal = ordinal;
  desc->impl_->name = context->device->name;
  desc->impl_->guid = device_attr.sys_image_guid;
  desc->impl_->port = port;
  if (port_attr.link_layer == IBV_LINK_LAYER_INFINIBAND) {
    desc->impl_->link_layer = kNetIBDeviceDescriptorLinkLayerInfiniBand;
  } else {
    UNIMPLEMENTED();
  }
  return std::shared_ptr<const NetIBDeviceDescriptor>(desc);
}

std::shared_ptr<const NetIBDeviceDescriptor> NetIBDeviceDescriptor::Deserialize(
    const std::string& serialized) {
  auto json_object = nlohmann::json::parse(serialized);
  auto* desc = new NetIBDeviceDescriptor();
  desc->impl_->ordinal = json_object[kJsonKeyOrdinal];
  desc->impl_->name = json_object[kJsonKeyName];
  desc->impl_->guid = json_object[kJsonKeyGUID];
  desc->impl_->port = json_object[kJsonKeyPort];
  const std::string link_layer_value = json_object[kJsonKeyLankLayer];
  if (link_layer_value == kJsonValueLinkLayerInfiniBand) {
    desc->impl_->link_layer = kNetIBDeviceDescriptorLinkLayerInfiniBand;
  } else {
    UNIMPLEMENTED();
  }
  return std::shared_ptr<const NetIBDeviceDescriptor>(desc);
}

}  // namespace device

}  // namespace oneflow

#endif  // WITH_RDMA
