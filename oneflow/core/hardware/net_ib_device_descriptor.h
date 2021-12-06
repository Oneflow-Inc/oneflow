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
#ifndef ONEFLOW_CORE_HARDWARE_NET_IB_DEVICE_DESCRIPTOR_H_
#define ONEFLOW_CORE_HARDWARE_NET_IB_DEVICE_DESCRIPTOR_H_

#include "oneflow/core/hardware/device_descriptor.h"
#include "oneflow/core/common/util.h"
#include <string>
#include <memory>

#ifdef WITH_RDMA

#include "oneflow/core/platform/include/ibv.h"

namespace oneflow {

namespace hardware {

constexpr char kNetIBDeviceDescriptorClassName[] = "net_ib";

enum NetIBDeviceDescriptorLinkLayer {
  kNetIBDeviceDescriptorLinkLayerInvalid = 0,
  kNetIBDeviceDescriptorLinkLayerInfiniBand = 1,
  kNetIBDeviceDescriptorLinkLayerEthernet = 2,
};

class NetIBDeviceDescriptor : public DeviceDescriptor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NetIBDeviceDescriptor);
  ~NetIBDeviceDescriptor() override;

  int32_t Ordinal() const;
  const std::string& Name() const;
  uint64_t GUID() const;
  uint8_t Port() const;
  NetIBDeviceDescriptorLinkLayer LinkLayer() const;
  const std::string& PCIBusID() const;
  void Serialize(std::string* serialized) const;
  static std::shared_ptr<const NetIBDeviceDescriptor> Query(int32_t ordinal, ibv_context* context,
                                                            uint8_t port);
  static std::shared_ptr<const NetIBDeviceDescriptor> Deserialize(const std::string& serialized);

 private:
  NetIBDeviceDescriptor();

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace hardware

}  // namespace oneflow

#endif  // WITH_RDMA

#endif  // ONEFLOW_CORE_HARDWARE_NET_IB_DEVICE_DESCRIPTOR_H_
