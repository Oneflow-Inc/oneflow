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

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/framework/device.h"
#include "device.h"

namespace ofapi {

namespace of = oneflow;

void CheckDeviceType(const std::string& type) {
  if (of::Device::type_supported.find(type) == of::Device::type_supported.end()) {
    std::string error_msg =
        "Expected one of cpu, cuda device type at start of device string " + type;
    throw std::runtime_error(error_msg);
  }
}

Device::Device(const std::string& type_or_type_with_device_id) {
  std::string type;
  int device_id = -1;
  oneflow::ParsingDeviceTag(type_or_type_with_device_id, &type, &device_id).GetOrThrow();
  if (device_id == -1) {
    CheckDeviceType(type);
    device_ = std::make_shared<of::Symbol<of::Device>>(of::Device::New(type).GetOrThrow());
  } else {
    device_ =
        std::make_shared<of::Symbol<of::Device>>(of::Device::New(type, device_id).GetOrThrow());
  }
}

Device::Device(const std::string& type, int64_t device_id) {
  CheckDeviceType(type);
  device_ = std::make_shared<of::Symbol<of::Device>>(of::Device::New(type, device_id).GetOrThrow());
}

}  // namespace ofapi
