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

#include "oneflow/api/common/device.h"

namespace oneflow {

namespace {

void CheckDeviceType(const std::string& type) {
  if (Device::type_supported.find(type) == Device::type_supported.end()) {
    std::string error_msg =
        "Expected one of cpu, cuda device type at start of device string " + type;
    throw std::runtime_error(error_msg);
  }
}

}  // namespace

/* static */ Maybe<Symbol<Device>> DeviceExportUtil::ParseAndNew(
    const std::string& type_or_type_with_device_id) {
  std::string type;
  int device_id = -1;
  ParsingDeviceTag(type_or_type_with_device_id, &type, &device_id).GetOrThrow();
  CheckDeviceType(type);
  if (device_id == -1) {
    return Device::New(type);
  } else {
    return Device::New(type, device_id);
  }
}

/* static */ Maybe<Symbol<Device>> DeviceExportUtil::New(const std::string& type,
                                                         int64_t device_id) {
  CheckDeviceType(type);
  return Device::New(type, device_id);
}

}  // namespace oneflow
