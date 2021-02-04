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
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/to_string.h"
#include <map>

namespace oneflow {

Maybe<const char*> DeviceTag4DeviceType(DeviceType device_type) {
  auto device_type_to_tag_pairs = DeviceRegistryMgr::Get().DeviceType2TagPair();
  if (device_type_to_tag_pairs.find(device_type) == device_type_to_tag_pairs.end()) { UNIMPLEMENTED(); }
  return device_type_to_tag_pairs[device_type].c_str();
}

Maybe<DeviceType> DeviceType4DeviceTag(const std::string& device_tag) {
  auto device_tag_to_type_pairs = DeviceRegistryMgr::Get().DeviceTag2TypePair();
  if (device_tag_to_type_pairs.find(device_tag) == device_tag_to_type_pairs.end()) { UNIMPLEMENTED(); }
  return device_tag_to_type_pairs[device_tag];
}



}  // namespace oneflow
