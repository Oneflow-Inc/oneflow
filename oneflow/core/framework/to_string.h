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
#ifndef ONEFLOW_CORE_FRAMEWORK_TO_STRING_H_
#define ONEFLOW_CORE_FRAMEWORK_TO_STRING_H_

#include "oneflow/core/common/to_string.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/common/maybe.h"

namespace oneflow {

Maybe<const char*> DeviceTag4DeviceType(DeviceType device_type);
Maybe<DeviceType> DeviceType4DeviceTag(const std::string& device_tag);

template<>
inline std::string ToString(const DataType& data_type) {
  return DataType_Name(data_type);
}

template<>
inline std::string ToString(const DeviceType& device_type) {
  return CHECK_JUST(DeviceTag4DeviceType(device_type));
}

template<>
inline std::string ToString(const std::string& value) {
  return value;
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_TO_STRING_H_
