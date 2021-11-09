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

#ifndef ONEFLOW_API_COMMON_DEVICE_H_
#define ONEFLOW_API_COMMON_DEVICE_H_

#include "oneflow/core/framework/device.h"

namespace oneflow {
struct DeviceExportUtil final {
  static Maybe<Symbol<Device>> ParseAndNew(const std::string& type_or_type_with_device_id);

  static Maybe<Symbol<Device>> New(const std::string& type, int64_t device_id);
};
}  // namespace oneflow

#endif  // !ONEFLOW_API_COMMON_DEVICE_H_
