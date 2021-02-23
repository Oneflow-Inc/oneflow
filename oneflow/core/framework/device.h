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
#ifndef ONEFLOW_FRAMEWORK_CORE_DEVICE_H_
#define ONEFLOW_FRAMEWORK_CORE_DEVICE_H_

#include "oneflow/core/common/device_type.pb.h"

namespace oneflow {
namespace one {
class Device {
 public:
  Device(const std::string& type, int64_t device_id)
      : type_(type), device_id_(device_id) {}
  std::string type() const { return type_; }
  int64_t device_id() const { return device_id_; }

 private:
  std::string type_;
  int64_t device_id_;
};
}  // namespace one
}  // namespace oneflow
#endif

