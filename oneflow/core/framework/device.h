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
#ifndef ONEFLOW_CORE_FRAMEWORK_DEVICE_H_
#define ONEFLOW_CORE_FRAMEWORK_DEVICE_H_

#include <string>

namespace oneflow {

class Device final {
 public:
  Device(const std::string& type, int64_t device_id) : type_(type), device_id_(device_id) {}
  Device(const Device&) = default;
  Device(Device&&) = default;
  ~Device() = default;
  const std::string& type() const { return type_; }
  int64_t device_id() const { return device_id_; }

 private:
  const std::string type_;
  const int64_t device_id_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_DEVICE_H_
