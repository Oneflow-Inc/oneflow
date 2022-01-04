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
#ifndef ONEFLOW_API_CPP_FRAMEWORK_DEVICE_H_
#define ONEFLOW_API_CPP_FRAMEWORK_DEVICE_H_

#include <string>
#include <memory>

namespace oneflow {

class Device;

template<typename T>
class Symbol;

}  // namespace oneflow

namespace oneflow_api {

class Device final {
  friend class Tensor;
  friend class Graph;

 public:
  explicit Device(const std::string& type_or_type_with_device_id);
  explicit Device(const std::string& type, int64_t device_id);
  [[nodiscard]] const std::string& type() const;
  [[nodiscard]] int64_t device_id() const;

  [[nodiscard]] bool operator==(const Device& rhs) const;
  [[nodiscard]] bool operator!=(const Device& rhs) const;

 private:
  std::shared_ptr<oneflow::Symbol<oneflow::Device>> device_ = nullptr;
};

}  // namespace oneflow_api

#endif  // !ONEFLOW_API_CPP_FRAMEWORK_DEVICE_H_
