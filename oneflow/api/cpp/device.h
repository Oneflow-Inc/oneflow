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
#ifndef ONEFLOW_API_CPP_DEVICE_H_
#define ONEFLOW_API_CPP_DEVICE_H_

#include <string>
#include <memory>

namespace oneflow_api {
class Device final {
 public:
  struct Impl;
  ~Device() = default;
  explicit Device(const std::shared_ptr<Impl>& impl) : impl_(impl) {}
  explicit Device(std::shared_ptr<Impl>&& impl) : impl_(std::move(impl)) {}
  Device(const Device& device) : impl_(device.impl_) {}
  Device(Device&& device) noexcept : impl_(std::move(device.impl_)) {}
  Device& operator=(const Device& device) {
    this->impl_.reset();
    this->impl_ = device.impl_;
    return *this;
  }
  Device& operator=(Device&& device) noexcept {
    this->impl_.reset();
    this->impl_ = std::move(device.impl_);
    return *this;
  }

  static void CheckDeviceType(const std::string& type);
  static std::shared_ptr<Device> ParseAndNew(const std::string& type_and_id);
  static std::shared_ptr<Device> New(const std::string& type);
  static std::shared_ptr<Device> New(const std::string& type, int64_t device_id);

 private:
  std::shared_ptr<Impl> impl_;
};
}  // namespace oneflow_api

#endif  // !ONEFLOW_API_CPP_DEVICE_H_
