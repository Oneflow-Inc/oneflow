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
#ifndef ONEFLOW_CORE_EP_TEST_TEST_UTIL_
#define ONEFLOW_CORE_EP_TEST_TEST_UTIL_

#include <gtest/gtest.h>
#include "oneflow/core/ep/include/device_manager_registry.h"

namespace oneflow {

namespace ep {

namespace test {

class TestCase : public ::testing::Test {
 protected:
  void SetUp() override {
    for (const auto& device_type : device_manager_registry_.GetRegisteredDeviceTypes()) {
      // ignore mock device
      if (device_type == DeviceType::kMockDevice) { continue; }
      if (device_manager_registry_.GetDeviceManager(device_type)->GetDeviceCount() > 0) {
        available_device_types_.insert(device_type);
      }
    }
  }
  void TearDown() override {
    // do nothing
  }
  DeviceManagerRegistry device_manager_registry_;
  std::set<DeviceType> available_device_types_;
};

class DeviceMemoryGuard {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DeviceMemoryGuard);
  DeviceMemoryGuard(Device* device, size_t size) : device_(device), options_{} {
    CHECK_JUST(device_->Alloc(options_, &ptr_, size));
  }

  ~DeviceMemoryGuard() { device_->Free(options_, ptr_); }

  template<typename T = void>
  T* ptr() {
    return reinterpret_cast<T*>(ptr_);
  }

 private:
  Device* device_;
  AllocationOptions options_;
  void* ptr_{};
};

class PinnedMemoryGuard {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PinnedMemoryGuard);
  PinnedMemoryGuard(Device* device, size_t size) : device_(device) {
    options_.SetPinnedDevice(device->device_type(), 0);
    CHECK_JUST(device_->AllocPinned(options_, &ptr_, size));
  }

  ~PinnedMemoryGuard() { device_->FreePinned(options_, ptr_); }

  template<typename T = void>
  T* ptr() {
    return reinterpret_cast<T*>(ptr_);
  }

 private:
  AllocationOptions options_;
  Device* device_;
  void* ptr_{};
};

class StreamGuard {
 public:
  OF_DISALLOW_COPY_AND_MOVE(StreamGuard);
  explicit StreamGuard(Device* device) : device_(device) {
    stream_ = device_->CreateStream();
    CHECK_JUST(stream_->OnExecutionContextSetup());
  }

  ~StreamGuard() {
    CHECK_JUST(stream_->OnExecutionContextTeardown());
    device_->DestroyStream(stream_);
  }

  Stream* stream() { return stream_; }

 private:
  Device* device_;
  Stream* stream_;
};

}  // namespace test

}  // namespace ep

}  // namespace oneflow

#endif  // ONEFLOW_CORE_EP_TEST_TEST_UTIL_
