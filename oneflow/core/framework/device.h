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

#include <fmt/core.h>
#include <fmt/ostream.h>
#include <memory>
#include <string>
#include <unordered_set>
#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/common/optional.h"

namespace oneflow {

class ParallelDesc;
class MemoryCase;

inline size_t GetInstructionHighWaterMark() { return 40000; }
inline size_t GetInstructionLowWaterMark() { return 20000; }

class Device final {
 public:
  Device(const Device&) = default;
  Device(Device&&) = default;
  ~Device() = default;
  Device& operator=(const Device&) = delete;
  const std::string& type() const { return type_; }
  DeviceType enum_type() const { return enum_type_; }
  int64_t device_id() const { return device_id_; }
  std::string ToString() const;
  std::string ToRepr() const;
  size_t hash_value() const { return hash_value_; }
  bool operator==(const Device& device) const {
    return type_ == device.type() && device_id_ == device.device_id();
  }
  bool operator!=(const Device& device) const { return !operator==(device); }
  const std::shared_ptr<MemoryCase>& mem_case() const { return mem_case_; }

  static Maybe<Symbol<Device>> New(const std::string& type, int64_t device_id);
  static Maybe<Symbol<Device>> New(const std::string& type);
  static Maybe<Symbol<Device>> ParseAndNew(const std::string& type_or_type_with_device_id);

  static Maybe<Symbol<Device>> MakeDeviceByParallelDesc(const ParallelDesc& parallel_desc);

  static Maybe<Symbol<ParallelDesc>> (*GetPlacement)(const Device& device);

 private:
  Device(const std::string& type, int64_t device_id);
  Maybe<void> Init();

  const std::string type_;
  DeviceType enum_type_;
  const int64_t device_id_;
  const size_t hash_value_;
  std::shared_ptr<MemoryCase> mem_case_;
};

std::ostream& operator<<(std::ostream& os, Symbol<Device> device);

extern Maybe<Symbol<ParallelDesc>> (*Placement4Device)(Symbol<Device> device);

Maybe<std::pair<std::string, int>> ParseDeviceString(const std::string& device_str);

}  // namespace oneflow

template<>
struct fmt::formatter<oneflow::Symbol<oneflow::Device>> : ostream_formatter {};

namespace std {
template<>
struct hash<oneflow::Device> final {
  size_t operator()(const oneflow::Device& device) const { return device.hash_value(); }
};
}  // namespace std

#endif  // ONEFLOW_CORE_FRAMEWORK_DEVICE_H_
