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

#include <memory>
#include <string>
#include <unordered_set>
#include "oneflow/core/common/maybe.h"

namespace oneflow {

class ParallelDesc;
class MemoryCase;
class VmLocalDepObject;

class Device final : public std::enable_shared_from_this<Device> {
 public:
  Device(const Device&) = default;
  Device(Device&&) = default;
  ~Device() = default;
  Device& operator=(const Device&) = default;
  const std::string& type() const { return type_; }
  Maybe<const std::string&> of_type() const;
  int64_t device_id() const { return device_id_; }
  std::string ToString() const;
  size_t hash_value() const { return hash_value_; }
  bool operator==(const Device& device) const {
    return type_ == device.type() && device_id_ == device.device_id();
  }
  const std::shared_ptr<const ParallelDesc>& parallel_desc_ptr() const;
  const std::shared_ptr<MemoryCase>& mem_case() const { return mem_case_; }

  static Maybe<const Device> New(const std::string& type, int64_t device_id);
  static Maybe<const Device> New(const std::string& typed);

  static Maybe<const Device> MakeDeviceByParallelDesc(const ParallelDesc& parallel_desc);
  static const std::unordered_set<std::string> type_supported;

  Maybe<const std::string&> local_call_instruction_name() const;
  VmLocalDepObject* mut_compute_local_dep_object() const { return compute_local_dep_object_.get(); }

 private:
  Device(const std::string& type, int64_t device_id);
  Maybe<void> Init();

  const std::string type_;
  const int64_t device_id_;
  const size_t hash_value_;
  std::shared_ptr<MemoryCase> mem_case_;
  std::shared_ptr<VmLocalDepObject> compute_local_dep_object_;
};

}  // namespace oneflow

namespace std {
template<>
struct hash<oneflow::Device> final {
  size_t operator()(const oneflow::Device& device) const { return device.hash_value(); }
};
}  // namespace std

#endif  // ONEFLOW_CORE_FRAMEWORK_DEVICE_H_
