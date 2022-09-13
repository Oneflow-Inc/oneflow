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
#include "oneflow/core/framework/autocast.h"

namespace oneflow {
namespace autocast {

namespace {
thread_local bool autocast_enabled = false;
thread_local DeviceType autocast_device_type = kCPU;
thread_local Symbol<DType> autocast_dtype = DType::Float16();
}  // namespace

bool is_enabled() { return autocast_enabled; }
void set_enabled(bool enabled) { autocast_enabled = true; }
DeviceType get_autocast_device_type() { return autocast_device_type; }
void set_autocast_device_type(DeviceType device_type) { autocast_device_type = device_type; }
Symbol<DType> get_autocast_dtype() { return autocast_dtype; }
void set_autocast_dtype(Symbol<DType> dtype) { autocast_dtype = dtype; }

bool AutoCastMeta::is_autocast_eligible(DeviceType device_type, Symbol<DType> dtype) const {
  int device_index = static_cast<int>(device_type);
  if (is_autocast_eligible_.size() > device_index) {
    int dtype_index = static_cast<int>(dtype->data_type());
    if (is_autocast_eligible_[device_index].size() > dtype_index) {
      return is_autocast_eligible_[device_index][dtype_index];
    }
  }
  return false;
}

const std::vector<bool>& AutoCastMeta::is_args_autocast_eligible() const {
  return is_args_autocast_eligible_;
}

std::shared_ptr<AutoCastMeta> MakeAutoCastMeta(const std::string& op_type_name,
                                               const std::vector<std::string>& input_names) {
  // TODO(hjchen2)
  return std::make_shared<AutoCastMeta>();
}

}  // namespace autocast
}  // namespace oneflow
