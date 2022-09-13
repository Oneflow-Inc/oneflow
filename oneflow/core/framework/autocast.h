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
#ifndef ONEFLOW_CORE_FRAMEWORK_AUTOCAST_H_
#define ONEFLOW_CORE_FRAMEWORK_AUTOCAST_H_

#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/dtype.h"

namespace oneflow {
namespace autocast {

bool is_enabled();
void set_enabled(bool enabled);

DeviceType get_autocast_device_type();
void set_autocast_device_type(DeviceType device_type);
Symbol<DType> get_autocast_dtype();
void set_autocast_dtype(Symbol<DType> dtype);

class AutoCastMeta final {
 public:
  AutoCastMeta() = default;

  bool is_autocast_eligible(DeviceType device_type, Symbol<DType> dtype) const;
  const std::vector<bool>& is_args_autocast_eligible() const;

 private:
  std::vector<std::vector<bool>> is_autocast_eligible_;
  std::vector<bool> is_args_autocast_eligible_;
};

std::shared_ptr<AutoCastMeta> MakeAutoCastMeta(const std::string& op_type_name,
                                               const std::vector<std::string>& input_names);

}  // namespace autocast
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_AUTOCAST_H_
