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
Symbol<DType> get_autocast_cpu_dtype();
Symbol<DType> get_autocast_gpu_dtype();

void set_autocast_dtype(Symbol<DType> dtype);
void set_autocast_cpu_dtype(Symbol<DType> dtype);
void set_autocast_gpu_dtype(Symbol<DType> dtype);

bool is_autocast_cache_enabled();
void set_autocast_cache_enabled(bool enabled);
void clear_cache();

enum AutoCastColor { kNoColor, kWhite, kGray, kClear, kBlack };

class AutoCastMeta final {
 public:
  AutoCastMeta() : AutoCastMeta(0) {}
  explicit AutoCastMeta(int args_num)
      : autocast_color_(kNoColor), is_args_autocast_eligible_(args_num, false) {}

  AutoCastColor autocast_color() const;

  bool is_autocast_eligible(DeviceType device_type, Symbol<DType> dtype) const;

  bool is_args_autocast_eligible(int arg_index) const;
  const std::vector<bool>& is_args_autocast_eligible() const;

  void set_autocast_color(AutoCastColor color);
  void set_autocast_eligible(DeviceType device_type, Symbol<DType> dtype);
  void set_arg_autocast_eligible(int arg_index);

 private:
  AutoCastColor autocast_color_;
  std::vector<std::vector<bool>> is_autocast_eligible_;
  std::vector<bool> is_args_autocast_eligible_;
};

std::shared_ptr<AutoCastMeta> MakeAutoCastMeta(
    const std::string& op_type_name,
    const std::vector<std::pair<std::string, int32_t>>& input_args);

}  // namespace autocast
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_AUTOCAST_H_
