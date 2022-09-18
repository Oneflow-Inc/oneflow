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
#include "oneflow/core/common/throw.h"
#include "oneflow/core/framework/autocast.h"
#include "oneflow/core/job_rewriter/auto_mixed_precision.h"
#include "oneflow/core/job_rewriter/auto_mixed_precision_lists.h"

namespace oneflow {
namespace autocast {

namespace {

bool* autocast_enabled() {
  static thread_local bool autocast_enabled = false;
  return &autocast_enabled;
}
DeviceType* autocast_device_type() {
  static thread_local DeviceType autocast_device_type = kCUDA;
  return &autocast_device_type;
}
Symbol<DType>* autocast_dtype() {
  static thread_local Symbol<DType> autocast_dtype = DType::Float16();
  return &autocast_dtype;
}
Symbol<DType>* autocast_cpu_dtype() {
  static thread_local Symbol<DType> autocast_cpu_dtype = DType::BFloat16();
  return &autocast_cpu_dtype;
}
Symbol<DType>* autocast_gpu_dtype() {
  static thread_local Symbol<DType> autocast_gpu_dtype = DType::Float16();
  return &autocast_gpu_dtype;
}
bool* cache_enabled() {
  static thread_local bool cache_enabled = true;
  return &cache_enabled;
}

}  // namespace

bool is_enabled() { return *autocast_enabled(); }
void set_enabled(bool enabled) { *autocast_enabled() = enabled; }

DeviceType get_autocast_device_type() { return *autocast_device_type(); }
void set_autocast_device_type(DeviceType device_type) { *autocast_device_type() = device_type; }

Symbol<DType> get_autocast_dtype() { return *autocast_dtype(); }
Symbol<DType> get_autocast_cpu_dtype() { return *autocast_cpu_dtype(); }
Symbol<DType> get_autocast_gpu_dtype() { return *autocast_gpu_dtype(); }

void set_autocast_dtype(Symbol<DType> dtype) { *autocast_dtype() = dtype; }
void set_autocast_cpu_dtype(Symbol<DType> dtype) { *autocast_cpu_dtype() = dtype; }
void set_autocast_gpu_dtype(Symbol<DType> dtype) { *autocast_gpu_dtype() = dtype; }

bool is_autocast_cache_enabled() { return *cache_enabled(); }
void set_autocast_cache_enabled(bool enabled) { *cache_enabled() = enabled; }
void clear_cache() {
  // TODO(hjchen2)
}

AutoCastColor AutoCastMeta::autocast_color() const { return autocast_color_; }

void AutoCastMeta::set_autocast_color(AutoCastColor color) { autocast_color_ = color; }

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

void AutoCastMeta::set_autocast_eligible(DeviceType device_type, Symbol<DType> dtype) {
  int device_index = static_cast<int>(device_type);
  while (is_autocast_eligible_.size() <= device_index) {
    is_autocast_eligible_.resize(device_index + 1);
  }
  int dtype_index = static_cast<int>(dtype->data_type());
  while (is_autocast_eligible_[device_index].size() <= dtype_index) {
    is_autocast_eligible_[device_index].resize(dtype_index + 1);
  }
  is_autocast_eligible_[device_index][dtype_index] = true;
}

bool AutoCastMeta::is_args_autocast_eligible(int arg_index) const {
  CHECK_LT_OR_THROW(arg_index, is_args_autocast_eligible_.size());  // NOLINT
  return is_args_autocast_eligible_[arg_index];
}

const std::vector<bool>& AutoCastMeta::is_args_autocast_eligible() const {
  return is_args_autocast_eligible_;
}

void AutoCastMeta::set_arg_autocast_eligible(int arg_index) {
  CHECK_LT_OR_THROW(arg_index, is_args_autocast_eligible_.size());  // NOLINT
  is_args_autocast_eligible_[arg_index] = true;
}

std::shared_ptr<AutoCastMeta> MakeAutoCastMeta(
    const std::string& op_type_name,
    const std::vector<std::pair<std::string, int32_t>>& input_args) {
  auto autocast_meta = std::make_shared<AutoCastMeta>(input_args.size());
  if (AutoMixedPrecisionLists::WhiteList().count(op_type_name)) {
    autocast_meta->set_autocast_color(kWhite);
  } else if (AutoMixedPrecisionLists::GrayList().count(op_type_name)) {
    autocast_meta->set_autocast_color(kGray);
  } else if (AutoMixedPrecisionLists::ClearList().count(op_type_name)) {
    autocast_meta->set_autocast_color(kClear);
  } else {
    autocast_meta->set_autocast_color(kBlack);
  }
  for (int i = 0; i < input_args.size(); ++i) {
    if (!amp::IsNoCast(op_type_name, input_args[i])) {
      autocast_meta->set_arg_autocast_eligible(i);
    }
  }
  // autocast only supports the following device type(s) and low precision type(s):
  //   - device type: CUDA
  //   - low precision type: half, bfloat16
  static std::vector<DeviceType> autocast_device_types{kCUDA};
  static std::vector<Symbol<DType>> autocast_dtypes{DType::Float16(), DType::BFloat16()};

  if (autocast_meta->autocast_color() != kBlack) {
    for (auto device_type : autocast_device_types) {
      for (auto dtype : autocast_dtypes) {
        autocast_meta->set_autocast_eligible(device_type, dtype);
      }
    }
  }
  return autocast_meta;
}

}  // namespace autocast
}  // namespace oneflow
