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
#include "oneflow/core/functional/functional.h"

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

inline Symbol<DType> get_lower_precision_fp_from_device_type(DeviceType device_type) {
  if (device_type == DeviceType::kCPU) { return get_autocast_cpu_dtype(); };
  return get_autocast_gpu_dtype();
}

// The structure below is referenced from PyTorch:
// https://github.com/pytorch/pytorch/blob/41d79695907cd4105b8e7167cf8a57ba48e1f079/aten/src/ATen/autocast_mode.cpp#L60-L63
// The weakref keeps the source's TensorImpl from being deleted.  We need to because we're
// using the source TensorImpl* as the key.  If it were deleted, another random Tensor could
// be allocated whose TensorImpl* happened to have the same value.  This TensorImpl* would
// then mistakenly hit in cache: a rare, intermittent, unpredictable bug.
using val_type = std::pair<std::weak_ptr<one::Tensor>, std::shared_ptr<one::Tensor>>;
using key_type = std::pair<const one::EagerLocalTensorImpl*, DataType>;
using cached_map = std::unordered_map<key_type, val_type>;

std::unordered_map<key_type, val_type>* cached_casts() {
  static thread_local std::unordered_map<key_type, val_type> cached_casts;
  return &cached_casts;
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

Maybe<one::Tensor> cached_cast(const std::shared_ptr<one::Tensor>& tensor, Symbol<DType> cast_type,
                               DeviceType device_type) {
  bool use_cache = (is_autocast_cache_enabled() && tensor->requires_grad()
                    && cast_type == get_lower_precision_fp_from_device_type(device_type)
                    && tensor->dtype()->data_type() == DataType::kFloat && tensor->is_leaf()
                    && !tensor->is_view());
  if (use_cache) {
    auto it = cached_casts()->find(
        std::make_pair(JUST(tensor->mut_eager_local_tensor_impl()), cast_type->data_type()));
    if (it == cached_casts()->end() || it->second.first.lock() == nullptr) {
      const std::shared_ptr<one::Tensor>& result =
          JUST(one::functional::To(tensor, cast_type, /*copy*/ false));
      if (it == cached_casts()->end()) {
        cached_casts()->emplace(
            std::make_pair(JUST(tensor->mut_eager_local_tensor_impl()), cast_type->data_type()),
            std::make_pair(tensor->weak_from_this(), result));
      } else {
        it->second.first = tensor->weak_from_this();
        it->second.second = result;
      }
      return result;
    } else {
      return it->second.second;
    }
  } else {
    return one::functional::To(tensor, cast_type, /*copy*/ false);
  }
};

void clear_cache() { cached_casts()->clear(); }

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
