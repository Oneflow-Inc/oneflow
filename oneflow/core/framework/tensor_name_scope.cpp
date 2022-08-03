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
#include "oneflow/core/framework/tensor_name_scope.h"
#include <cstdint>

namespace oneflow {
namespace one {

/* static */ TensorNameScope* TensorNameScope::Global() {
  static TensorNameScope scope;
  return &scope;
}

const std::string& TensorNameScope::Lookup(const Tensor* tensor) const {
  uint64_t key = reinterpret_cast<uint64_t>(tensor);
  const auto* tensor_names = [&]() {
    if (tensor->is_lazy()) { return &lazy_tensor_names_; }
    return &eager_tensor_names_;
  }();
  std::lock_guard<std::mutex> lock(mutex_);
  const auto& it = tensor_names->find(key);
  if (it != tensor_names->end()) {
    return it->second;
  } else {
    return default_tensor_name_;
  }
}

const std::string& TensorNameScope::Lookup(const std::shared_ptr<Tensor>& tensor) const {
  return Lookup(tensor.get());
}

void TensorNameScope::Record(const Tensor* tensor, const std::string& name) {
  uint64_t key = reinterpret_cast<uint64_t>(tensor);
  auto* tensor_names = [&]() {
    if (tensor->is_lazy()) { return &lazy_tensor_names_; }
    return &eager_tensor_names_;
  }();
  std::lock_guard<std::mutex> lock(mutex_);
  // We assume that the name of the tensor will be update more than once.
  (*tensor_names)[key] = name;
}

void TensorNameScope::Record(const std::shared_ptr<Tensor>& tensor, const std::string& name) {
  Record(tensor.get(), name);
}

void TensorNameScope::Clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  lazy_tensor_names_.clear();
  eager_tensor_names_.clear();
}

}  // namespace one
}  // namespace oneflow
