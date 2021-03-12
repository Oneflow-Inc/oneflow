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
#ifndef ONEFLOW_CORE_FRAMEWORK_TENSOR_NAME_SCOPE_H_
#define ONEFLOW_CORE_FRAMEWORK_TENSOR_NAME_SCOPE_H_

#include <string>

#include "oneflow/core/framework/tensor.h"

namespace oneflow {
namespace one {

class TensorNameScope {
 public:
  static TensorNameScope* Global();

  const std::string& Lookup(const std::shared_ptr<Tensor>& tensor) const;

  void Record(const std::shared_ptr<Tensor>& tensor, const std::string& name);

 private:
  TensorNameScope() = default;
  virtual ~TensorNameScope() = default;

 private:
  mutable std::mutex mutex_;

  std::string default_tensor_name_ = "";
  // uint64_t(Tensor*) -> the name of the tensor.
  std::unordered_map<uint64_t, std::string> tensor_names_;
};

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_TENSOR_NAME_SCOPE_H_
