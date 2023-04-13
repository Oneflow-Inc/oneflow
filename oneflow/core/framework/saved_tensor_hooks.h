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
#ifndef ONEFLOW_CORE_FRAMEWORK_SAVED_TENSOR_HOOKS_H_
#define ONEFLOW_CORE_FRAMEWORK_SAVED_TENSOR_HOOKS_H_

#include "oneflow/core/framework/tensor.h"

namespace oneflow {
namespace one {
class SavedTensorHook {
 public:
  virtual ~SavedTensorHook() = default;
  virtual void pack(const std::shared_ptr<Tensor>& tensor) = 0;
  virtual std::shared_ptr<Tensor> unpack() = 0;
};

class SavedTensorHookCreator {
 public:
  virtual ~SavedTensorHookCreator() = default;
  virtual std::unique_ptr<SavedTensorHook> new_saved_tensor_hook() const = 0;
};

}  // namespace one
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_SAVED_TENSOR_HOOKS_H_
