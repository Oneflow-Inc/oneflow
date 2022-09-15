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
#include "oneflow/core/framework/op_expr_grad_function.h"

#include "oneflow/core/framework/saved_tensor_hooks.h"

namespace oneflow {
namespace one {

void AutoGradCaptureState::unpack() {
  if (saved_tensors_.empty() && !hooks_.empty()) {
    for (const auto& hook : hooks_) { saved_tensors_.push_back(hook->unpack()); }
    hooks_.clear();
  }
}

size_t AutoGradCaptureState::SaveTensorForBackward(const std::shared_ptr<Tensor>& tensor) {
  auto hook = []() -> std::unique_ptr<SavedTensorHook> {
    if (auto* hook_creator = Singleton<SavedTensorHookCreator>::Get()) {
      return hook_creator->new_saved_tensor_hook();
    }
    return nullptr;
  }();
  if (hook) {
    hook->pack(tensor);
    size_t offset = hooks_.size();
    hooks_.push_back(std::move(hook));
    return offset;
  } else {
    size_t offset = saved_tensors_.size();
    saved_tensors_.emplace_back(tensor);
    return offset;
  }
}

}  // namespace one
}  // namespace oneflow
