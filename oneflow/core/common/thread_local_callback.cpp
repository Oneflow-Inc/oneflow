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
#include "oneflow/core/common/thread_local_callback.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"

namespace oneflow {

namespace blocking {

using StackInfoCallbackType = std::function<std::string()>;

StackInfoCallbackType GetDefaultStackInfoCallback() {
  return []() {
    return "[rank=" + std::to_string(GlobalProcessCtx::Rank()) + "]" + " Blocking detected.";
  };
}

StackInfoCallbackType* GetMutStackInfoCallback() {
  static thread_local StackInfoCallbackType StackInfoCallback = GetDefaultStackInfoCallback();
  return &StackInfoCallback;
}

StackInfoCallbackType GetStackInfoCallback() { return *GetMutStackInfoCallback(); }

std::string GetStackInfo() {
  return "[rank=" + std::to_string(GlobalProcessCtx::Rank()) + "]"
         + " Blocking detected. Python stack:\n" + GetStackInfoCallback()();
}

void RegisterStackInfoCallback(const StackInfoCallbackType& Callback) {
  *GetMutStackInfoCallback() = Callback;
}

void ClearStackInfoCallback() { *GetMutStackInfoCallback() = GetDefaultStackInfoCallback(); }

}  // namespace blocking

}  // namespace oneflow
