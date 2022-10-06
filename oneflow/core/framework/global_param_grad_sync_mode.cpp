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

#include "oneflow/core/framework/global_param_grad_sync_mode.h"

namespace oneflow {

namespace {

bool* GetThreadLocalGradSyncMode() {
  static thread_local bool g_grad_mode = true;
  return &g_grad_mode;
}

}  // namespace

bool GlobalGradSyncMode::is_enabled() { return *GetThreadLocalGradSyncMode(); }

void GlobalGradSyncMode::set_enabled(bool enabled) { *GetThreadLocalGradSyncMode() = enabled; }

}  // namespace oneflow
