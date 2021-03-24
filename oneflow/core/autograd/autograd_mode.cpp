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

#include "oneflow/core/autograd/autograd_mode.h"

namespace oneflow {

namespace autograd {

namespace {

bool* GetThreadLocalGradMode() {
  static thread_local bool g_grad_mode = true;
  return &g_grad_mode;
}

}  // namespace

bool GradMode::is_enabled() { return *GetThreadLocalGradMode(); }

void GradMode::set_enabled(bool enabled) { *GetThreadLocalGradMode() = enabled; }

}  // namespace autograd

}  // namespace oneflow
