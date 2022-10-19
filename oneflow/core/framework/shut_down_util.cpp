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
#include "oneflow/core/framework/shut_down_util.h"

namespace oneflow {

namespace {

std::atomic<bool>* GetShuttingDown() {
  static std::atomic<bool> shutting_down{false};
  return &shutting_down;
}

}  // namespace

bool IsShuttingDown() {
  auto* shutting_down = GetShuttingDown();
  bool is_interpreter_shutdown = *shutting_down;
  return is_interpreter_shutdown;
}

void SetShuttingDown(bool arg_shutting_down) {
  auto* shutting_down = GetShuttingDown();
  *shutting_down = arg_shutting_down;
}

}  // namespace oneflow
