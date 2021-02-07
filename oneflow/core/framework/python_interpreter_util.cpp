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
#include <vector>
#include "oneflow/core/framework/python_interpreter_util.h"

namespace oneflow {

namespace {

Maybe<std::vector<bool>*> GetShuttingDown() {
  static std::vector<bool> shutting_down{false};
  return &shutting_down;
}

}  // namespace

Maybe<bool> IsShuttingDown() {
  auto* shutting_down = JUST(GetShuttingDown());
  CHECK_EQ_OR_RETURN(shutting_down->size(), 1);
  bool is_interpreter_shutdown = (*shutting_down)[0];
  return is_interpreter_shutdown;
}

Maybe<void> SetShuttingDown() {
  auto* shutting_down = JUST(GetShuttingDown());
  CHECK_EQ_OR_RETURN(shutting_down->size(), 1);
  (*shutting_down)[0] = true;
  return Maybe<void>::Ok();
}

}  // namespace oneflow
