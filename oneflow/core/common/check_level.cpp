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
#include <cstdlib>
#include <type_traits>
#include "oneflow/core/common/just.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/env_var/debug_mode.h"

namespace oneflow {

bool IsEnvEnabled(int32_t check_level) {
  static const int env_check_level = ParseIntegerFromEnv("ONEFOW_CHECK_LEVEL", -1);
  static const bool env_debug_mode = IsInDebugMode();
  return env_debug_mode || env_check_level >= check_level;
}

}  // namespace oneflow
