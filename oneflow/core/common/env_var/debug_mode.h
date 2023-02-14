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
#ifndef ONEFLOW_CORE_COMMON_ENV_VAR_DEBUG_MODE_H_
#define ONEFLOW_CORE_COMMON_ENV_VAR_DEBUG_MODE_H_

#include "oneflow/core/common/env_var/env_var.h"

namespace oneflow {

DEFINE_ENV_BOOL(ONEFLOW_DEBUG_MODE, false);
DEFINE_ENV_BOOL(ONEFLOW_DEBUG, false);

inline bool IsInDebugMode() { return EnvBool<ONEFLOW_DEBUG_MODE>() || EnvBool<ONEFLOW_DEBUG>(); }

DEFINE_ENV_BOOL(ENABLE_LOGICAL_CHAIN, false);
inline bool EnableLogicalChain() { return EnvBool<ENABLE_LOGICAL_CHAIN>(); }

inline bool IsPythonStackGetterEnabledByDebugBuild() {
  if (std::getenv("ONEFLOW_DEBUG_MODE") == nullptr && std::getenv("ONEFLOW_DEBUG") == nullptr
      && std::getenv("ONEFLOW_PYTHON_STACK_GETTER") == nullptr) {
    return std::string(OF_PP_STRINGIZE(ONEFLOW_CMAKE_BUILD_TYPE)) == "Debug";
  }
  return false;
}

inline bool IsPythonStackGetterEnabled() {
  if (IsPythonStackGetterEnabledByDebugBuild()) { return true; }
  return ParseBooleanFromEnv("ONEFLOW_PYTHON_STACK_GETTER", IsInDebugMode());
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_ENV_VAR_DEBUG_MODE_H_
