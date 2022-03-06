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
#ifndef ONEFLOW_CORE_COMMON_ENV_VAR_H_
#define ONEFLOW_CORE_COMMON_ENV_VAR_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

template<typename EnvVar>
int64_t EnvInteger() {
  return EnvVar::GetEnvInteger();
}

#define DEFINE_ENV_INTEGER(EnvVar, default_value)                         \
  struct EnvVar {                                                         \
    static int64_t GetEnvInteger() {                                      \
      return ParseIntegerFromEnv(OF_PP_STRINGIZE(EnvVar), default_value); \
    }                                                                     \
  };

DEFINE_ENV_INTEGER(ONEFLOW_TIMEOUT_SECONDS, 7200);
DEFINE_ENV_INTEGER(ONEFLOW_CHECK_TIMEOUT_SLEEP_SECONDS, EnvInteger<ONEFLOW_TIMEOUT_SECONDS>());

DEFINE_ENV_INTEGER(ONEFLOW_VM_BLOCKING_DEBUG_INSTRUCTIONS_DISPLAY_LIMIT, 100);
DEFINE_ENV_INTEGER(ONEFLOW_DELETE_OUTDATED_SHM_NAMES_INTERVAL, 1000);

template<typename EnvVar>
int64_t ConstEnvInteger() {
  return EnvVar::GetConstEnvInteger();
}

#define DEFINE_CONST_ENV_INTEGER(EnvVar, default_value)                                         \
  struct EnvVar {                                                                               \
    static int64_t GetConstEnvInteger() {                                                       \
      thread_local int64_t value = ParseIntegerFromEnv(OF_PP_STRINGIZE(EnvVar), default_value); \
      return value;                                                                             \
    }                                                                                           \
  };

template<typename EnvVar>
bool ConstEnvBoolean() {
  return EnvVar::GetConstEnvBoolean();
}

#define DEFINE_CONST_ENV_BOOLEAN(EnvVar, default_value)                                      \
  struct EnvVar {                                                                            \
    static bool GetConstEnvBoolean() {                                                       \
      thread_local bool value = ParseBooleanFromEnv(OF_PP_STRINGIZE(EnvVar), default_value); \
      return value;                                                                          \
    }                                                                                        \
  };

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_ENV_VAR_H_
