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
#ifndef ONEFLOW_CORE_COMMON_ENV_VAR_ENV_VAR_H_
#define ONEFLOW_CORE_COMMON_ENV_VAR_ENV_VAR_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

template<typename env_var>
bool EnvBool();

#define DEFINE_ENV_BOOL(env_var, default_value)                          \
  struct env_var {};                                                     \
  template<>                                                             \
  inline bool EnvBool<env_var>() {                                       \
    return ParseBooleanFromEnv(OF_PP_STRINGIZE(env_var), default_value); \
  }

template<typename env_var>
int64_t EnvInteger();

#define DEFINE_ENV_INTEGER(env_var, default_value)                       \
  struct env_var {};                                                     \
  template<>                                                             \
  inline int64_t EnvInteger<env_var>() {                                 \
    return ParseIntegerFromEnv(OF_PP_STRINGIZE(env_var), default_value); \
  }

DEFINE_ENV_INTEGER(ONEFLOW_TIMEOUT_SECONDS, 7200);
DEFINE_ENV_INTEGER(ONEFLOW_CHECK_TIMEOUT_SLEEP_SECONDS, EnvInteger<ONEFLOW_TIMEOUT_SECONDS>());

DEFINE_ENV_INTEGER(ONEFLOW_VM_BLOCKING_DEBUG_INSTRUCTIONS_DISPLAY_LIMIT, 100);
DEFINE_ENV_INTEGER(ONEFLOW_DELETE_OUTDATED_SHM_NAMES_INTERVAL, 1000);

template<typename env_var>
bool ThreadLocalEnvBool();

#define DEFINE_THREAD_LOCAL_ENV_BOOL(env_var, default_value)                                \
  struct env_var {};                                                                        \
  template<>                                                                                \
  inline bool ThreadLocalEnvBool<env_var>() {                                               \
    thread_local bool value = ParseBooleanFromEnv(OF_PP_STRINGIZE(env_var), default_value); \
    return value;                                                                           \
  }

template<typename env_var>
int64_t ThreadLocalEnvInteger();

#define DEFINE_THREAD_LOCAL_ENV_INTEGER(env_var, default_value)                                \
  struct env_var {};                                                                           \
  template<>                                                                                   \
  inline int64_t ThreadLocalEnvInteger<env_var>() {                                            \
    thread_local int64_t value = ParseIntegerFromEnv(OF_PP_STRINGIZE(env_var), default_value); \
    return value;                                                                              \
  }

DEFINE_THREAD_LOCAL_ENV_INTEGER(ONEFLOW_THRAED_LOCAL_CACHED_SIZE, 128 * 1024);

template<typename env_var>
const std::string& ThreadLocalEnvString();

#define DEFINE_THREAD_LOCAL_ENV_STRING(env_var, default_value)                                  \
  struct env_var {};                                                                            \
  template<>                                                                                    \
  inline const std::string& ThreadLocalEnvString<env_var>() {                                   \
    thread_local std::string value = GetStringFromEnv(OF_PP_STRINGIZE(env_var), default_value); \
    return value;                                                                               \
  }

DEFINE_THREAD_LOCAL_ENV_BOOL(ONEFLOW_ENABLE_LAZY_SEPARATE_COMPILE, false);
// Default compilation mode during graph compilation. There 2 modes to choose:
// "naive", master rank compile the full plan.
// "rank_per_process", multi process(rank) run seperation compile.
DEFINE_THREAD_LOCAL_ENV_STRING(ONEFLOW_LAZY_COMPILE_MODE, "naive");
// Default number of threads during graph compilation.
DEFINE_THREAD_LOCAL_ENV_INTEGER(ONEFLOW_LAZY_COMPILE_RPC_THREAD_NUM, 16);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_ENV_VAR_ENV_VAR_H_
