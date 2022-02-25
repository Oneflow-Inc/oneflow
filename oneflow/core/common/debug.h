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
#ifndef ONEFLOW_CORE_COMMON_DEBUG_H_
#define ONEFLOW_CORE_COMMON_DEBUG_H_

#include "oneflow/core/common/just.h"
#include "oneflow/core/common/maybe.h"

namespace oneflow {

const int OF_INFO = 0;
const int OF_WARNING = 1;
const int OF_ERROR = 2;
const int OF_FATAL = 3;

struct WithEnvCheck {
  static bool env_check(int32_t level) {
    const char* env_debug_level = std::getenv("ONEFOW_DEBUG_LEVEL");
    const char* env_debug_mode = std::getenv("ONEFLOW_DEBUG_MODE");
    return env_debug_mode != nullptr
           || (env_debug_level != nullptr && std::atoi(env_debug_level) >= level);
  }

  template<typename T, typename = void>
  struct EnvCheck;

  template<typename T, typename... Args>
  struct EnvCheck<T (*)(Args...)> final {
    template<int32_t debug_level, T (*func)(Args...)>
    static Maybe<void> Call(Args... args) {
      if (env_check(debug_level)) JUST(func(std::forward<Args>(args)...));
      return Maybe<void>::Ok();
    }
  };
};

#define DEBUG(debug_level, fn_ptr) \
  (&WithEnvCheck::EnvCheck<decltype(fn_ptr)>::Call<debug_level, fn_ptr>)

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_DEBUG_H_