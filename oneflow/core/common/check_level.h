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
#ifndef ONEFLOW_CORE_CHECK_LEVEL_H_
#define ONEFLOW_CORE_CHECK_LEVEL_H_

#include <type_traits>
#include "oneflow/core/common/just.h"
#include "oneflow/core/common/maybe.h"

namespace oneflow {

struct WithCheckLevel {
  static bool is_enabled_check(int32_t check_level) {
    const char* env_check_level = std::getenv("ONEFOW_CHECK_LEVEL");
    const char* env_debug_mode = std::getenv("ONEFLOW_DEBUG_MODE");
    return env_debug_mode != nullptr
           || (env_check_level != nullptr && std::atoi(env_check_level) >= check_level);
  }

  template<typename T, typename = void>
  struct Check;

  template<typename T, typename... Args>
  struct Check<T (*)(Args...)> final {
    template<int32_t check_level, T (*func)(Args...)>
    static T Call(Args... args) {
      static_assert(std::is_same<T, Maybe<void>>::value,
                    "returned value type must be Maybe<void>.");
      if (is_enabled_check(check_level)) JUST(func(std::forward<Args>(args)...));
      return Maybe<void>::Ok();
    }
  };
};

#define CHECK_LEVEL(check_level, fn_ptr) \
  (&WithCheckLevel::Check<decltype(fn_ptr)>::Call<check_level, fn_ptr>)

}  // namespace oneflow

#endif  // ONEFLOW_CORE_CHECK_LEVEL_H_
