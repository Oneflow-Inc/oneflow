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
// Inspired by https://github.com/NVIDIA/DALI/blob/main/include/dali/core/static_switch.h
// and https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Dispatch.h

// modified from static_switch.h
// because MSVC cannot handle std::conditional with constexpr variable

#pragma once

/// @param COND       - a boolean expression to switch by
/// @param ...       - code to execute for true and false
///
/// Usage:
/// ```
/// FP16_SWITCH(flag, [&] {
///     some_function(...);
/// });
/// ```
#define FP16_SWITCH(COND, ...)         \
  [&] {                                \
    if (COND) {                        \
      using elem_type = __nv_bfloat16; \
      return __VA_ARGS__();            \
    } else {                           \
      using elem_type = __half;        \
      return __VA_ARGS__();            \
    }                                  \
  }()