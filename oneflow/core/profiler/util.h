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

#ifndef ONEFLOW_CORE_PROFILER_UTIL_H_
#define ONEFLOW_CORE_PROFILER_UTIL_H_

#include <cstdint>
#include <time.h>

namespace oneflow {

namespace profiler {

using time_t = int64_t;

inline time_t GetTimeNow(bool allow_monotonic = false) {
  struct timespec t {};
  auto mode = CLOCK_REALTIME;
  if (allow_monotonic) { mode = CLOCK_MONOTONIC; }
  clock_gettime(mode, &t);
  return static_cast<time_t>(t.tv_sec) * 1000000000 + static_cast<time_t>(t.tv_nsec);
}

}  // namespace profiler
}  // namespace oneflow

#endif  // ONEFLOW_CORE_PROFILER_UTIL_H_
