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
#ifndef ONEFLOW_USER_SUMMARY_ENV_TIME_H_
#define ONEFLOW_USER_SUMMARY_ENV_TIME_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

namespace summary {

static constexpr uint64_t kMicroTimeToNanoTime = 1000ULL;
static constexpr uint64_t kSecondToNanoTime = 1000ULL * 1000ULL * 1000ULL;
static constexpr uint64_t kMircoTimeToSecondTime = 1000ULL * 1000ULL;

inline uint64_t CurrentNanoTime() {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return (static_cast<uint64_t>(ts.tv_sec) * kSecondToNanoTime + static_cast<uint64_t>(ts.tv_nsec));
}

inline uint64_t CurrentMircoTime() { return CurrentNanoTime() / kMicroTimeToNanoTime; }

inline uint64_t CurrentSecondTime() { return CurrentMircoTime() / kMircoTimeToSecondTime; }

inline double GetWallTime() {
  return static_cast<double>(CurrentNanoTime() / kMicroTimeToNanoTime) / 1.0e6;
}

}  // namespace summary

}  // namespace oneflow

#endif  // ONEFLOW_USER_SUMMARY_ENV_TIME_H_
