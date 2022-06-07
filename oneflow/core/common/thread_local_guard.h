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
#ifndef ONEFLOW_CORE_COMMON_THREAD_LOCAL_GUARD_H_
#define ONEFLOW_CORE_COMMON_THREAD_LOCAL_GUARD_H_

#include <memory>
#include <glog/logging.h>

namespace oneflow {

// Interfaces:
//   - ThreadLocalGuard::CurrentValue()
//   - ThreadLocalGuard::HasCurrentValue()
template<typename T>
class ThreadLocalGuard;

template<>
class ThreadLocalGuard<bool> {
 public:
  explicit ThreadLocalGuard(bool value) {
    old_value_ = *MutThreadLocalValue();
    *MutThreadLocalValue() = int(value);
  }
  ~ThreadLocalGuard() { *MutThreadLocalValue() = old_value_; }

  static bool CurrentValue() {
    int value = *MutThreadLocalValue();
    CHECK_GE(value, 0);
    return value > 0;
  }

  static bool HasCurrentValue() { return *MutThreadLocalValue() >= 0; }

 private:
  static int* MutThreadLocalValue() {
    static thread_local int value = -1;
    return &value;
  }

  // -1: not exists.
  // 0: false.
  // 1: true.
  int old_value_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_THREAD_LOCAL_GUARD_H_
