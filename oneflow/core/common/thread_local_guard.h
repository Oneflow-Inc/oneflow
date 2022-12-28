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
#include "oneflow/core/common/optional.h"

namespace oneflow {

template<typename T>
class ThreadLocalGuard {
 public:
  ThreadLocalGuard() {
    old_value_ = *MutThreadLocalValue();
    *MutThreadLocalValue() = Optional<T>();
  }
  explicit ThreadLocalGuard(const T& value) {
    old_value_ = *MutThreadLocalValue();
    *MutThreadLocalValue() = Optional<T>(value);
  }
  ~ThreadLocalGuard() { *MutThreadLocalValue() = old_value_; }

  static const Optional<T>& Current() { return *MutThreadLocalValue(); }

 private:
  static Optional<T>* MutThreadLocalValue() {
    static thread_local Optional<T> value{};
    return &value;
  }

  Optional<T> old_value_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_THREAD_LOCAL_GUARD_H_
