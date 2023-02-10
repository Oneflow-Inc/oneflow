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
#ifndef ONEFLOW_CORE_THREAD_THREAD_EXECUTOR_FACTORY_H_
#define ONEFLOW_CORE_THREAD_THREAD_EXECUTOR_FACTORY_H_

#include "oneflow/core/thread/thread_runtime.h"

namespace oneflow {
namespace thread {

constexpr bool IsTbbEnabled() {
#ifdef WITH_TBB
  return true;
#else
  return false;
#endif
}

constexpr bool IsOmpEnabled() {
#ifdef WITH_OMP
  return true;
#else
  return false;
#endif
}

enum class RuntimeType {
  kSeq,
  kOf,
  kTbb,
  kOmp,
};

class RuntimeFactory {
 public:
  static Maybe<thread::RuntimeBase> Create(RuntimeType type);
  static Maybe<thread::RuntimeBase> Create(const std::string& type);
};

}  // namespace thread
}  // namespace oneflow
#endif  // ONEFLOW_CORE_THREAD_THREAD_EXECUTOR_FACTORY_H_
