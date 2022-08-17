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
#ifndef ONEFLOW_CORE_COMMON_FOREIGN_STACK_GETTER_H
#define ONEFLOW_CORE_COMMON_FOREIGN_STACK_GETTER_H

#include <cstdint>
#include <utility>
#include "oneflow/core/common/optional.h"
#include "oneflow/core/common/thread_local_guard.h"

namespace oneflow {

struct StackIdGuardKind {};

using StackId = int64_t;

using StackIdThreadLocalGuard = ThreadLocalGuard<StackId, StackIdGuardKind>;

// Only intended for use in main thread, because only main thread
// has meaningful stack. So this function is not designed to be thread-safe.
inline StackId GetNextStackId() {
  static StackId next_stack_id = 0;
  return next_stack_id++;
}

class ForeignStackGetter {
 public:
  virtual ~ForeignStackGetter() = default;
  virtual std::string GetCurrentStack(size_t max_size) const = 0;
  virtual void RecordCurrentStack(StackId id) = 0;
  virtual std::string GetFormatted(StackId id) const = 0;
};
}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_FOREIGN_STACK_GETTER_H
