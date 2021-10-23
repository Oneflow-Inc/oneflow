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
#ifndef ONEFLOW_CORE_COMMON_FOREIGN_LOCK_HELPER_H
#define ONEFLOW_CORE_COMMON_FOREIGN_LOCK_HELPER_H
#include <functional>
#include "oneflow/core/common/maybe.h"

namespace oneflow {
class ForeignLockHelper {
 public:
  virtual ~ForeignLockHelper() = default;
  virtual Maybe<void> WithScopedRelease(const std::function<Maybe<void>()>&) const = 0;
  virtual Maybe<void> WithScopedAcquire(const std::function<Maybe<void>()>&) const = 0;
};
}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_FOREIGN_LOCK_HELPER_H
