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
#ifndef ONEFLOW_CORE_FRAMEWORK_SCOPE_GUARD_H_
#define ONEFLOW_CORE_FRAMEWORK_SCOPE_GUARD_H_

#include "oneflow/core/framework/scope_util.h"

namespace oneflow {

class ScopeGuard final {
 public:
  explicit ScopeGuard(const std::shared_ptr<Scope>& scope) : scope_(scope) {
    ThreadLocalScopeStackPush(scope_).GetOrThrow();
  }
  ~ScopeGuard() {
    const auto& scope = GetCurrentScope().GetPtrOrThrow();
    CHECK(scope == scope_);
    ThreadLocalScopeStackPop().GetOrThrow();
  }

 private:
  const std::shared_ptr<Scope>& scope_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_SCOPE_GUARD_H_
