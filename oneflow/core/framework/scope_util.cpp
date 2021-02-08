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
#include <mutex>
#include "oneflow/core/framework/scope_util.h"

namespace oneflow {

std::mutex GLOBAL_SCOPE_STACK_MUTEX;
namespace {

std::vector<std::shared_ptr<Scope>>* GlobalScopeStack() {
  static std::vector<std::shared_ptr<Scope>> scope_stack;
  return &scope_stack;
}

}  // namespace

Maybe<Scope> GetCurrentScope() {
  std::unique_lock<std::mutex> lock(GLOBAL_SCOPE_STACK_MUTEX);
  auto* scope_stack = GlobalScopeStack();
  CHECK_GT_OR_RETURN(scope_stack->size(), 0);
  return scope_stack->back();
}

Maybe<void> InitGlobalScopeStack(const std::shared_ptr<Scope>& scope) {
  std::unique_lock<std::mutex> lock(GLOBAL_SCOPE_STACK_MUTEX);
  auto* scope_stack = GlobalScopeStack();
  scope_stack->clear();
  scope_stack->emplace_back(scope);
  return Maybe<void>::Ok();
}

Maybe<void> GlobalScopeStackPush(const std::shared_ptr<Scope>& scope) {
  std::unique_lock<std::mutex> lock(GLOBAL_SCOPE_STACK_MUTEX);
  auto* scope_stack = GlobalScopeStack();
  scope_stack->emplace_back(scope);
  return Maybe<void>::Ok();
}

Maybe<void> GlobalScopeStackPop() {
  std::unique_lock<std::mutex> lock(GLOBAL_SCOPE_STACK_MUTEX);
  auto* scope_stack = GlobalScopeStack();
  scope_stack->pop_back();
  return Maybe<void>::Ok();
}

}  // namespace oneflow
