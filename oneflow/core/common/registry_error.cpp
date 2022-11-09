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

#include "oneflow/core/common/registry_error.h"

namespace oneflow {

namespace {
std::shared_ptr<StackedError>* MutRegistryError() {
  static std::shared_ptr<StackedError> registry_error;
  return &registry_error;
}
}  // namespace

Maybe<void> CheckAndClearRegistryFlag() {
  if (!*MutRegistryError()) { return Maybe<void>::Ok(); }
  std::shared_ptr<StackedError> registry_error_old = *MutRegistryError();
  *MutRegistryError() = nullptr;
  return registry_error_old;
}

void CatchRegistryError(const std::function<Maybe<void>()>& handler) {
  const auto& maybe_error = TRY(handler());
  if (!maybe_error.IsOk()) {
    if (!*MutRegistryError()) { *MutRegistryError() = maybe_error.stacked_error(); }
  }
}

}  // namespace oneflow
