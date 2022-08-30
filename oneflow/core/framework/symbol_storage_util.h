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
#ifndef ONEFLOW_CORE_FRAMEWORK_SYMBOL_STORAGE_H_
#define ONEFLOW_CORE_FRAMEWORK_SYMBOL_STORAGE_H_

#include "oneflow/core/vm/symbol_storage.h"

namespace oneflow {

template<typename SymbolT>
Maybe<SymbolT> GetSymbol(int64_t symbol_id) {
  const auto& symbol_storage = *Singleton<symbol::Storage<SymbolT>>::Get();
  const auto& ptr = JUST(symbol_storage.MaybeGetPtr(symbol_id));
  JUST(ptr->symbol_id());
  return ptr;
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_SYMBOL_STORAGE_H_
