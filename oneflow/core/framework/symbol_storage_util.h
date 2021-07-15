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

#include "oneflow/core/framework/symbol_id_cache.h"
#include "oneflow/core/vm/symbol_storage.h"

namespace oneflow {

template<typename SymbolConfT>
Maybe<bool> HasSymbol(const SymbolConfT& symbol_conf) {
  const auto& id_cache = *JUST(GlobalMaybe<symbol::IdCache<SymbolConfT>>());
  return id_cache.Has(symbol_conf);
}

template<typename SymbolConfT, typename SymbolT>
Maybe<SymbolT> GetSymbol(const SymbolConfT& symbol_conf) {
  const auto& id_cache = *JUST(GlobalMaybe<symbol::IdCache<SymbolConfT>>());
  const auto& symbol_storage = *Global<symbol::Storage<SymbolT>>::Get();
  int64_t symbol_id = JUST(id_cache.Get(symbol_conf));
  const auto& ptr = JUST(symbol_storage.MaybeGetPtr(symbol_id));
  JUST(ptr->symbol_id());
  return ptr;
}

// TODO(hanbibin): the second template arg will be moved after symbol_storage is refactored
template<typename SymbolConfT, typename SymbolPbT, typename SymbolT>
Maybe<void> AddSymbol(int64_t symbol_id, const SymbolConfT& symbol_conf) {
  SymbolPbT symbol_pb;
  symbol_conf.ToProto(&symbol_pb);
  JUST(Global<symbol::Storage<SymbolT>>::Get()->Add(symbol_id, symbol_pb));
  auto* id_cache = JUST(GlobalMaybe<symbol::IdCache<SymbolConfT>>());
  CHECK_OR_RETURN(!id_cache->Has(symbol_conf));
  JUST(id_cache->FindOrCreate(symbol_conf, [&symbol_id]() -> Maybe<int64_t> { return symbol_id; }));
  return Maybe<void>::Ok();
}

template<typename SymbolConfT, typename SymbolT>
Maybe<SymbolT> GetSymbol(int64_t symbol_id) {
  const auto& symbol_storage = *Global<symbol::Storage<SymbolT>>::Get();
  const auto& ptr = JUST(symbol_storage.MaybeGetPtr(symbol_id));
  JUST(ptr->symbol_id());
  return ptr;
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_SYMBOL_STORAGE_H_
