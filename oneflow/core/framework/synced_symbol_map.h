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
#ifndef ONEFLOW_CORE_FRAMEWORK_SYNCED_SYMBOL_MAP_H_
#define ONEFLOW_CORE_FRAMEWORK_SYNCED_SYMBOL_MAP_H_

#include <unordered_map>
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/common/type_traits.h"
#include "oneflow/core/job/rank_group_scope.h"

namespace oneflow {

uint64_t GetAutoIncrementalSymbolId();

template<typename T>
struct SyncedSymbolMap final {
  template<typename SyncT>
  static Maybe<uint64_t> FindOrSync(Symbol<T> symbol, const SyncT& Sync) {
    auto* map = JUST(MutThreadLocalSymbol2SyncedSymbolId());
    const auto& iter = map->find(symbol);
    if (iter != map->end()) { return iter->second; }
    uint64_t symbol_id = GetAutoIncrementalSymbolId();
    JUST(Sync(symbol_id, symbol));
    JUST(Emplace(symbol_id, symbol));
    return symbol_id;
  }

  static Maybe<Symbol<T>> Symbol4SyncedSymbolId(uint64_t synced_symbol_id) {
    auto* map = JUST(MutThreadLocalSyncedSymbolId2Symbol());
    return JUST(MapAt(*map, synced_symbol_id));
  }

 private:
  static Maybe<void> Emplace(uint64_t synced_symbol_id, Symbol<T> symbol) {
    auto* id2symbol = JUST(MutThreadLocalSyncedSymbolId2Symbol());
    CHECK_OR_RETURN(id2symbol->emplace(synced_symbol_id, symbol).second);
    auto* symbol2id = JUST(MutThreadLocalSymbol2SyncedSymbolId());
    CHECK_OR_RETURN(symbol2id->emplace(symbol, synced_symbol_id).second);
    return Maybe<void>::Ok();
  }

  static Maybe<std::unordered_map<uint64_t, Symbol<T>>*> MutThreadLocalSyncedSymbolId2Symbol() {
    static thread_local auto* map =
        new std::unordered_map<Symbol<RankGroup>, std::unordered_map<uint64_t, Symbol<T>>>();
    const auto& rank_group = JUST(RankGroupScope::CurrentRankGroup());
    return &(*map)[rank_group];
  }

  static Maybe<std::unordered_map<Symbol<T>, uint64_t>*> MutThreadLocalSymbol2SyncedSymbolId() {
    static thread_local auto* map =
        new std::unordered_map<Symbol<RankGroup>, std::unordered_map<Symbol<T>, uint64_t>>();
    const auto& rank_group = JUST(RankGroupScope::CurrentRankGroup());
    return &(*map)[rank_group];
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_SYNCED_SYMBOL_MAP_H_
