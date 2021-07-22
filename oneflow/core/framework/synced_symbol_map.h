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
		auto* map = JUST(MutThreadLocalSyncedSymbolId2Symbol());
		const auto& iter = map->find(symbol);
		if (iter != map->end()) { return iter->second; }
		uint64_t symbol_id = GetAutoIncrementalSymbolId();
		JUST(Sync(symbol_id, symbol));
		JUST(Emplace(symbol_id, symbol));
		return symbol_id;
	}

	static Maybe<Symbol<T>> Symbol4SyncedSymbolId(uint64_t synced_symbol_id) {
		auto* map = JUST(MutThreadLocalSyncedSymbolId2Symbol());
		return JUST(MaptAt(*map, synced_symbol_id));
	}

 private:
	static Maybe<uint64_t> SyncedSymbolId4Symbol(Symbol<T> symbol) {
		auto* map = JUST(*MutThreadLocalSymbol2SyncedSymbolId());
		return JUST(MapAt(map, symbol));
	}

	static Maybe<void> Emplace(uint64_t synced_symbol_id, Symbol<T> symbol) {
		const auto* id2symbol = JUST(MutThreadLocalSyncedSymbolId2Symbol());
		JUST(id2symbol->emplace(synced_symbol_id, symbol).second);
		const auto* symbol2id = JUST(MutThreadLocalSymbol2SyncedSymbolId());
		JUST(symbol2id->emplace(symbol, synced_symbol_id).second);
		return Maybe<void>::Ok();
	}

	static Maybe<std::unordered_map<uint64_t, Symbol<T>>*> MutThreadLocalSyncedSymbolId2Symbol() {
		static thread_local auto* map = new std::unordered_map<Symbol<RankGroup>, std::unordered_map<uint64_t, Symbol<T>>>();
		const auto& rank_group = JUST(RankGroupScope::CurrentRankGroup());
		return &(*map)[rank_group]; 
	}

	static Maybe<std::unordered_map<Symbol<T>, uint64_t>*> MutThreadLocalSymbol2SyncedSymbolId() {
		static thread_local auto* map = new std::unordered_map<Symbol<RankGroup>, std::unordered_map<Symbol<T>, uint64_t>>();
		const auto& rank_group = JUST(RankGroupScope::CurrentRankGroup());
		return &(*map)[rank_group]; 
	}
};

}

#endif  // ONEFLOW_CORE_FRAMEWORK_SYNCED_SYMBOL_MAP_H_
