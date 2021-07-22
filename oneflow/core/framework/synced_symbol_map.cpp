#include "oneflow/core/framework/synced_symbol_map.h"

namespace oneflow {

uint64_t GetAutoIncrementalSymbolId() {
	static thread_local uint64_t id = 4096;
	return id++;
}

}
