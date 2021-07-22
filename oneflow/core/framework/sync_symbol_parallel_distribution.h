#ifndef ONEFLOW_CORE_FRAMEWORK_SYNC_SYMBOL_PARALLEL_DISTRIBUTION_H_
#define ONEFLOW_CORE_FRAMEWORK_SYNC_SYMBOL_PARALLEL_DISTRIBUTION_H_

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/framework/rpc_util.h"
#include "oneflow/core/framework/rpc_token.h"

namespace oneflow {

namespace cfg {

class ParallelDistribution;

}

Maybe<void> SyncSymbolParallelDistribution(uint64_t symbol_id, Symbol<cfg::ParallelDistribution>);

}

#endif  // ONEFLOW_CORE_FRAMEWORK_SYNC_SYMBOL_PARALLEL_DISTRIBUTION_H_
