#ifndef ONEFLOW_CORE_FRAMEWORK_SYNC_SYMBOL_PARALLEL_DESC_H_
#define ONEFLOW_CORE_FRAMEWORK_SYNC_SYMBOL_PARALLEL_DESC_H_

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/framework/rpc_util.h"
#include "oneflow/core/framework/rpc_token.h"

namespace oneflow {

class ParallelDesc;

Maybe<void> SyncSymbolParallelDesc(uint64_t symbol_id, Symbol<ParallelDesc>);

}

#endif  // ONEFLOW_CORE_FRAMEWORK_SYNC_SYMBOL_PARALLEL_DESC_H_
