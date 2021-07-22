#ifndef ONEFLOW_CORE_FRAMEWORK_SYNC_SYMBOL_CONSISTENT_TENSOR_META_H_
#define ONEFLOW_CORE_FRAMEWORK_SYNC_SYMBOL_CONSISTENT_TENSOR_META_H_

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/framework/rpc_util.h"
#include "oneflow/core/framework/rpc_token.h"

namespace oneflow {

namespace one {
class ConsistentTensorMeta;
}

Maybe<void> SyncSymbolConsistentTensorMeta(uint64_t symbol_id, Symbol<one::ConsistentTensorMeta>);

}

#endif  // ONEFLOW_CORE_FRAMEWORK_SYNC_SYMBOL_CONSISTENT_TENSOR_META_H_
