#ifndef ONEFLOW_CORE_FRAMEWORK_PLACEMENT_RPC_UTIL_H_
#define ONEFLOW_CORE_FRAMEWORK_PLACEMENT_RPC_UTIL_H_

#include "oneflow/core/framework/rpc_token.h"
#include "oneflow/core/framework/rpc_util.h"
#include "oneflow/core/common/symbol.h"

namespace oneflow {

// Do nothing unless except rpc_token check.
class NaiveTokenCheckAsyncRpcCtx : public AsyncRpcCtx {
 public:
	NaiveTokenCheckAsyncRpcCtx() = default;
	~NaiveTokenCheckAsyncRpcCtx() override = default;

	Maybe<void> MakeDataBufferAndCallback(
		int64_t rank, void** buffer, std::size_t* size, std::function<void()>* Callback) override;
};

class ParallelDesc;

Maybe<NaiveTokenCheckAsyncRpcCtx> CheckRpcToken(Symbol<ParallelDesc> parallel_desc);

Maybe<RpcToken> GetAutoIncrementalRpcToken(Symbol<ParallelDesc> parallel_desc);

}

#endif  // ONEFLOW_CORE_FRAMEWORK_PLACEMENT_RPC_UTIL_H_
