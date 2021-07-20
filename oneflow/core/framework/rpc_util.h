#ifndef ONEFLOW_CORE_FRAMEWORK_RPC_UTIL_H_
#define ONEFLOW_CORE_FRAMEWORK_RPC_UTIL_H_

#include <atomic>
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/framework/rpc_token.h"

namespace oneflow {

class AsyncRpcCtx {
 public:
	AsyncRpcCtx() : flying_cnt_(new std::atomic<int64_t>(0)) {}
	virtual ~AsyncRpcCtx() = default;

	std::shared_ptr<std::atomic<int64_t>> flying_cnt() const { return flying_cnt_; }

	virtual Maybe<void> MakeDataBufferAndCallback(
		int64_t rank, void** buffer, std::size_t* size, std::function<void()>* Callback) = 0;

 private:
	std::shared_ptr<std::atomic<int64_t>> flying_cnt_;
};

class SortedRankRanges;

struct RpcUtil final {

	static Maybe<uint32_t> GetRpcTokenCmdMajor(RpcTokenCmdLocalMajor cmd_local_major);

	static Maybe<void> WaitUntilDoneOrTimeout(const AsyncRpcCtx& ctx, int64_t seconds);

	static Maybe<void> BroadcastToAllOtherRanks(
			Symbol<SortedRankRanges> rank_ranges, const RpcToken& token, AsyncRpcCtx* ctx);

	static Maybe<void> CollectFromAllOtherRanks(
			Symbol<SortedRankRanges> rank_ranges, const RpcToken& token, AsyncRpcCtx* ctx);

	static Maybe<void> SendToNextRankInRing(
			Symbol<SortedRankRanges> rank_ranges, const RpcToken& token, AsyncRpcCtx* ctx);

	static Maybe<void> ReceiveFromPrevRankInRing(
			Symbol<SortedRankRanges> rank_ranges, const RpcToken& token, AsyncRpcCtx* ctx);

};

}

#endif  // ONEFLOW_CORE_FRAMEWORK_RPC_UTIL_H_
