#include "oneflow/core/framework/rpc_token.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/thread/consistent_unique_id.h"
#include "oneflow/core/framework/rank_group_rpc_util.h"

namespace oneflow {

/*static*/RpcToken RpcToken::NewDataRpcToken() {
	static auto* seq_id = new std::atomic<int64_t>();
	RpcToken rpc_token(kDataRpcTokenType);
	JUST(rpc_token->set_data_seq_id(++*seq_id));
	return rpc_token;
}

/*static*/Maybe<RpcToken> NewOpTensorMetaRpcToken() {
	int32_t thread_consistent_unique_id = JUST(GetThisThreadConsistentUniqueId());
	int32_t rank_group_id = JUST(GetCurrentRankGroupId());
	static const int kLimit = 128;
	CHECK_GE_OR_RETURN(rank_group_id, 0);
	CHECK_LT_OR_RETURN(rank_group_id, kLimit);	
	static thread_local std::array<std::unique_ptr<RpcToken>, kLimit> rank_group_stack;
	auto* current_rpc_token = &rank_group_stack[rank_group_id];
	if (!*current_rpc_token) {
		const auto& init = NewOpTensorMetaRpcToken(thread_consistent_unique_id, rank_group_id);
		current_rpc_token->reset(new RpcToken(init));
	}
	return ++**current_rpc_token;
}

/*static*/Maybe<RpcToken> NewCmdRpcToken(RankGroupRpcCmd cmd) {
	int32_t thread_consistent_unique_id = JUST(GetThisThreadConsistentUniqueId());
	int32_t rank_group_id = JUST(GetCurrentRankGroupId());
	static const int kLimit = 128;
	static thread_local std::array<std::array<std::unique_ptr<RpcToken>, kSizeOfRankGroupRpcCmd>, kLimit> rank_group_stack;
	CHECK_GE_OR_RETURN(rank_group_id, 0);
	CHECK_LT_OR_RETURN(rank_group_id, kLimit);	
	CHECK_GE_OR_RETURN(static_cast<int>(cmd), 0);
	CHECK_LT_OR_RETURN(static_cast<int>(cmd), kSizeOfRankGroupRpcCmd);
	auto* current_rpc_token = &rank_group_stack[rank_group_id][cmd];
	if (!*current_rpc_token) {
		const auto& init = NewCmdRpcToken(cmd, thread_consistent_unique_id, rank_group_id);
		current_rpc_token->reset(new RpcToken(init));
	}
	return ++**current_rpc_token;
}

Maybe<int64_t> RpcToken::thread_consistent_unique_id() const {
	CHECK_OR_RETURN(type() == kOpTensorMetaRpcTokenType || type() == kCmdRpcTokenType);
	return thread_consistent_unique_id_;
}

Maybe<int64_t> RpcToken::rank_group_id() const {
	CHECK_OR_RETURN(type() == kOpTensorMetaRpcTokenType || type() == kCmdRpcTokenType);
	return rank_group_id_;
}

Maybe<int64_t> RpcToken::cmd() const {
	CHECK_OR_RETURN(type() == kCmdRpcTokenType);
	return cmd_;
}

Maybe<void> RpcToken::set_src_rank(int64_t src_rank) {
	CHECK_LT_OR_RETURN(src_rank, GetMaxVal<uint16_t>());
	src_rank_ = src_rank;
	return Maybe<void>::Ok();
}

Maybe<void> RpcToken::set_dst_rank(int64_t dst_rank) {
	CHECK_LT_OR_RETURN(dst_rank, GetMaxVal<uint16_t>());
	dst_rank_ = dst_rank;
	return Maybe<void>::Ok();
}

Maybe<void> set_data_seq_id(int64_t data_seq_id) {
	CHECK_EQ_OR_RETURN(type(), kCmdRpcTokenType);
	data_seq_id = data_seq_id % (1 << 30);
	data_seq_id_ = data_seq_id;
	return Maybe<void>::Ok();
}

RpcToken::operator uint64_t() const {
  return *reinterpret_cast<const uint64_t*>(this);
}

RpcToken& RpcToken::operator++() {
	RpcTokenType rpc_token_type = type();
	if (rpc_token_type == kDataRpcTokenType) {
		++data_seq_id_;
	} else if (rpc_token_type == kOpTensorMetaRpcTokenType) {
		++meta_seq_id_;
	} else if (rpc_token_type == kCmdRpcTokenType) {
		++cmd_seq_id_;
	} else {
		UNIMPLEMENTED();
	}
  return *this;
}

/*static*/RpcToken RpcToken::NewOpTensorMetaRpcToken(int32_t thread_consistent_unique_id, int32_t rank_group_id) {
	RpcToken rpc_token(kOpTensorMetaRpcTokenType);
	rpc_token.thread_consistent_unique_id_ = thread_consistent_unique_id;
	rpc_token.rank_group_id_ = rank_group_id;
	return rpc_token;
}

/*static*/RpcToken RpcToken::NewCmdRpcToken(RankGroupRpcCmd cmd, int32_t thread_consistent_unique_id, int32_t rank_group_id) {
	RpcToken rpc_token(kCmdRpcTokenType);
	rpc_token.thread_consistent_unique_id_ = thread_consistent_unique_id;
	rpc_token.rank_group_id_ = rank_group_id;
	rpc_token.cmd_ = static_cast<uint8_t>(cmd);
	return rpc_token;
}

}
