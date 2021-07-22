#include "oneflow/core/framework/sync_symbol_consistent_tensor_meta.h"
#include "oneflow/core/framework/sync_symbol_parallel_desc.h"
#include "oneflow/core/framework/sync_symbol_parallel_distribution.h"
#include "oneflow/core/framework/rank_group_rpc_util.h"
#include "oneflow/core/framework/tensor_meta.h"
#include "oneflow/core/framework/synced_symbol_map.h"
#include "oneflow/core/common/flat_shape.h"

namespace oneflow {

struct FlatConsistentTensorMeta final {

  static Maybe<FlatConsistentTensorMeta> New(uint64_t symbol_id, Symbol<one::ConsistentTensorMeta> consistent_tensor_meta) {
		const auto& meta = std::make_shared<FlatConsistentTensorMeta>();
		JUST(meta->Init(symbol_id, consistent_tensor_meta));
		return meta;
	}

	Maybe<void> Init(uint64_t symbol_id, Symbol<one::ConsistentTensorMeta> consistent_tensor_meta) {
		JUST(this->shape.Init(consistent_tensor_meta->shape_ptr()));
		this->dtype = static_cast<int32_t>(consistent_tensor_meta->dtype());
		this->is_dynamic = consistent_tensor_meta->is_dynamic();
		this->parallel_distribution =
			JUST(SyncedSymbolMap<cfg::ParallelDistribution>::FindOrSync(
				consistent_tensor_meta->parallel_distribution(), &SyncSymbolParallelDistribution));
	  this->parallel_desc =
		  JUST(SyncedSymbolMap<ParallelDesc>::FindOrSync(
				consistent_tensor_meta->parallel_desc(), &SyncSymbolParallelDesc));	
		return Maybe<void>::Ok();
	}

	Maybe<void> Check(uint64_t symbol_id, Symbol<one::ConsistentTensorMeta> consistent_tensor_meta) {
		JUST(this->shape.Check(consistent_tensor_meta->shape_ptr()));
		CHECK_EQ_OR_RETURN(static_cast<DataType>(this->dtype), consistent_tensor_meta->dtype());
		CHECK_EQ_OR_RETURN(static_cast<bool>(this->is_dynamic, consistent_tensor_meta->is_dynamic()));
		const auto& parallel_distribution =
			JUST(SyncedSymbolMap<cfg::ParallelDistribution>::Symbol4SyncedSymbolId(
				this->parallel_distribution));
		CHECK_OR_RETURN(parallel_distribution, consistent_tensor_meta->parallel_distribution());
		const auto& parallel_desc = 
			JUST(SyncedSymbolMap<ParallelDesc>::Symbol4SyncedSymbolId(this->parallel_desc));
		CHECK_OR_RETURN(parallel_desc, consistent_tensor_meta->parallel_desc());
		return Maybe<void>::Ok();
	}

  FlatShape shape;
  int32_t dtype;
  bool is_dynamic;
  uint64_t parallel_distribution;
  uint64_t parallel_desc;
};

Maybe<void> SyncSymbolConsistentTensorMeta(uint64_t symbol_id, Symbol<one::ConsistentTensorMeta> consistent_tensor_meta) {
	const auto& send_buffer = JUST(FlatConsistentTensorMeta::New(symbol_id, consistent_tensor_meta));
	NaiveAsyncRpcCtx send_ctx(
		[](void** buffer, std::size_t* size, std::function<void()>* Cb)->Maybe<void>{
			*buffer = send_buffer.get();
			*size = sizeof(FlatConsistentTensorMeta);
			*Cb = [send_buffer]{};
			return Maybe<void>::Ok();
		});
	const auto& recv_buffer = std::make_shared<FlatConsistentTensorMeta>();
	NaiveAsyncRpcCtx recv_ctx(
		[recv_buffer](void** buffer, std::size_t* size, std::function<void()>* Cb)->Maybe<void>{
			*buffer = recv_buffer.get();
			*size = sizeof(FlatConsistentTensorMeta);
			*Cb = [recv_buffer]{};
			return Maybe<void>::Ok();
		});
	const auto& rank_group = JUST(RankGroupScope::CurrentRankGroup());
	const auto& rpc_token =
			JUST(RpcToken::NewCmdRpcToken(kRankGroupRpcCmdSyncSymbolConsistentTensorMeta));
	JUST(RpcUtil::SendToNextRankInRing(rank_group, rpc_token, &send_ctx));
	JUST(RpcUtil::ReceiveFromPrevRankInRing(rank_group, rpc_token, &recv_ctx));
	JUST(RpcUtil::WaitUntilDoneOrTimeout(send_ctx, RpcUtil::TimeoutSeconds()));
	JUST(RpcUtil::WaitUntilDoneOrTimeout(recv_ctx, RpcUtil::TimeoutSeconds()));
	JUST(recv_buffer->Check(symbol_id, consistent_tensor_meta));
	return Maybe<void>::Ok();
}

}
