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
#include "oneflow/core/framework/tensor_rpc_util.h"
#include "oneflow/core/framework/sync_symbol_consistent_tensor_meta.h"
#include "oneflow/core/framework/synced_symbol_map.h"
#include "oneflow/core/framework/rank_group_rpc_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/common/flat_shape.h"
#include "oneflow/core/common/shape_vec.h"
#include "oneflow/core/job/rank_group.h"
#include "oneflow/core/job/rank_group_scope.h"

namespace oneflow {

namespace {

struct FlatTensorConsistency final {

	static Maybe<FlatTensorConsistency> New(
			Symbol<ConsistentTensorMeta> tensor_meta,
  		const RpcToken& tensor_rpc_token) {
		const auto& consistency = std::make_shared<FlatTensorConsistency>();
		JUST(consistency->Init(tensor_meta, tensor_rpc_token));
		return consistency;
	}

	Maybe<void> Init(Symbol<ConsistentTensorMeta> tensor_meta, const RpcToken& tensor_rpc_token) {
		this->synced_tensor_meta =
				JUST(SyncedSymbolMap<ConsistentTensorMeta>::FindOrSync(
					tensor_meta, &SyncSymbolConsistentTensorMeta));
		this->tensor_rpc_token = static_cast<uint64_t>(tensor_rpc_token);
		return Maybe<void>::Ok();
	}

	Maybe<void> Check(Symbol<ConsistentTensorMeta> tensor_meta, const RpcToken& tensor_rpc_token) {
		const auto& this_synced_tensor_meta =
			JUST(SyncedSymbolMap<ConsistentTensorMeta>::Symbol4SyncedSymbolId(this->synced_tensor_meta));
		CHECK_OR_RETURN(this_synced_tensor_meta, tensor_meta);
		CHECK_EQ_OR_RETURN(this->tensor_rpc_token, tensor_rpc_token);
		return Maybe<void>::Ok();
	}

  uint64_t synced_tensor_meta;
	uint64_t tensor_rpc_token;
};

}

CheckConsistencyAsyncRpcCtx::~CheckConsistencyAsyncRpcCtx() {}

Maybe<void> CheckConsistencyAsyncRpcCtx::MakeDataBufferAndCallback(
    int64_t rank, void** buffer, std::size_t* size, std::function<void()>* Callback) {
  const auto& flat_tensor_consistency = std::make_shared<FlatTensorConsistency>();
  *buffer = flat_tensor_consistency.get();
  *size = sizeof(FlatTensorConsistency);
  *Callback = [flat_tensor_consistency]() {};
	flat_tensor_consistency_ = flat_tensor_consistency;
  return Maybe<void>::Ok();
}

Maybe<void> CheckConsistencyAsyncRpcCtx::Check() const {
  JUST(flat_tensor_consistency_->Check(tensor_meta_, tensor_rpc_token_));
  return Maybe<void>::Ok();
}

namespace {

Maybe<void> SendTensorMetaToNextRankInRing(
		const one::Tensor& tensor, Symbol<RankGroup> rank_group, const RpcToken& rpc_token) {
	const auto& tensor_meta = JUST(tensor.consistent_tensor_meta());
	const RpcToken& tensor_rpc_token = JUST(tensor.rpc_token());
	NaiveAsyncRpcCtx ctx(
		[](void** buffer, std::size_t* size, std::function<void()>* Callback)->Maybe<void>{
			const auto& tensor_consistency = JUST(FlatTensorConsistency::New(tensor_meta, tensor_rpc_token));
			*buffer = tensor_consistency.get();
			*size = sizeof(FlatTensorConsistency);
			*Callback = [tensor_consistency]{};
			return Maybe<void>::Ok();
		});
  JUST(RpcUtil::SendToNextRankInRing(rank_group, rpc_token, &ctx));
  return ctx;
}

Maybe<CheckConsistencyAsyncRpcCtx> ReceiveTensorMetaFromPrevRankInRing(
		const one::Tensor& tensor, Symbol<RankGroup> rank_group, const RpcToken& rpc_token) {
	const auto& tensor_meta = JUST(tensor.consistent_tensor_meta());
	const RpcToken& tensor_rpc_token = JUST(tensor.rpc_token());
	const auto& ctx = std::make_shared<CheckConsistencyAsyncRpcCtx>(tensor_meta, tensor_rpc_token);
  JUST(RpcUtil::ReceiveFromPrevRankInRing(rank_group, rpc_token, ctx.get()));
  return ctx;
}

}  // namespace

Maybe<CheckConsistencyAsyncRpcCtx> LaunchTensorMetaConsistencyCheck(const one::Tensor& tensor) {
  const auto& rank_group = JUST(RankGroupScope::CurrentRankGroup());
  const auto& rpc_token = JUST(GetAutoIncrementalRpcToken(rank_group));
  JUST(SendTensorMetaToNextRankInRing(tensor, rpc_token));
  return ReceiveTensorMetaFromPrevRankInRing(tensor, rpc_token);
}

}  // namespace oneflow
