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
#include <memory>
#include <chrono>
#include "oneflow/core/framework/rank_group_rpc_util.h"
#include "oneflow/core/framework/rpc_util.h"
#include "oneflow/core/thread/thread_unique_tag.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/job/rank_group.h"
#include "oneflow/core/job/rank_group_scope.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"

namespace oneflow {

std::atomic<int64_t>* MutTimeoutSeconds4InitializingPlacementMajor() {
  static auto* seconds = new std::atomic<int64_t>(60 * 5);
  return seconds;
}

namespace {

int64_t TimeoutSeconds4InitializingPlacementMajor() {
  return *MutTimeoutSeconds4InitializingPlacementMajor();
}

HashMap<Symbol<RankGroup>, std::unique_ptr<RpcToken>>* MutThreadLocalRpcTokenMap() {
	static thread_local HashMap<Symbol<RankGroup>, std::unique_ptr<RpcToken>> map;
  return &map;
}

Maybe<RpcToken> GetInitialRpcToken() {
	int32_t consistent_thread_id = JUST(RpcUtil::GetCurrentThreadUid());
	return RpcToken::NewOpTensorMetaRpcToken(consistent_thread_id, 0);
}

}  // namespace

Maybe<void> MakeInitialRankGroupRpcToken() {
	const auto& root_rank_group = JUST(RankGroupScope::rootRankGroup());
	{
		const auto& current_rank_group = JUST(RankGroupScope::CurrentRankGroup());
		CHECK_EQ_OR_RETURN(root_rank_group, current_rank_group);
	}
	auto* map = MutThreadLocalRpcTokenMap();
	CHECK_OR_RETURN(map->empty());
	map->emplace(root_rank_group, JUST(GetInitialRpcToken()));
  return Maybe<void>::Ok();
}

Maybe<RpcToken> GetAutoIncrementalRpcToken(Symbol<RankGroup> rank_group) {
  OF_RETURN_IF_ERROR(GetThisThreadUniqueTag()) << "this thread are not tagged with sync label";
  CHECK_OR_RETURN(rank_group->ContainingCurrentRank());
  const auto& token = JUST(MapAt(*MutThreadLocalRpcTokenMap(), rank_group));
  CHECK_OR_RETURN(token);
  return ++*token;
}

Maybe<NaiveAsyncRpcCtx> CheckRpcToken(Symbol<RankGroup> rank_group) {
  const auto& rpc_token = JUST(GetAutoIncrementalRpcToken(rank_group));
  const auto& ctx = std::make_shared<NaiveAsyncRpcCtx>(
		[](void** buffer, std::size_t* size, std::function<void()>* Callback)->Maybe<void>{
			const auto& placeholder = std::make_shared<uint32_t>();
			*buffer = placeholder.get();
			*size = sizeof(uint32_t);
			*Callback = [placeholder]() {};
			return Maybe<void>::Ok();
		});
  JUST(RpcUtil::SendToNextRankInRing(rank_group, rpc_token, ctx.get()));
  JUST(RpcUtil::ReceiveFromPrevRankInRing(rank_group, rpc_token, ctx.get()));
  return ctx;
}

Maybe<int64_t> GetCurrentRankGroupId() {
	const auto& rank_group = RankGroupScope::CurrentRankGroup();
	const auto& root_rank_group = RankGroupScope::RootRankGroup();
	CHECK_EQ_OR_RETURN(rank_group, root_rank_group);
	return static_cast<int64_t>(0);
}

}  // namespace oneflow
