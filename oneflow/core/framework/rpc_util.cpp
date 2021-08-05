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
#include "oneflow/core/framework/rpc_token.h"
#include "oneflow/core/framework/rpc_util.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/transport/transport.h"
#include "oneflow/core/thread/consistent_unique_id.h"
#include "oneflow/core/job/rank_group.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"

namespace oneflow {

/*static*/ Maybe<void> RpcUtil::WaitUntilDoneOrTimeout(const AsyncRpcCtx& ctx, int64_t seconds) {
  const auto& start = std::chrono::steady_clock::now();
  const auto& cond_cnt = ctx.flying_cnt();
  while (*cond_cnt > 0) {
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    CHECK_LT_OR_RETURN(elapsed_seconds.count(), seconds)
        << Error::TimeoutError() << "Timeout error at " << seconds << " seconds.";
  }
  if (ctx.rpc_token().type() == kCtrlRpcTokenType) { JUST(ctx.rpc_token().ReleaseCtrlRpcToken()); }
  return Maybe<void>::Ok();
}

namespace {

template<Maybe<void> (*SendOrRecv)(const RpcToken&, int64_t, void*, std::size_t,
                                   const std::function<void()>&),
         Maybe<void> (AsyncRpcCtx::*Prepare)(int64_t, void**, std::size_t*, std::function<void()>*)>
Maybe<void> AccessToAllOtherRanks(Symbol<RankGroup> rank_group, const RpcToken& token,
                                  AsyncRpcCtx* ctx) {
  CHECK_OR_RETURN(rank_group->ContainingCurrentRank());
  const auto& flying_cnt = ctx->flying_cnt();
  JUST(rank_group->ForEachRank([&](int64_t rank) -> Maybe<void> {
    if (rank == GlobalProcessCtx::Rank()) { return Maybe<void>::Ok(); }
    ++*flying_cnt;
    void* buffer = nullptr;
    std::size_t size = 0;
    std::function<void()> Callback;
    JUST((ctx->*Prepare)(rank, &buffer, &size, &Callback));
    JUST(SendOrRecv(token, rank, buffer, size, [flying_cnt, Callback]() {
      Callback();
      --*flying_cnt;
    }));
    return Maybe<void>::Ok();
  }));
  return Maybe<void>::Ok();
}

template<Maybe<int64_t> (RankGroup::*GetPrevOrNext)() const,
         Maybe<void> (*SendOrRecv)(const RpcToken&, int64_t, void*, std::size_t,
                                   const std::function<void()>&),
         Maybe<void> (AsyncRpcCtx::*Prepare)(int64_t, void**, std::size_t*, std::function<void()>*)>
Maybe<void> AccessToNearbyRank(Symbol<RankGroup> rank_group, const RpcToken& token,
                               AsyncRpcCtx* ctx) {
  if (rank_group->size() == 1) { return Maybe<void>::Ok(); }
  const auto* rank_ranges_ptr = &*rank_group;
  int64_t rank = JUST((rank_ranges_ptr->*GetPrevOrNext)());
  CHECK_NE_OR_RETURN(rank, GlobalProcessCtx::Rank());
  const auto& flying_cnt = ctx->flying_cnt();
  ++*flying_cnt;
  void* buffer = nullptr;
  std::size_t size = 0;
  std::function<void()> Callback;
  JUST((ctx->*Prepare)(rank, &buffer, &size, &Callback));
  JUST(SendOrRecv(token, rank, buffer, size, [flying_cnt, Callback]() {
    Callback();
    --*flying_cnt;
  }));
  return Maybe<void>::Ok();
}

Maybe<void> Send(const RpcToken& token, int64_t rank, void* buffer, std::size_t size,
                 const std::function<void()>& Callback) {
  auto* transport = JUST(GlobalMaybe<Transport>());
  RpcToken transport_token(token);
  JUST(transport_token.set_src_rank(GlobalProcessCtx::Rank()));
  JUST(transport_token.set_dst_rank(rank));
  transport->Send(static_cast<uint64_t>(transport_token), rank, buffer, size, Callback);
  return Maybe<void>::Ok();
}

Maybe<void> Recv(const RpcToken& token, int64_t rank, void* buffer, std::size_t size,
                 const std::function<void()>& Callback) {
  auto* transport = JUST(GlobalMaybe<Transport>());
  RpcToken transport_token(token);
  JUST(transport_token.set_src_rank(rank));
  JUST(transport_token.set_dst_rank(GlobalProcessCtx::Rank()));
  transport->Receive(static_cast<uint64_t>(transport_token), rank, buffer, size, Callback);
  return Maybe<void>::Ok();
}

}  // namespace

/*static*/ Maybe<void> RpcUtil::BroadcastToAllOtherRanks(Symbol<RankGroup> rank_group,
                                                         const RpcToken& token, AsyncRpcCtx* ctx) {
  JUST(AccessToAllOtherRanks<&Send, &AsyncRpcCtx::PrepareSendBufferAndCallback>(rank_group, token,
                                                                                ctx));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> RpcUtil::CollectFromAllOtherRanks(Symbol<RankGroup> rank_group,
                                                         const RpcToken& token, AsyncRpcCtx* ctx) {
  JUST(AccessToAllOtherRanks<&Recv, &AsyncRpcCtx::PrepareRecvBufferAndCallback>(rank_group, token,
                                                                                ctx));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> RpcUtil::SendToNextRankInRing(Symbol<RankGroup> rank_group,
                                                     const RpcToken& token, AsyncRpcCtx* ctx) {
  JUST(AccessToNearbyRank<&RankGroup::GetNextRankInRing, &Send,
                          &AsyncRpcCtx::PrepareSendBufferAndCallback>(rank_group, token, ctx));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> RpcUtil::ReceiveFromPrevRankInRing(Symbol<RankGroup> rank_group,
                                                          const RpcToken& token, AsyncRpcCtx* ctx) {
  JUST(AccessToNearbyRank<&RankGroup::GetPrevRankInRing, &Recv,
                          &AsyncRpcCtx::PrepareRecvBufferAndCallback>(rank_group, token, ctx));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
