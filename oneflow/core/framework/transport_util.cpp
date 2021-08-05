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
#include "oneflow/core/framework/transport_token.h"
#include "oneflow/core/framework/transport_util.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/transport/transport.h"
#include "oneflow/core/thread/consistent_unique_id.h"
#include "oneflow/core/job/rank_group.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"

namespace oneflow {

/*static*/ Maybe<void> TransportUtil::WaitUntilDoneOrTimeout(const AsyncTransportCtx& ctx,
                                                             int64_t seconds) {
  const auto& start = std::chrono::steady_clock::now();
  const auto& cond_cnt = ctx.flying_cnt();
  while (*cond_cnt > 0) {
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    CHECK_LT_OR_RETURN(elapsed_seconds.count(), seconds)
        << Error::TimeoutError() << "Timeout error at " << seconds << " seconds.";
  }
  if (ctx.transport_token().type() == kCtrlTransportTokenType) {
    JUST(ctx.transport_token().ReleaseCtrlTransportToken());
  }
  return Maybe<void>::Ok();
}

namespace {

template<Maybe<void> (*SendOrRecv)(const TransportToken&, int64_t, void*, std::size_t,
                                   const std::function<void()>&),
         Maybe<void> (AsyncTransportCtx::*Prepare)(int64_t, void**, std::size_t*,
                                                   std::function<void()>*),
         typename ForEachRankT>
Maybe<void> AccessToOtherRanks(const ForEachRankT& ForEachRank, const TransportToken& token,
                               AsyncTransportCtx* ctx) {
  const auto& flying_cnt = ctx->flying_cnt();
  JUST(ForEachRank([&](int64_t rank) -> Maybe<void> {
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

template<Maybe<void> (*SendOrRecv)(const TransportToken&, int64_t, void*, std::size_t,
                                   const std::function<void()>&),
         Maybe<void> (AsyncTransportCtx::*Prepare)(int64_t, void**, std::size_t*,
                                                   std::function<void()>*)>
Maybe<void> AccessToAllOtherRanks(Symbol<RankGroup> rank_group, const TransportToken& token,
                                  AsyncTransportCtx* ctx) {
  CHECK_OR_RETURN(rank_group->ContainingCurrentRank());
  const auto& ForEachRank = [&](const std::function<Maybe<void>(int64_t)>& DoEach) -> Maybe<void> {
    return rank_group->ForEachRank(DoEach);
  };
  return AccessToOtherRanks<SendOrRecv, Prepare>(ForEachRank, token, ctx);
}

template<Maybe<int64_t> (RankGroup::*GetPrevOrNext)() const,
         Maybe<void> (*SendOrRecv)(const TransportToken&, int64_t, void*, std::size_t,
                                   const std::function<void()>&),
         Maybe<void> (AsyncTransportCtx::*Prepare)(int64_t, void**, std::size_t*,
                                                   std::function<void()>*)>
Maybe<void> AccessToNearbyRank(Symbol<RankGroup> rank_group, const TransportToken& token,
                               AsyncTransportCtx* ctx) {
  CHECK_OR_RETURN(rank_group->ContainingCurrentRank());
  const auto& ForEachRank = [&](const std::function<Maybe<void>(int64_t)>& DoEach) -> Maybe<void> {
    return DoEach(JUST(((*rank_group).*GetPrevOrNext)()));
  };
  return AccessToOtherRanks<SendOrRecv, Prepare>(ForEachRank, token, ctx);
}

Maybe<void> Send(const TransportToken& token, int64_t rank, void* buffer, std::size_t size,
                 const std::function<void()>& Callback) {
#ifdef __linux__
  auto* transport = JUST(GlobalMaybe<Transport>());
  TransportToken transport_token(token);
  JUST(transport_token.set_src_rank(GlobalProcessCtx::Rank()));
  JUST(transport_token.set_dst_rank(rank));
  transport->Send(static_cast<uint64_t>(transport_token), rank, buffer, size, Callback);
  return Maybe<void>::Ok();
#else
  UNIMPLEMENTED();
  return Maybe<void>::Ok();
#endif  // __linux__
}

Maybe<void> Recv(const TransportToken& token, int64_t rank, void* buffer, std::size_t size,
                 const std::function<void()>& Callback) {
#ifdef __linux__
  auto* transport = JUST(GlobalMaybe<Transport>());
  TransportToken transport_token(token);
  JUST(transport_token.set_src_rank(rank));
  JUST(transport_token.set_dst_rank(GlobalProcessCtx::Rank()));
  transport->Receive(static_cast<uint64_t>(transport_token), rank, buffer, size, Callback);
  return Maybe<void>::Ok();
#else
  UNIMPLEMENTED();
  return Maybe<void>::Ok();
#endif  // __linux__
}

}  // namespace

/*static*/ Maybe<void> TransportUtil::BroadcastToAllOtherRanks(Symbol<RankGroup> rank_group,
                                                               const TransportToken& token,
                                                               AsyncTransportCtx* ctx) {
  JUST(AccessToAllOtherRanks<&Send, &AsyncTransportCtx::PrepareSendBufferAndCallback>(rank_group,
                                                                                      token, ctx));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> TransportUtil::CollectFromAllOtherRanks(Symbol<RankGroup> rank_group,
                                                               const TransportToken& token,
                                                               AsyncTransportCtx* ctx) {
  JUST(AccessToAllOtherRanks<&Recv, &AsyncTransportCtx::PrepareRecvBufferAndCallback>(rank_group,
                                                                                      token, ctx));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> TransportUtil::SendToNextRankInRing(Symbol<RankGroup> rank_group,
                                                           const TransportToken& token,
                                                           AsyncTransportCtx* ctx) {
  JUST(
      AccessToNearbyRank<&RankGroup::GetNextRankInRing, &Send,
                         &AsyncTransportCtx::PrepareSendBufferAndCallback>(rank_group, token, ctx));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> TransportUtil::ReceiveFromPrevRankInRing(Symbol<RankGroup> rank_group,
                                                                const TransportToken& token,
                                                                AsyncTransportCtx* ctx) {
  JUST(
      AccessToNearbyRank<&RankGroup::GetPrevRankInRing, &Recv,
                         &AsyncTransportCtx::PrepareRecvBufferAndCallback>(rank_group, token, ctx));
  return Maybe<void>::Ok();
}

namespace {

Maybe<int64_t> GetCurrentRankIndex(const std::vector<int64_t>& rank_heap) {
  for (int i = 0; i < rank_heap.size(); ++i) {
    if (rank_heap.at(i) == GlobalProcessCtx::Rank()) { return i; }
  }
  UNIMPLEMENTED_THEN_RETURN();
}

}  // namespace

/*static*/ Maybe<void> TransportUtil::SendDataToChildrenInHeap(
    const std::vector<int64_t>& rank_heap, const TransportToken& token, AsyncTransportCtx* ctx) {
  int64_t current_rank_index = JUST(GetCurrentRankIndex(rank_heap));
  const auto& ForEachRank = [&](const std::function<Maybe<void>(int64_t)>& DoEach) -> Maybe<void> {
    int64_t left_index = current_rank_index * 2 + 1;
    if (left_index < rank_heap.size()) { JUST(DoEach(rank_heap.at(left_index))); }
    int64_t right_index = current_rank_index * 2 + 2;
    if (right_index < rank_heap.size()) { JUST(DoEach(rank_heap.at(right_index))); }
    return Maybe<void>::Ok();
  };
  return AccessToOtherRanks<&Send, &AsyncTransportCtx::PrepareSendBufferAndCallback>(ForEachRank,
                                                                                     token, ctx);
}

/*static*/ Maybe<void> TransportUtil::ReceiveDataFromParentInHeap(
    const std::vector<int64_t>& rank_heap, const TransportToken& token, AsyncTransportCtx* ctx) {
  int64_t current_rank_index = JUST(GetCurrentRankIndex(rank_heap));
  const auto& ForEachRank = [&](const std::function<Maybe<void>(int64_t)>& DoEach) -> Maybe<void> {
    if (current_rank_index == 0) { return Maybe<void>::Ok(); }
    return DoEach(rank_heap.at((current_rank_index - 1) / 2));
  };
  return AccessToOtherRanks<&Recv, &AsyncTransportCtx::PrepareRecvBufferAndCallback>(ForEachRank,
                                                                                     token, ctx);
}

}  // namespace oneflow
