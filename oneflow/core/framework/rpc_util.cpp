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
#include "oneflow/core/thread/thread_unique_tag.h"
#include "oneflow/core/job/sorted_rank_ranges.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"

namespace oneflow {

namespace {

Maybe<uint32_t> GetCurrentThreadUid() {
  const auto& thread_unique_tag = JUST(GetThisThreadUniqueTag());
  // Only the main thread supported now.
  CHECK_EQ_OR_RETURN(thread_unique_tag, "main");
  return 0;
}

}  // namespace

/*static*/ Maybe<uint32_t> RpcUtil::GetRpcTokenCmdMajor(RpcTokenCmdLocalMajor cmd_local_major) {
  CHECK_LT_OR_RETURN(cmd_local_major, kRpcTokenCmdLocalMajorSize);
  uint32_t thread_uid = JUST(GetCurrentThreadUid());
  static const uint32_t kOffset = RpcToken::kStartTokenMajor4Cmd;
  static const uint32_t kSize = kRpcTokenCmdLocalMajorSize;
  uint32_t ret = kOffset + thread_uid * kSize + cmd_local_major;
  static const uint32_t kLimit = RpcToken::kStartTokenMajor4Placement;
  CHECK_LT_OR_RETURN(ret, kLimit);
  return ret;
}

/*static*/ Maybe<void> RpcUtil::WaitUntilDoneOrTimeout(const AsyncRpcCtx& ctx, int64_t seconds) {
  const auto& start = std::chrono::steady_clock::now();
  const auto& cond_cnt = ctx.flying_cnt();
  while (*cond_cnt > 0) {
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    CHECK_LT_OR_RETURN(elapsed_seconds.count(), seconds)
        << Error::TimeoutError() << "Timeout error at " << seconds << " seconds.";
  }
  return Maybe<void>::Ok();
}

namespace {

template<Maybe<void> (*SendOrRecv)(const RpcToken&, int64_t, void*, std::size_t,
                                   const std::function<void()>&)>
Maybe<void> AccessToAllOtherRanks(Symbol<SortedRankRanges> rank_ranges, const RpcToken& token,
                                  AsyncRpcCtx* ctx) {
  CHECK_OR_RETURN(rank_ranges->ContainingCurrentRank());
  const auto& flying_cnt = ctx->flying_cnt();
  JUST(rank_ranges->ForEachRank([&](int64_t rank) -> Maybe<void> {
    if (rank == GlobalProcessCtx::Rank()) { return Maybe<void>::Ok(); }
    ++*flying_cnt;
    void* buffer = nullptr;
    std::size_t size = 0;
    std::function<void()> Callback;
    JUST(ctx->MakeDataBufferAndCallback(rank, &buffer, &size, &Callback));
    JUST(SendOrRecv(token, rank, buffer, size, [flying_cnt, Callback]() {
      Callback();
      --*flying_cnt;
    }));
    return Maybe<void>::Ok();
  }));
  return Maybe<void>::Ok();
}

template<Maybe<int64_t> (SortedRankRanges::*GetPrevOrNext)() const,
         Maybe<void> (*SendOrRecv)(const RpcToken&, int64_t, void*, std::size_t,
                                   const std::function<void()>&)>
Maybe<void> AccessToNearbyRank(Symbol<SortedRankRanges> rank_ranges, const RpcToken& token,
                               AsyncRpcCtx* ctx) {
  if (rank_ranges->size() == 1) { return Maybe<void>::Ok(); }
  const auto* rank_ranges_ptr = &*rank_ranges;
  int64_t rank = JUST((rank_ranges_ptr->*GetPrevOrNext)());
  CHECK_NE_OR_RETURN(rank, GlobalProcessCtx::Rank());
  const auto& flying_cnt = ctx->flying_cnt();
  ++*flying_cnt;
  void* buffer = nullptr;
  std::size_t size = 0;
  std::function<void()> Callback;
  JUST(ctx->MakeDataBufferAndCallback(rank, &buffer, &size, &Callback));
  JUST(SendOrRecv(token, rank, buffer, size, [flying_cnt, Callback]() {
    Callback();
    --*flying_cnt;
  }));
  return Maybe<void>::Ok();
}

Maybe<void> Send(const RpcToken& token, int64_t rank, void* buffer, std::size_t size,
                 const std::function<void()>& Callback) {
  auto* transport = JUST(GlobalMaybe<Transport>());
  transport->Send(static_cast<uint64_t>(token), rank, buffer, size, Callback);
  return Maybe<void>::Ok();
}

Maybe<void> Recv(const RpcToken& token, int64_t rank, void* buffer, std::size_t size,
                 const std::function<void()>& Callback) {
  auto* transport = JUST(GlobalMaybe<Transport>());
  transport->Receive(static_cast<uint64_t>(token), rank, buffer, size, Callback);
  return Maybe<void>::Ok();
}

}  // namespace

/*static*/ Maybe<void> RpcUtil::BroadcastToAllOtherRanks(Symbol<SortedRankRanges> rank_ranges,
                                                         const RpcToken& token, AsyncRpcCtx* ctx) {
  JUST(AccessToAllOtherRanks<&Send>(rank_ranges, token, ctx));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> RpcUtil::CollectFromAllOtherRanks(Symbol<SortedRankRanges> rank_ranges,
                                                         const RpcToken& token, AsyncRpcCtx* ctx) {
  JUST(AccessToAllOtherRanks<&Recv>(rank_ranges, token, ctx));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> RpcUtil::SendToNextRankInRing(Symbol<SortedRankRanges> rank_ranges,
                                                     const RpcToken& token, AsyncRpcCtx* ctx) {
  JUST(AccessToNearbyRank<&SortedRankRanges::GetNextRankInRing, &Send>(rank_ranges, token, ctx));
  return Maybe<void>::Ok();
}

/*static*/ Maybe<void> RpcUtil::ReceiveFromPrevRankInRing(Symbol<SortedRankRanges> rank_ranges,
                                                          const RpcToken& token, AsyncRpcCtx* ctx) {
  JUST(AccessToNearbyRank<&SortedRankRanges::GetPrevRankInRing, &Recv>(rank_ranges, token, ctx));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
