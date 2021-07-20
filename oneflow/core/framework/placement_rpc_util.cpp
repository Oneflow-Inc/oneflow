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
#include "oneflow/core/framework/placement_rpc_util.h"
#include "oneflow/core/framework/rpc_util.h"
#include "oneflow/core/thread/thread_unique_tag.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/job/sorted_rank_ranges.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"

namespace oneflow {

Maybe<void> NaiveTokenCheckAsyncRpcCtx::MakeDataBufferAndCallback(int64_t rank, void** buffer,
                                                                  std::size_t* size,
                                                                  std::function<void()>* Callback) {
  const auto& placeholder = std::make_shared<uint32_t>();
  *buffer = placeholder.get();
  *size = sizeof(uint32_t);
  *Callback = [placeholder]() {};
  return Maybe<void>::Ok();
}

std::atomic<int64_t>* MutTimeoutSeconds4InitializingPlacementMajor() {
  static auto* seconds = new std::atomic<int64_t>(60 * 5);
  return seconds;
}

namespace {

int64_t TimeoutSeconds4InitializingPlacementMajor() {
  return *MutTimeoutSeconds4InitializingPlacementMajor();
}

Maybe<const RpcToken&> RpcToken4InitializingPlacementMajor() {
  uint32_t token_major = JUST(RpcUtil::GetRpcTokenCmdMajor(kInitializingPlacementCmdLocalMajor));
  static thread_local auto* token = new RpcToken(token_major, 0);
  return *token;
}

Maybe<const RpcToken&> RpcToken4CheckingParallelConfSize() {
  auto token_major = JUST(RpcUtil::GetRpcTokenCmdMajor(kCheckingParallelConfSizeCmdLocalMajor));
  static thread_local auto* token = new RpcToken(token_major, 0);
  return *token;
}

Maybe<uint32_t> AutoIncrementalMajorValue4Placement() {
  static const uint32_t kStart = RpcToken::kStartTokenMajor4Placement;
  static auto* major_value = new std::atomic<int64_t>(kStart);
  CHECK_LT_OR_RETURN(static_cast<int64_t>(*major_value), GetMaxVal<uint32_t>());
  return static_cast<uint32_t>(++*major_value);
}

namespace {

struct FlatParallelInfo {
  static std::shared_ptr<FlatParallelInfo> New(uint32_t major, const std::string& parallel_conf) {
    size_t buffer_size = parallel_conf.size();
    auto* ptr =
        reinterpret_cast<FlatParallelInfo*>(std::malloc(buffer_size + sizeof(FlatParallelInfo)));
    std::shared_ptr<FlatParallelInfo> flat_parallel_info(ptr);
    flat_parallel_info->candidate_major = major;
    flat_parallel_info->buffer_size = buffer_size;
    std::memcpy(flat_parallel_info->buffer, parallel_conf.data(), buffer_size);
    return flat_parallel_info;
  }

  static std::shared_ptr<FlatParallelInfo> New(size_t buffer_size) {
    auto* ptr =
        reinterpret_cast<FlatParallelInfo*>(std::malloc(buffer_size + sizeof(FlatParallelInfo)));
    std::shared_ptr<FlatParallelInfo> flat_parallel_info(ptr);
    flat_parallel_info->candidate_major = 0;
    flat_parallel_info->buffer_size = buffer_size;
    return flat_parallel_info;
  }

  size_t TotalSize() const { return this->buffer_size + sizeof(FlatParallelInfo); }

  uint32_t candidate_major;
  size_t buffer_size;
  char buffer[0];
};

static_assert(std::is_standard_layout<FlatParallelInfo>::value, "");

}  // namespace

class BroadcastCandidateMajorAsyncRpcCtx final : public AsyncRpcCtx {
 public:
  BroadcastCandidateMajorAsyncRpcCtx(uint32_t major, const std::string& parallel_conf)
      : flat_parallel_info_(FlatParallelInfo::New(major, parallel_conf)) {}
  ~BroadcastCandidateMajorAsyncRpcCtx() override = default;

  Maybe<void> MakeDataBufferAndCallback(int64_t rank, void** buffer, std::size_t* size,
                                        std::function<void()>* Callback) override {
    const auto& flat_parallel_info = flat_parallel_info_;
    *buffer = flat_parallel_info.get();
    *size = flat_parallel_info->TotalSize();
    *Callback = [flat_parallel_info]() {};
    return Maybe<void>::Ok();
  }

 private:
  std::shared_ptr<FlatParallelInfo> flat_parallel_info_;
};

Maybe<void> BroadcastMyCandidateMajorToAllOtherRanks(Symbol<ParallelDesc> parallel_desc,
                                                     uint32_t major) {
  const auto& rank_ranges =
      JUST(SortedRankRanges::New4SoleDevicePerRankParallelDesc(parallel_desc));
  const auto& token = JUST(RpcToken4InitializingPlacementMajor());
  BroadcastCandidateMajorAsyncRpcCtx ctx(major, parallel_desc->parallel_conf().DebugString());
  JUST(RpcUtil::BroadcastToAllOtherRanks(rank_ranges, token, &ctx));
  // no need to wait callback done.
  return Maybe<void>::Ok();
}

class CollectCandidateMajorAsyncRpcCtx final : public AsyncRpcCtx {
 public:
  CollectCandidateMajorAsyncRpcCtx(const std::string& parallel_conf)
      : parallel_conf_(parallel_conf) {}
  ~CollectCandidateMajorAsyncRpcCtx() override = default;

  Maybe<void> MakeDataBufferAndCallback(int64_t rank, void** buffer, std::size_t* size,
                                        std::function<void()>* Callback) override {
    const auto& flat_parallel_info = FlatParallelInfo::New(parallel_conf_.size());
    *buffer = flat_parallel_info.get();
    *size = flat_parallel_info->TotalSize();
    *Callback = [flat_parallel_info]() {};
    collected_parallel_info_.push_back(flat_parallel_info);
    return Maybe<void>::Ok();
  }

  Maybe<void> ForEachCandidateMajor(const std::function<void(uint32_t)>& DoEach) const {
    for (const auto& parallel_info : collected_parallel_info_) {
      const char* expected_data = parallel_conf_.data();
      size_t size = parallel_conf_.size();
      CHECK_EQ_OR_RETURN(std::memcmp(parallel_info->buffer, expected_data, size), 0);
      DoEach(parallel_info->candidate_major);
    }
    return Maybe<void>::Ok();
  };

 private:
  std::string parallel_conf_;
  std::vector<std::shared_ptr<FlatParallelInfo>> collected_parallel_info_;
};

Maybe<void> ForEachReceivedCandidateMajor(Symbol<ParallelDesc> parallel_desc,
                                          const std::function<void(uint32_t)>& DoEach) {
  const auto& rank_ranges =
      JUST(SortedRankRanges::New4SoleDevicePerRankParallelDesc(parallel_desc));
  const auto& token = JUST(RpcToken4InitializingPlacementMajor());
  CollectCandidateMajorAsyncRpcCtx ctx(parallel_desc->parallel_conf().DebugString());
  JUST(RpcUtil::CollectFromAllOtherRanks(rank_ranges, token, &ctx));
  JUST(RpcUtil::WaitUntilDoneOrTimeout(ctx, TimeoutSeconds4InitializingPlacementMajor()));
  JUST(ctx.ForEachCandidateMajor(DoEach));
  return Maybe<void>::Ok();
}

class BroadcastParallelConfSizeAsyncRpcCtx final : public AsyncRpcCtx {
 public:
  BroadcastParallelConfSizeAsyncRpcCtx(size_t parallel_conf_size)
      : parallel_conf_size_(new size_t(parallel_conf_size)) {}
  ~BroadcastParallelConfSizeAsyncRpcCtx() override = default;

  Maybe<void> MakeDataBufferAndCallback(int64_t rank, void** buffer, std::size_t* size,
                                        std::function<void()>* Callback) override {
    const auto& parallel_conf_size = parallel_conf_size_;
    *buffer = parallel_conf_size.get();
    *size = sizeof(size_t);
    *Callback = [parallel_conf_size]() {};
    return Maybe<void>::Ok();
  }

 private:
  std::shared_ptr<size_t> parallel_conf_size_;
};

Maybe<void> BroadcastParallelConfSizeToOthers(Symbol<SortedRankRanges> rank_ranges,
                                              size_t parallel_conf_size) {
  const auto& token = JUST(RpcToken4CheckingParallelConfSize());
  BroadcastParallelConfSizeAsyncRpcCtx ctx(parallel_conf_size);
  JUST(RpcUtil::BroadcastToAllOtherRanks(rank_ranges, token, &ctx));
  // no need to wait callback done.
  return Maybe<void>::Ok();
}

class CollectParallelConfSizeAsyncRpcCtx final : public AsyncRpcCtx {
 public:
  CollectParallelConfSizeAsyncRpcCtx() = default;
  ~CollectParallelConfSizeAsyncRpcCtx() override = default;

  Maybe<void> MakeDataBufferAndCallback(int64_t rank, void** buffer, std::size_t* size,
                                        std::function<void()>* Callback) override {
    const auto& parallel_conf_size = std::make_shared<size_t>();
    *buffer = parallel_conf_size.get();
    *size = sizeof(size_t);
    *Callback = [parallel_conf_size]() {};
    collected_parallel_conf_sizes_.push_back(parallel_conf_size);
    return Maybe<void>::Ok();
  }

  Maybe<void> ForEachParallelConfSize(const std::function<Maybe<void>(size_t)>& DoEach) const {
    for (const auto& parallel_conf_size : collected_parallel_conf_sizes_) {
      JUST(DoEach(*parallel_conf_size));
    }
    return Maybe<void>::Ok();
  }

 private:
  std::vector<std::shared_ptr<size_t>> collected_parallel_conf_sizes_;
};

Maybe<CollectParallelConfSizeAsyncRpcCtx> CollectParallelConfSizeFromOthers(
    Symbol<SortedRankRanges> rank_ranges) {
  const auto& token = JUST(RpcToken4CheckingParallelConfSize());
  const auto& ctx = std::make_shared<CollectParallelConfSizeAsyncRpcCtx>();
  JUST(RpcUtil::CollectFromAllOtherRanks(rank_ranges, token, ctx.get()));
  return ctx;
}

Maybe<void> CheckParallelConfSize(Symbol<ParallelDesc> parallel_desc) {
  size_t parallel_conf_size = parallel_desc->parallel_conf().DebugString().size();
  const auto& rank_ranges =
      JUST(SortedRankRanges::New4SoleDevicePerRankParallelDesc(parallel_desc));
  JUST(BroadcastParallelConfSizeToOthers(rank_ranges, parallel_conf_size));
  const auto& ctx = JUST(CollectParallelConfSizeFromOthers(rank_ranges));
  JUST(RpcUtil::WaitUntilDoneOrTimeout(*ctx, TimeoutSeconds4InitializingPlacementMajor()));
  JUST(ctx->ForEachParallelConfSize([&](size_t other_parallel_conf_size) -> Maybe<void> {
    CHECK_EQ_OR_RETURN(parallel_conf_size, other_parallel_conf_size);
    return Maybe<void>::Ok();
  }));
  return Maybe<void>::Ok();
}

Maybe<uint32_t> GetSynchronizedMajorValue4Placement(Symbol<ParallelDesc> parallel_desc) {
  uint32_t candidate_major = JUST(AutoIncrementalMajorValue4Placement());
  const auto& max_major = std::make_shared<uint32_t>(candidate_major);
  JUST(BroadcastMyCandidateMajorToAllOtherRanks(parallel_desc, candidate_major));
  JUST(ForEachReceivedCandidateMajor(parallel_desc, [max_major](uint32_t other_candidate_major) {
    *max_major = std::max(*max_major, other_candidate_major);
  }));
  return *max_major;
}

std::unique_ptr<RpcToken>* MutThreadLocalRpcToken(Symbol<ParallelDesc> parallel_desc) {
  static thread_local HashMap<Symbol<ParallelDesc>, std::unique_ptr<RpcToken>>
      parallel_desc2sync_token;
  return &parallel_desc2sync_token[parallel_desc];
}

}  // namespace

Maybe<void> InitCurrentRpcToken(Symbol<ParallelDesc> parallel_desc) {
  auto* token = MutThreadLocalRpcToken(parallel_desc);
  CHECK_OR_RETURN(!static_cast<bool>(*token));
  if (!parallel_desc->containing_current_rank()) { return Maybe<void>::Ok(); }
  JUST(CheckParallelConfSize(parallel_desc));
  uint32_t major_value = JUST(GetSynchronizedMajorValue4Placement(parallel_desc));
  token->reset(new RpcToken(major_value, 0));
  return Maybe<void>::Ok();
}

Maybe<RpcToken> GetAutoIncrementalRpcToken(Symbol<ParallelDesc> parallel_desc) {
  OF_RETURN_IF_ERROR(GetThisThreadUniqueTag()) << "this thread are not tagged with sync label";
  CHECK_OR_RETURN(parallel_desc->containing_current_rank());
  const auto& token = *MutThreadLocalRpcToken(parallel_desc);
  if (!token) { JUST(InitCurrentRpcToken(parallel_desc)); }
  CHECK_OR_RETURN(token);
  return ++*token;
}

Maybe<NaiveTokenCheckAsyncRpcCtx> CheckRpcToken(Symbol<ParallelDesc> parallel_desc) {
  const auto& rank_ranges =
      JUST(SortedRankRanges::New4SoleDevicePerRankParallelDesc(parallel_desc));
  const auto& rpc_token = JUST(GetAutoIncrementalRpcToken(parallel_desc));
  const auto& ctx = std::make_shared<NaiveTokenCheckAsyncRpcCtx>();
  JUST(RpcUtil::SendToNextRankInRing(rank_ranges, rpc_token, ctx.get()));
  JUST(RpcUtil::ReceiveFromPrevRankInRing(rank_ranges, rpc_token, ctx.get()));
  return ctx;
}

}  // namespace oneflow
