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
#ifndef ONEFLOW_CORE_FRAMEWORK_RPC_UTIL_H_
#define ONEFLOW_CORE_FRAMEWORK_RPC_UTIL_H_

#include <atomic>
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/framework/transport_token.h"

namespace oneflow {

class AsyncTransportCtx {
 public:
  explicit AsyncTransportCtx(const TransportToken& transport_token)
      : transport_token_(transport_token), flying_cnt_(new std::atomic<int64_t>(0)) {}
  virtual ~AsyncTransportCtx() = default;

  const TransportToken& transport_token() const { return transport_token_; }
  std::shared_ptr<std::atomic<int64_t>> flying_cnt() const { return flying_cnt_; }

  virtual Maybe<void> PrepareSendBufferAndCallback(int64_t rank, void** buffer, std::size_t* size,
                                                   std::function<void()>* Callback) = 0;

  virtual Maybe<void> PrepareRecvBufferAndCallback(int64_t rank, void** buffer, std::size_t* size,
                                                   std::function<void()>* Callback) = 0;

 private:
  TransportToken transport_token_;
  std::shared_ptr<std::atomic<int64_t>> flying_cnt_;
};

class NaiveAsyncTransportCtx final : public AsyncTransportCtx {
 public:
  NaiveAsyncTransportCtx(
      const TransportToken& transport_token,
      const std::function<Maybe<void>(void**, std::size_t*, std::function<void()>*)>& PrepareSend,
      const std::function<Maybe<void>(void**, std::size_t*, std::function<void()>*)>& PrepareRecv)
      : AsyncTransportCtx(transport_token),
        prepare_send_(PrepareSend),
        prepare_recv_(PrepareRecv) {}

  NaiveAsyncTransportCtx(
      const TransportToken& transport_token,
      const std::function<Maybe<void>(void**, std::size_t*, std::function<void()>*)>& PrepareSend,
      const std::function<Maybe<void>(int64_t, void**, std::size_t*, std::function<void()>*)>&
          PrepareRecvWithRank)
      : AsyncTransportCtx(transport_token),
        prepare_send_(PrepareSend),
        prepare_recv_with_rank_(PrepareRecvWithRank) {}

  NaiveAsyncTransportCtx(
      const TransportToken& transport_token,
      const std::function<Maybe<void>(int64_t, void**, std::size_t*, std::function<void()>*)>&
          PrepareSendWithRank,
      const std::function<Maybe<void>(void**, std::size_t*, std::function<void()>*)>& PrepareRecv)
      : AsyncTransportCtx(transport_token),
        prepare_send_with_rank_(PrepareSendWithRank),
        prepare_recv_(PrepareRecv) {}

  NaiveAsyncTransportCtx(
      const TransportToken& transport_token,
      const std::function<Maybe<void>(int64_t, void**, std::size_t*, std::function<void()>*)>&
          PrepareSendWithRank,
      const std::function<Maybe<void>(int64_t, void**, std::size_t*, std::function<void()>*)>&
          PrepareRecvWithRank)
      : AsyncTransportCtx(transport_token),
        prepare_send_with_rank_(PrepareSendWithRank),
        prepare_recv_with_rank_(PrepareRecvWithRank) {}

  ~NaiveAsyncTransportCtx() override = default;

  Maybe<void> PrepareSendBufferAndCallback(int64_t rank, void** buffer, std::size_t* size,
                                           std::function<void()>* Callback) override {
    if (prepare_send_with_rank_) { return prepare_send_with_rank_(rank, buffer, size, Callback); }
    return prepare_send_(buffer, size, Callback);
  }

  Maybe<void> PrepareRecvBufferAndCallback(int64_t rank, void** buffer, std::size_t* size,
                                           std::function<void()>* Callback) override {
    if (prepare_recv_with_rank_) { return prepare_recv_with_rank_(rank, buffer, size, Callback); }
    return prepare_recv_(buffer, size, Callback);
  }

 private:
  std::function<Maybe<void>(void**, std::size_t*, std::function<void()>*)> prepare_send_;
  std::function<Maybe<void>(int64_t, void**, std::size_t*, std::function<void()>*)>
      prepare_send_with_rank_;
  std::function<Maybe<void>(void**, std::size_t*, std::function<void()>*)> prepare_recv_;
  std::function<Maybe<void>(int64_t, void**, std::size_t*, std::function<void()>*)>
      prepare_recv_with_rank_;
};

class RankGroup;

struct TransportUtil final {
  static int64_t TimeoutSeconds() { return 60 * 5; }

  static Maybe<void> WaitUntilDoneOrTimeout(const AsyncTransportCtx& ctx, int64_t seconds);

  static Maybe<void> SendToNextRankInRing(Symbol<RankGroup> rank_group, const TransportToken& token,
                                          AsyncTransportCtx* ctx);

  static Maybe<void> ReceiveFromPrevRankInRing(Symbol<RankGroup> rank_group,
                                               const TransportToken& token, AsyncTransportCtx* ctx);

  static Maybe<void> BroadcastToAllOtherRanks(Symbol<RankGroup> rank_group,
                                              const TransportToken& token, AsyncTransportCtx* ctx);

  static Maybe<void> CollectFromAllOtherRanks(Symbol<RankGroup> rank_group,
                                              const TransportToken& token, AsyncTransportCtx* ctx);
  static Maybe<void> SendDataToChildrenInHeap(const std::vector<int64_t>& rank_heap,
                                              const TransportToken& token, AsyncTransportCtx* ctx);
  static Maybe<void> ReceiveDataFromParentInHeap(const std::vector<int64_t>& rank_heap,
                                                 const TransportToken& token,
                                                 AsyncTransportCtx* ctx);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_RPC_UTIL_H_
