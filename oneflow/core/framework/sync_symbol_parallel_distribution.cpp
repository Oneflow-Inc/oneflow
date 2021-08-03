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
#include "oneflow/core/object_msg/flat_msg.h"
#include "oneflow/core/framework/sync_symbol_parallel_distribution.h"
#include "oneflow/core/framework/rank_group_rpc_util.h"
#include "oneflow/core/job/rank_group_scope.h"
#include "oneflow/core/job/sbp_parallel.cfg.h"
#include "oneflow/core/common/shape_vec.h"

namespace oneflow {

namespace {

FLAT_MSG_BEGIN(FlatSplitParallel);
FLAT_MSG_DEFINE_OPTIONAL(int64_t, axis);
FLAT_MSG_END(FlatSplitParallel);

FLAT_MSG_BEGIN(FlatBroadcastParallel);
FLAT_MSG_END(FlatBroadcastParallel);

FLAT_MSG_BEGIN(FlatPartialSumParallel);
FLAT_MSG_END(FlatPartialSumParallel);

FLAT_MSG_BEGIN(FlatSbpParallel);
Maybe<void> Init(const cfg::SbpParallel& sbp_parallel) {
  if (sbp_parallel.has_split_parallel()) {
    this->mutable_split_parallel()->set_axis(sbp_parallel.split_parallel().axis());
  } else if (sbp_parallel.has_broadcast_parallel()) {
    this->mutable_broadcast_parallel();
  } else if (sbp_parallel.has_partial_sum_parallel()) {
    this->mutable_partial_sum_parallel();
  } else {
    OF_UNIMPLEMENTED();
  }
  return Maybe<void>::Ok();
}

Maybe<void> Check(const cfg::SbpParallel& sbp_parallel) const {
  if (sbp_parallel.has_split_parallel()) {
    CHECK_EQ_OR_RETURN(this->split_parallel().axis(), sbp_parallel.split_parallel().axis());
  } else if (sbp_parallel.has_broadcast_parallel()) {
    CHECK_OR_RETURN(this->has_broadcast_parallel());
  } else if (sbp_parallel.has_partial_sum_parallel()) {
    CHECK_OR_RETURN(this->has_partial_sum_parallel());
  } else {
    OF_UNIMPLEMENTED();
  }
  return Maybe<void>::Ok();
}

FLAT_MSG_DEFINE_ONEOF(parallel_type,
                      FLAT_MSG_ONEOF_FIELD(FlatSplitParallel, split_parallel)
                          FLAT_MSG_ONEOF_FIELD(FlatBroadcastParallel, broadcast_parallel)
                              FLAT_MSG_ONEOF_FIELD(FlatPartialSumParallel, partial_sum_parallel));
FLAT_MSG_END(FlatSbpParallel);

FLAT_MSG_BEGIN(FlatParallelDistribution);
OF_PUBLIC Maybe<void> Init(uint64_t symbol_id,
                           Symbol<cfg::ParallelDistribution> parallel_distribution) {
  this->set_symbol_id(symbol_id);
  this->set_size(parallel_distribution->sbp_parallel_size());
  for (int i = 0; i < this->size(); ++i) {
    const auto& sbp_parallel = parallel_distribution->sbp_parallel(i);
    JUST(this->mutable_sbp_parallel()->Mutable(i)->Init(sbp_parallel));
  }
  return Maybe<void>::Ok();
}

OF_PUBLIC Maybe<void> Check(uint64_t symbol_id,
                            Symbol<cfg::ParallelDistribution> parallel_distribution) const {
  CHECK_EQ_OR_RETURN(this->symbol_id(), symbol_id);
  CHECK_EQ_OR_RETURN(this->size(), parallel_distribution->sbp_parallel_size());
  for (int i = 0; i < this->size(); ++i) {
    JUST(this->sbp_parallel().Get(i).Check(parallel_distribution->sbp_parallel(i)));
  }
  return Maybe<void>::Ok();
}

FLAT_MSG_DEFINE_OPTIONAL(uint64_t, symbol_id);
FLAT_MSG_DEFINE_OPTIONAL(size_t, size);
FLAT_MSG_DEFINE_REPEATED(FlatSbpParallel, sbp_parallel, SHAPE_MAX_AXIS_SIZE);
FLAT_MSG_END(FlatParallelDistribution);

class FlatParallelDistributionAsyncRpcCtx : public AsyncRpcCtx {
 public:
  FlatParallelDistributionAsyncRpcCtx(const RpcToken& rpc_token, uint64_t symbol_id,
                                      Symbol<cfg::ParallelDistribution> parallel_distribution)
      : AsyncRpcCtx(rpc_token),
        symbol_id_(symbol_id),
        parallel_distribution_(parallel_distribution) {}

  ~FlatParallelDistributionAsyncRpcCtx() override {}

  Maybe<void> PrepareSendBufferAndCallback(int64_t rank, void** buffer, std::size_t* size,
                                           std::function<void()>* Callback) override {
    const auto& flat_parallel_distribution = std::make_shared<FlatParallelDistribution>();
    JUST(flat_parallel_distribution->Init(symbol_id_, parallel_distribution_));
    *buffer = flat_parallel_distribution.get();
    *size = sizeof(FlatParallelDistribution);
    *Callback = [flat_parallel_distribution]() {};
    return Maybe<void>::Ok();
  }

  Maybe<void> PrepareRecvBufferAndCallback(int64_t rank, void** buffer, std::size_t* size,
                                           std::function<void()>* Callback) override {
    const auto& flat_parallel_distribution = std::make_shared<FlatParallelDistribution>();
    *buffer = flat_parallel_distribution.get();
    *size = sizeof(FlatParallelDistribution);
    *Callback = [flat_parallel_distribution]() {};
    flat_parallel_distribution_ = flat_parallel_distribution;
    return Maybe<void>::Ok();
  }

  Maybe<void> Check() const {
    CHECK_NOTNULL_OR_RETURN(flat_parallel_distribution_.get());
    JUST(flat_parallel_distribution_->Check(symbol_id_, parallel_distribution_));
    return Maybe<void>::Ok();
  }

 private:
  uint64_t symbol_id_;
  Symbol<cfg::ParallelDistribution> parallel_distribution_;
  std::shared_ptr<FlatParallelDistribution> flat_parallel_distribution_;
};

}  // namespace

namespace {}

Maybe<void> SyncSymbolParallelDistribution(uint64_t symbol_id,
                                           Symbol<cfg::ParallelDistribution> symbol) {
  const auto& rank_group = JUST(RankGroupScope::CurrentRankGroup());
  const auto& rpc_token =
      JUST(RpcToken::AcquireCtrlRpcToken(kRankGroupRpcCmdSyncSymbolParallelDistribution));
  FlatParallelDistributionAsyncRpcCtx ctx(rpc_token, symbol_id, symbol);
  JUST(RpcUtil::SendToNextRankInRing(rank_group, rpc_token, &ctx));
  JUST(RpcUtil::ReceiveFromPrevRankInRing(rank_group, rpc_token, &ctx));
  JUST(RpcUtil::WaitUntilDoneOrTimeout(ctx, RpcUtil::TimeoutSeconds()));
  JUST(ctx.Check());
  return Maybe<void>::Ok();
}

}  // namespace oneflow
