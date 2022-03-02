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
#include "oneflow/core/intrusive/flat_msg.h"
#include "oneflow/core/framework/sync_symbol_nd_sbp.h"
#include "oneflow/core/framework/rank_group_rpc_util.h"
#include "oneflow/core/job/rank_group_scope.h"
#include "oneflow/core/job/sbp_parallel.h"
#include "oneflow/core/common/shape_vec.h"
#include "oneflow/core/common/constant.h"

namespace oneflow {

namespace {

// clang-format off
FLAT_MSG_BEGIN(FlatSplitParallel);
  FLAT_MSG_DEFINE_OPTIONAL(int64_t, axis);
FLAT_MSG_END(FlatSplitParallel);

FLAT_MSG_BEGIN(FlatBroadcastParallel);
FLAT_MSG_END(FlatBroadcastParallel);

FLAT_MSG_BEGIN(FlatPartialSumParallel);
FLAT_MSG_END(FlatPartialSumParallel);

FLAT_MSG_BEGIN(FlatSbpParallel);
 public:
  Maybe<void> Init(const SbpParallel& sbp_parallel) {
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

  Maybe<void> Check(const SbpParallel& sbp_parallel) const {
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

 private:
  FLAT_MSG_DEFINE_ONEOF(parallel_type,
    FLAT_MSG_ONEOF_FIELD(FlatSplitParallel, split_parallel)
    FLAT_MSG_ONEOF_FIELD(FlatBroadcastParallel, broadcast_parallel)
    FLAT_MSG_ONEOF_FIELD(FlatPartialSumParallel, partial_sum_parallel));
FLAT_MSG_END(FlatSbpParallel);

FLAT_MSG_BEGIN(FlatNdSbp);
 public:
  Maybe<void> Init(uint64_t symbol_id, Symbol<NdSbp> nd_sbp) {
    this->set_symbol_id(symbol_id);
    this->set_size(nd_sbp->sbp_parallel_size());
    for (int i = 0; i < this->size(); ++i) {
      const auto& sbp_parallel = nd_sbp->sbp_parallel(i);
      JUST(this->mutable_sbp_parallel()->Mutable(i)->Init(sbp_parallel));
    }
    return Maybe<void>::Ok();
  }

  Maybe<void> Check(uint64_t symbol_id, Symbol<NdSbp> nd_sbp) const {
    CHECK_EQ_OR_RETURN(this->symbol_id(), symbol_id);
    CHECK_EQ_OR_RETURN(this->size(), nd_sbp->sbp_parallel_size());
    for (int i = 0; i < this->size(); ++i) {
      JUST(this->sbp_parallel().Get(i).Check(nd_sbp->sbp_parallel(i)));
    }
    return Maybe<void>::Ok();
  }

 private:
  FLAT_MSG_DEFINE_OPTIONAL(uint64_t, symbol_id);
  FLAT_MSG_DEFINE_OPTIONAL(size_t, size);
  FLAT_MSG_DEFINE_REPEATED(FlatSbpParallel, sbp_parallel, SHAPE_MAX_AXIS_SIZE);
FLAT_MSG_END(FlatNdSbp);
// clang-format on

class FlatNdSbpAsyncTransportCtx : public AsyncTransportCtx {
 public:
  FlatNdSbpAsyncTransportCtx(const TransportToken& transport_token, uint64_t symbol_id,
                             Symbol<NdSbp> nd_sbp)
      : AsyncTransportCtx(transport_token), symbol_id_(symbol_id), nd_sbp_(nd_sbp) {}

  ~FlatNdSbpAsyncTransportCtx() override {}

  Maybe<void> PrepareSendBufferAndCallback(int64_t rank, void** buffer, std::size_t* size,
                                           std::function<void()>* Callback) override {
    const auto& flat_nd_sbp = std::make_shared<FlatNdSbp>();
    JUST(flat_nd_sbp->Init(symbol_id_, nd_sbp_));
    *buffer = flat_nd_sbp.get();
    *size = sizeof(FlatNdSbp);
    *Callback = [flat_nd_sbp]() {};
    return Maybe<void>::Ok();
  }

  Maybe<void> PrepareRecvBufferAndCallback(int64_t rank, void** buffer, std::size_t* size,
                                           std::function<void()>* Callback) override {
    const auto& flat_nd_sbp = std::make_shared<FlatNdSbp>();
    *buffer = flat_nd_sbp.get();
    *size = sizeof(FlatNdSbp);
    *Callback = [flat_nd_sbp]() {};
    flat_nd_sbp_ = flat_nd_sbp;
    return Maybe<void>::Ok();
  }

  Maybe<void> Check() const {
    CHECK_NOTNULL_OR_RETURN(flat_nd_sbp_.get());
    JUST(flat_nd_sbp_->Check(symbol_id_, nd_sbp_));
    return Maybe<void>::Ok();
  }

 private:
  uint64_t symbol_id_;
  Symbol<NdSbp> nd_sbp_;
  std::shared_ptr<FlatNdSbp> flat_nd_sbp_;
};

}  // namespace

namespace {}

Maybe<void> SyncSymbolNdSbp(uint64_t symbol_id, Symbol<NdSbp> symbol) {
  const auto& rank_group = JUST(RankGroupScope::CurrentRankGroup());
  const auto& transport_token =
      JUST(TransportToken::NewTransportToken(kTransportTokenTypeSyncSymbolNdSbp));
  FlatNdSbpAsyncTransportCtx ctx(transport_token, symbol_id, symbol);
  JUST(TransportUtil::SendToNextRankInRing(rank_group, transport_token, &ctx));
  JUST(TransportUtil::ReceiveFromPrevRankInRing(rank_group, transport_token, &ctx));
  JUST_MSG(ctx.WaitDone(), kAsymmetricCodeErrorMsg);
  JUST(ctx.Check());
  return Maybe<void>::Ok();
}

}  // namespace oneflow
