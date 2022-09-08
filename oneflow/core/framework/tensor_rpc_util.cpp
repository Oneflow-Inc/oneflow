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
#include "oneflow/core/framework/tensor_rpc_util.h"
#include "oneflow/core/framework/sync_symbol_global_tensor_meta.h"
#include "oneflow/core/framework/sync_symbol_nd_sbp.h"
#include "oneflow/core/framework/synced_symbol_map.h"
#include "oneflow/core/framework/rank_group_rpc_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/common/flat_shape.h"
#include "oneflow/core/common/shape_vec.h"
#include "oneflow/core/intrusive/flat_msg.h"
#include "oneflow/core/job/rank_group.h"
#include "oneflow/core/job/rank_group_scope.h"
#include "oneflow/core/common/constant.h"

namespace oneflow {
namespace private_details {

struct FlatTensorConsistency;

class CheckConsistencyAsyncTransportCtx : public AsyncTransportCtx {
 public:
  CheckConsistencyAsyncTransportCtx(const TransportToken& transport_token,
                                    Symbol<one::GlobalTensorMeta> tensor_meta,
                                    const Optional<Symbol<NdSbp>>& consumer_nd_sbp_constraint,
                                    const TransportToken& tensor_transport_token)
      : AsyncTransportCtx(transport_token),
        tensor_meta_(tensor_meta),
        consumer_nd_sbp_constraint_(consumer_nd_sbp_constraint),
        tensor_transport_token_(tensor_transport_token) {}

  ~CheckConsistencyAsyncTransportCtx() override;

  Maybe<void> PrepareSendBufferAndCallback(int64_t rank, void** buffer, std::size_t* size,
                                           std::function<void()>* Callback) override;

  Maybe<void> PrepareRecvBufferAndCallback(int64_t rank, void** buffer, std::size_t* size,
                                           std::function<void()>* Callback) override;

  Maybe<void> Check() const;

 private:
  Symbol<one::GlobalTensorMeta> tensor_meta_;
  Optional<Symbol<NdSbp>> consumer_nd_sbp_constraint_;
  TransportToken tensor_transport_token_;
  std::shared_ptr<FlatTensorConsistency> flat_tensor_consistency_;
};

// clang-format off
FLAT_MSG_BEGIN(FlatTensorConsistency);
 public:
  static Maybe<FlatTensorConsistency> New() {
    const auto& consistency = std::make_shared<FlatTensorConsistency>();
    consistency->clear();
    return consistency;
  }
  static Maybe<FlatTensorConsistency> New(
      Symbol<one::GlobalTensorMeta> tensor_meta,
      const Optional<Symbol<NdSbp>>& consumer_nd_sbp_constraint,
      const TransportToken& tensor_transport_token) {
    const auto& consistency = std::make_shared<FlatTensorConsistency>();
    consistency->clear();
    JUST(consistency->Init(tensor_meta, consumer_nd_sbp_constraint, tensor_transport_token));
    return consistency;
  }

  Maybe<void> Check(Symbol<one::GlobalTensorMeta> tensor_meta,
    const Optional<Symbol<NdSbp>>& consumer_nd_sbp_constraint,
                    const TransportToken& tensor_transport_token) {
    const auto& this_synced_tensor_meta =
        JUST(SyncedSymbolMap<one::GlobalTensorMeta>::Symbol4SyncedSymbolId(
            this->synced_tensor_meta_symbol_id()));
    CHECK_OR_RETURN(this_synced_tensor_meta == tensor_meta);
    CHECK_EQ_OR_RETURN(consumer_nd_sbp_constraint.has_value(),
                       this->has_consumer_nd_sbp_constraint_symbol_id());
    if (this->has_consumer_nd_sbp_constraint_symbol_id()) {
      const auto& that_rank_constaint =
          JUST(SyncedSymbolMap<one::GlobalTensorMeta>::Symbol4SyncedSymbolId(
            this->consumer_nd_sbp_constraint_symbol_id()))->nd_sbp();
      const auto& this_rank_constaint = JUST(consumer_nd_sbp_constraint);
      CHECK_OR_RETURN(this_rank_constaint == that_rank_constaint);
    }
    CHECK_EQ_OR_RETURN(this->tensor_transport_token(), tensor_transport_token);
    return Maybe<void>::Ok();
  }

 private:
  Maybe<void> Init(Symbol<one::GlobalTensorMeta> tensor_meta,
    const Optional<Symbol<NdSbp>>& consumer_nd_sbp_constraint,
                   const TransportToken& tensor_transport_token) {
    this->set_synced_tensor_meta_symbol_id(JUST(SyncedSymbolMap<one::GlobalTensorMeta>::FindOrSync(
        tensor_meta, &SyncSymbolGlobalTensorMeta)));
    if (consumer_nd_sbp_constraint.has_value()) {
      const auto& this_rank_constaint = JUST(consumer_nd_sbp_constraint);
      this->set_consumer_nd_sbp_constraint_symbol_id(
        JUST(SyncedSymbolMap<NdSbp>::FindOrSync(
              this_rank_constaint, &SyncSymbolNdSbp)));
    } else {
      this->clear_consumer_nd_sbp_constraint_symbol_id();
    }
    this->set_tensor_transport_token(static_cast<uint64_t>(tensor_transport_token));
    return Maybe<void>::Ok();
  }
  
  FLAT_MSG_DEFINE_OPTIONAL(uint64_t, synced_tensor_meta_symbol_id);
  FLAT_MSG_DEFINE_OPTIONAL(uint64_t, consumer_nd_sbp_constraint_symbol_id);
  FLAT_MSG_DEFINE_OPTIONAL(uint64_t, tensor_transport_token);
FLAT_MSG_END(FlatTensorConsistency);
// clang-format on

CheckConsistencyAsyncTransportCtx::~CheckConsistencyAsyncTransportCtx() {}

Maybe<void> CheckConsistencyAsyncTransportCtx::PrepareSendBufferAndCallback(
    int64_t rank, void** buffer, std::size_t* size, std::function<void()>* Callback) {
  const auto& tensor_consistency = JUST(FlatTensorConsistency::New(
      tensor_meta_, consumer_nd_sbp_constraint_, tensor_transport_token_));
  *buffer = tensor_consistency.get();
  *size = sizeof(FlatTensorConsistency);
  *Callback = [tensor_consistency] {};
  return Maybe<void>::Ok();
}

Maybe<void> CheckConsistencyAsyncTransportCtx::PrepareRecvBufferAndCallback(
    int64_t rank, void** buffer, std::size_t* size, std::function<void()>* Callback) {
  const auto& flat_tensor_consistency = JUST(FlatTensorConsistency::New());
  *buffer = flat_tensor_consistency.get();
  *size = sizeof(FlatTensorConsistency);
  *Callback = [flat_tensor_consistency]() {};
  flat_tensor_consistency_ = flat_tensor_consistency;
  return Maybe<void>::Ok();
}

Maybe<void> CheckConsistencyAsyncTransportCtx::Check() const {
  if (!flat_tensor_consistency_) { return Maybe<void>::Ok(); }
  JUST(flat_tensor_consistency_->Check(tensor_meta_, consumer_nd_sbp_constraint_,
                                       tensor_transport_token_));
  return Maybe<void>::Ok();
}

int64_t* MutThreadLocalTensorMetaCheckDepth() {
  static thread_local int64_t depth = 0;
  return &depth;
}

Maybe<CheckConsistencyAsyncTransportCtx> LaunchTensorMetaConsistencyCheck(
    const one::Tensor& tensor) {
  const auto& rank_group = JUST(RankGroupScope::CurrentRankGroup());
  const auto& transport_token =
      JUST(TransportToken::NewTransportToken(kTransportTokenTypeCheckTensorConsistency));
  const auto& tensor_meta = JUST(tensor.global_tensor_meta());
  const auto& constaint = JUST(tensor.consumer_nd_sbp_constraint());
  const TransportToken& tensor_transport_token = JUST(tensor.transport_token());
  const auto& ctx = std::make_shared<CheckConsistencyAsyncTransportCtx>(
      transport_token, tensor_meta, constaint, tensor_transport_token);
  JUST(TransportUtil::SendToNextRankInRing(rank_group, transport_token, ctx.get()));
  JUST(TransportUtil::ReceiveFromPrevRankInRing(rank_group, transport_token, ctx.get()));
  return ctx;
}

Maybe<void> BusyWaitAndCheck(std::shared_ptr<CheckConsistencyAsyncTransportCtx>& ctx) {
  JUST_MSG(ctx->WaitDone(), kAsymmetricCodeErrorMsg);
  JUST(ctx->Check());
  return Maybe<void>::Ok();
}

Maybe<void> RunCallback(const std::shared_ptr<one::Tensor>& tensor,
                        const std::function<Maybe<void>()>& Callback) {
  return Callback();
}

}  // namespace private_details
}  // namespace oneflow
