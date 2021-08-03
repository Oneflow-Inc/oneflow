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
#include "oneflow/core/framework/sync_symbol_consistent_tensor_meta.h"
#include "oneflow/core/framework/sync_symbol_parallel_distribution.h"
#include "oneflow/core/framework/synced_symbol_map.h"
#include "oneflow/core/framework/rank_group_rpc_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/common/flat_shape.h"
#include "oneflow/core/common/shape_vec.h"
#include "oneflow/core/object_msg/flat_msg.h"
#include "oneflow/core/job/rank_group.h"
#include "oneflow/core/job/rank_group_scope.h"

namespace oneflow {

// clang-format off
FLAT_MSG_BEGIN(FlatTensorConsistency);
  OF_PUBLIC static Maybe<FlatTensorConsistency> New() {
    const auto& consistency = std::make_shared<FlatTensorConsistency>();
    consistency->clear();
    return consistency;
  }
  OF_PUBLIC static Maybe<FlatTensorConsistency> New(
      Symbol<one::ConsistentTensorMeta> tensor_meta,
      const Optional<Symbol<cfg::ParallelDistribution>> consumer_parallel_distribution_constraint,
                                          const RpcToken& tensor_rpc_token) {
    const auto& consistency = std::make_shared<FlatTensorConsistency>();
    consistency->clear();
    JUST(consistency->Init(tensor_meta, consumer_parallel_distribution_constraint, tensor_rpc_token));
    return consistency;
  }

  OF_PUBLIC Maybe<void> Check(Symbol<one::ConsistentTensorMeta> tensor_meta,
    const Optional<Symbol<cfg::ParallelDistribution>> consumer_parallel_distribution_constraint,
                    const RpcToken& tensor_rpc_token) {
    const auto& this_synced_tensor_meta =
        JUST(SyncedSymbolMap<one::ConsistentTensorMeta>::Symbol4SyncedSymbolId(
            this->synced_tensor_meta_symbol_id()));
    CHECK_OR_RETURN(this_synced_tensor_meta == tensor_meta);
    CHECK_EQ_OR_RETURN(consumer_parallel_distribution_constraint.has_value(),
                       this->has_consumer_parallel_distribution_constraint_symbol_id());
    if (this->has_consumer_parallel_distribution_constraint_symbol_id()) {
      const auto& that_rank_constaint =
          JUST(SyncedSymbolMap<one::ConsistentTensorMeta>::Symbol4SyncedSymbolId(
            this->consumer_parallel_distribution_constraint_symbol_id()));
      const auto& this_rank_constaint = JUST(consumer_parallel_distribution_constraint.value());
      CHECK_OR_RETURN(this_rank_constaint == that_rank_constaint);
    }
    CHECK_EQ_OR_RETURN(this->tensor_rpc_token(), tensor_rpc_token);
    return Maybe<void>::Ok();
  }

  OF_PRIVATE Maybe<void> Init(Symbol<one::ConsistentTensorMeta> tensor_meta,
    const Optional<Symbol<cfg::ParallelDistribution>> consumer_parallel_distribution_constraint,
                   const RpcToken& tensor_rpc_token) {
    this->set_synced_tensor_meta_symbol_id(JUST(SyncedSymbolMap<one::ConsistentTensorMeta>::FindOrSync(
        tensor_meta, &SyncSymbolConsistentTensorMeta)));
    if (consumer_parallel_distribution_constraint.has_value()) {
      const auto& this_rank_constaint = JUST(consumer_parallel_distribution_constraint.value());
      this->set_consumer_parallel_distribution_constraint_symbol_id(
        JUST(SyncedSymbolMap<cfg::ParallelDistribution>::FindOrSync(
              this_rank_constaint, &SyncSymbolParallelDistribution)));
    } else {
      this->clear_consumer_parallel_distribution_constraint_symbol_id();
    }
    this->set_tensor_rpc_token(static_cast<uint64_t>(tensor_rpc_token));
    return Maybe<void>::Ok();
  }
  
  FLAT_MSG_DEFINE_OPTIONAL(uint64_t, synced_tensor_meta_symbol_id);
  FLAT_MSG_DEFINE_OPTIONAL(uint64_t, consumer_parallel_distribution_constraint_symbol_id);
  FLAT_MSG_DEFINE_OPTIONAL(uint64_t, tensor_rpc_token);
FLAT_MSG_END(FlatTensorConsistency);
// clang-format off

CheckConsistencyAsyncRpcCtx::~CheckConsistencyAsyncRpcCtx() {}

Maybe<void> CheckConsistencyAsyncRpcCtx::PrepareSendBufferAndCallback(
    int64_t rank, void** buffer, std::size_t* size, std::function<void()>* Callback) {
  const auto& tensor_consistency =
      JUST(FlatTensorConsistency::New(tensor_meta_, consumer_parallel_distribution_constraint_, tensor_rpc_token_));
  *buffer = tensor_consistency.get();
  *size = sizeof(FlatTensorConsistency);
  *Callback = [tensor_consistency] {};
  return Maybe<void>::Ok();
}

Maybe<void> CheckConsistencyAsyncRpcCtx::PrepareRecvBufferAndCallback(
    int64_t rank, void** buffer, std::size_t* size, std::function<void()>* Callback) {
  const auto& flat_tensor_consistency = JUST(FlatTensorConsistency::New());
  *buffer = flat_tensor_consistency.get();
  *size = sizeof(FlatTensorConsistency);
  *Callback = [flat_tensor_consistency]() {};
  flat_tensor_consistency_ = flat_tensor_consistency;
  return Maybe<void>::Ok();
}

Maybe<void> CheckConsistencyAsyncRpcCtx::Check() const {
  if (!flat_tensor_consistency_) { return Maybe<void>::Ok(); }
  JUST(flat_tensor_consistency_->Check(
      tensor_meta_, consumer_parallel_distribution_constraint_, tensor_rpc_token_));
  return Maybe<void>::Ok();
}

Maybe<CheckConsistencyAsyncRpcCtx> LaunchTensorMetaConsistencyCheck(const one::Tensor& tensor) {
  const auto& rank_group = JUST(RankGroupScope::CurrentRankGroup());
  const auto& rpc_token =
      JUST(RpcToken::AcquireCtrlRpcToken(kRankGroupRpcCmdCheckTensorConsistency));
  const auto& tensor_meta = JUST(tensor.consistent_tensor_meta());
  const auto& constaint = JUST(tensor.consumer_parallel_distribution_constraint());
  const RpcToken& tensor_rpc_token = JUST(tensor.rpc_token());
  const auto& ctx = std::make_shared<CheckConsistencyAsyncRpcCtx>(
    rpc_token, tensor_meta, constaint, tensor_rpc_token);
  JUST(RpcUtil::SendToNextRankInRing(rank_group, rpc_token, ctx.get()));
  JUST(RpcUtil::ReceiveFromPrevRankInRing(rank_group, rpc_token, ctx.get()));
  return ctx;
}

}  // namespace oneflow
