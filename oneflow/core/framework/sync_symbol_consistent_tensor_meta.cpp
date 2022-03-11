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
#include "oneflow/core/framework/sync_symbol_consistent_tensor_meta.h"
#include "oneflow/core/framework/sync_symbol_parallel_desc.h"
#include "oneflow/core/framework/sync_symbol_nd_sbp.h"
#include "oneflow/core/framework/rank_group_rpc_util.h"
#include "oneflow/core/framework/tensor_meta.h"
#include "oneflow/core/framework/synced_symbol_map.h"
#include "oneflow/core/common/flat_shape.h"

namespace oneflow {

struct FlatConsistentTensorMeta final {
  static Maybe<FlatConsistentTensorMeta> New(
      uint64_t symbol_id, Symbol<one::ConsistentTensorMeta> consistent_tensor_meta) {
    const auto& meta = std::make_shared<FlatConsistentTensorMeta>();
    JUST(meta->Init(symbol_id, consistent_tensor_meta));
    return meta;
  }

  Maybe<void> Init(uint64_t symbol_id, Symbol<one::ConsistentTensorMeta> consistent_tensor_meta) {
    this->symbol_id = symbol_id;
    JUST(this->shape.Init(consistent_tensor_meta->shape()));
    this->dtype = static_cast<int32_t>(consistent_tensor_meta->dtype());
    this->is_dynamic = consistent_tensor_meta->is_dynamic();
    this->nd_sbp = JUST(
        SyncedSymbolMap<NdSbp>::FindOrSync(consistent_tensor_meta->nd_sbp(), &SyncSymbolNdSbp));
    this->parallel_desc = JUST(SyncedSymbolMap<ParallelDesc>::FindOrSync(
        consistent_tensor_meta->parallel_desc(), &SyncSymbolParallelDesc));
    return Maybe<void>::Ok();
  }

  Maybe<void> Check(uint64_t symbol_id, Symbol<one::ConsistentTensorMeta> consistent_tensor_meta) {
    CHECK_EQ_OR_RETURN(this->symbol_id, symbol_id);
    JUST(this->shape.Check(consistent_tensor_meta->shape()));
    CHECK_EQ_OR_RETURN(static_cast<DataType>(this->dtype), consistent_tensor_meta->dtype());
    CHECK_EQ_OR_RETURN(this->is_dynamic, consistent_tensor_meta->is_dynamic());
    const auto& nd_sbp = JUST(SyncedSymbolMap<NdSbp>::Symbol4SyncedSymbolId(this->nd_sbp));
    CHECK_OR_RETURN(nd_sbp == consistent_tensor_meta->nd_sbp());
    const auto& parallel_desc =
        JUST(SyncedSymbolMap<ParallelDesc>::Symbol4SyncedSymbolId(this->parallel_desc));
    CHECK_OR_RETURN(parallel_desc == consistent_tensor_meta->parallel_desc());
    return Maybe<void>::Ok();
  }

  uint64_t symbol_id;
  FlatShape shape;
  int32_t dtype;
  bool is_dynamic;
  uint64_t nd_sbp;
  uint64_t parallel_desc;
};

Maybe<void> SyncSymbolConsistentTensorMeta(
    uint64_t symbol_id, Symbol<one::ConsistentTensorMeta> consistent_tensor_meta) {
  const auto& transport_token =
      JUST(TransportToken::NewTransportToken(kTransportTokenTypeSyncSymbolConsistentTensorMeta));
  const auto& recv_buffer = std::make_shared<FlatConsistentTensorMeta>();
  NaiveAsyncTransportCtx ctx(
      transport_token,
      [&](void** buffer, std::size_t* size, std::function<void()>* Cb) -> Maybe<void> {
        const auto& send_buffer =
            JUST(FlatConsistentTensorMeta::New(symbol_id, consistent_tensor_meta));
        *buffer = send_buffer.get();
        *size = sizeof(FlatConsistentTensorMeta);
        *Cb = [send_buffer] {};
        return Maybe<void>::Ok();
      },
      [recv_buffer](void** buffer, std::size_t* size, std::function<void()>* Cb) -> Maybe<void> {
        *buffer = recv_buffer.get();
        *size = sizeof(FlatConsistentTensorMeta);
        *Cb = [recv_buffer] {};
        return Maybe<void>::Ok();
      });
  const auto& rank_group = JUST(RankGroupScope::CurrentRankGroup());
  JUST(TransportUtil::SendToNextRankInRing(rank_group, transport_token, &ctx));
  JUST(TransportUtil::ReceiveFromPrevRankInRing(rank_group, transport_token, &ctx));
  JUST(ctx.WaitDone());
  JUST(recv_buffer->Check(symbol_id, consistent_tensor_meta));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
