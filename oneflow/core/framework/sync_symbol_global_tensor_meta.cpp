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
#include "oneflow/core/framework/sync_symbol_global_tensor_meta.h"
#include "oneflow/core/framework/sync_symbol_parallel_desc.h"
#include "oneflow/core/framework/sync_symbol_nd_sbp.h"
#include "oneflow/core/framework/rank_group_rpc_util.h"
#include "oneflow/core/common/tensor_meta.h"
#include "oneflow/core/framework/synced_symbol_map.h"
#include "oneflow/core/common/flat_shape.h"

namespace oneflow {

struct FlatGlobalTensorMeta final {
  static Maybe<FlatGlobalTensorMeta> New(uint64_t symbol_id,
                                         Symbol<one::GlobalTensorMeta> global_tensor_meta) {
    const auto& meta = std::make_shared<FlatGlobalTensorMeta>();
    JUST(meta->Init(symbol_id, global_tensor_meta));
    return meta;
  }

  Maybe<void> Init(uint64_t symbol_id, Symbol<one::GlobalTensorMeta> global_tensor_meta) {
    this->symbol_id = symbol_id;
    JUST(this->shape.Init(global_tensor_meta->shape()));
    this->dtype = static_cast<int32_t>(global_tensor_meta->dtype());
    this->is_dynamic = global_tensor_meta->is_dynamic();
    this->nd_sbp =
        JUST(SyncedSymbolMap<NdSbp>::FindOrSync(global_tensor_meta->nd_sbp(), &SyncSymbolNdSbp));
    this->parallel_desc = JUST(SyncedSymbolMap<ParallelDesc>::FindOrSync(
        global_tensor_meta->parallel_desc(), &SyncSymbolParallelDesc));
    return Maybe<void>::Ok();
  }

  Maybe<void> Check(uint64_t symbol_id, Symbol<one::GlobalTensorMeta> global_tensor_meta) {
    CHECK_EQ_OR_RETURN(this->symbol_id, symbol_id);
    JUST(this->shape.Check(global_tensor_meta->shape()));
    CHECK_EQ_OR_RETURN(static_cast<DataType>(this->dtype), global_tensor_meta->dtype());  // NOLINT
    CHECK_EQ_OR_RETURN(this->is_dynamic, global_tensor_meta->is_dynamic());               // NOLINT
    const auto& nd_sbp = JUST(SyncedSymbolMap<NdSbp>::Symbol4SyncedSymbolId(this->nd_sbp));
    CHECK_OR_RETURN(nd_sbp == global_tensor_meta->nd_sbp());  // NOLINT
    const auto& parallel_desc =
        JUST(SyncedSymbolMap<ParallelDesc>::Symbol4SyncedSymbolId(this->parallel_desc));
    CHECK_OR_RETURN(parallel_desc == global_tensor_meta->parallel_desc());  // NOLINT
    return Maybe<void>::Ok();
  }

  uint64_t symbol_id;
  FlatShape shape;
  int32_t dtype;
  bool is_dynamic;
  uint64_t nd_sbp;
  uint64_t parallel_desc;
};

Maybe<void> SyncSymbolGlobalTensorMeta(uint64_t symbol_id,
                                       Symbol<one::GlobalTensorMeta> global_tensor_meta) {
  const auto& transport_token =
      JUST(TransportToken::NewTransportToken(kTransportTokenTypeSyncSymbolGlobalTensorMeta));
  const auto& recv_buffer = std::make_shared<FlatGlobalTensorMeta>();
  NaiveAsyncTransportCtx ctx(
      transport_token,
      [&](void** buffer, std::size_t* size, std::function<void()>* Cb) -> Maybe<void> {
        const auto& send_buffer = JUST(FlatGlobalTensorMeta::New(symbol_id, global_tensor_meta));
        *buffer = send_buffer.get();
        *size = sizeof(FlatGlobalTensorMeta);
        *Cb = [send_buffer] {};
        return Maybe<void>::Ok();
      },
      [recv_buffer](void** buffer, std::size_t* size, std::function<void()>* Cb) -> Maybe<void> {
        *buffer = recv_buffer.get();
        *size = sizeof(FlatGlobalTensorMeta);
        *Cb = [recv_buffer] {};
        return Maybe<void>::Ok();
      });
  const auto& rank_group = JUST(RankGroupScope::CurrentRankGroup());
  JUST(TransportUtil::SendToNextRankInRing(rank_group, transport_token, &ctx));
  JUST(TransportUtil::ReceiveFromPrevRankInRing(rank_group, transport_token, &ctx));
  JUST(ctx.WaitDone());
  JUST(recv_buffer->Check(symbol_id, global_tensor_meta));
  return Maybe<void>::Ok();
}

}  // namespace oneflow
