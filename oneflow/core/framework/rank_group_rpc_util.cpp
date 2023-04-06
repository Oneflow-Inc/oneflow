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
#include "oneflow/core/framework/rank_group_rpc_util.h"
#include "oneflow/core/framework/transport_util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/job/rank_group.h"
#include "oneflow/core/job/rank_group_scope.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/thread/thread_global_id.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"

namespace oneflow {

Maybe<NaiveAsyncTransportCtx> CheckTransportToken(Symbol<RankGroup> rank_group) {
  const auto& transport_token =
      JUST(TransportToken::NewTransportToken(kTransportTokenTypeCheckRankGroupConsistency));
  const auto& PrepareBuffer = [](void** buffer, std::size_t* size,
                                 std::function<void()>* Callback) -> Maybe<void> {
    const auto& placeholder = std::make_shared<uint32_t>();
    *buffer = placeholder.get();
    *size = sizeof(uint32_t);
    *Callback = [placeholder]() {};
    return Maybe<void>::Ok();
  };
  const auto& ctx =
      std::make_shared<NaiveAsyncTransportCtx>(transport_token, PrepareBuffer, PrepareBuffer);
  JUST(TransportUtil::SendToNextRankInRing(rank_group, transport_token, ctx.get()));
  JUST(TransportUtil::ReceiveFromPrevRankInRing(rank_group, transport_token, ctx.get()));
  return ctx;
}

}  // namespace oneflow
