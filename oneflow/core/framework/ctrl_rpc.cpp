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
#include "oneflow/core/framework/ctrl_rpc.h"
#include "oneflow/core/common/flat_shape.h"
#include "oneflow/core/job/rank_group_scope.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"

namespace oneflow {

/*static*/ Maybe<std::map<int64_t, std::shared_ptr<FlatShape>>> CtrlRpc::All2AllSyncShape(
    const std::shared_ptr<const Shape>& shape) {
  const auto& send_buffer = JUST(FlatShape::New(*shape));
  NaiveAsyncRpcCtx send_ctx(
      [send_buffer](void** buffer, std::size_t* size, std::function<void()>* Cb) -> Maybe<void> {
        *buffer = send_buffer.get();
        *size = sizeof(FlatShape);
        *Cb = [send_buffer] {};
        return Maybe<void>::Ok();
      });
  const auto& map = std::make_shared<std::map<int64_t, std::shared_ptr<FlatShape>>>();
  NaiveAsyncRpcCtx recv_ctx([map](int64_t rank, void** buffer, std::size_t* size,
                                  std::function<void()>* Cb) -> Maybe<void> {
    const auto& recv_buffer = std::make_shared<FlatShape>();
    *buffer = recv_buffer.get();
    *size = sizeof(FlatShape);
    *Cb = [recv_buffer] {};
    CHECK_OR_RETURN(map->emplace(rank, recv_buffer).second);
    return Maybe<void>::Ok();
  });
  const auto& rank_group = JUST(RankGroupScope::CurrentRankGroup());
  const auto& rpc_token = JUST(RpcToken::NewCtrlRpcToken(kRankGroupRpcCmdAll2AllSyncShape));
  JUST(RpcUtil::BroadcastToAllOtherRanks(rank_group, rpc_token, &send_ctx));
  JUST(RpcUtil::CollectFromAllOtherRanks(rank_group, rpc_token, &recv_ctx));
  JUST(RpcUtil::WaitUntilDoneOrTimeout(send_ctx, RpcUtil::TimeoutSeconds()));
  JUST(RpcUtil::WaitUntilDoneOrTimeout(recv_ctx, RpcUtil::TimeoutSeconds()));
  CHECK_OR_RETURN(map->emplace(GlobalProcessCtx::Rank(), send_buffer).second);
  return map;
}

}  // namespace oneflow
