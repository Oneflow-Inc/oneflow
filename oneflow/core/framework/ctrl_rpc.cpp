#include "oneflow/core/framework/ctrl_rpc.h"
#include "oneflow/core/common/flat_shape.h"

namespace oneflow {

/*static*/ Maybe<Hash<int64_t, std::shared_ptr<FlatShape>>> CtrlRpc::All2AllSyncShape(const Shape& shape) {
	const auto& send_buffer = JUST(FlatShape::New(shape));
  NaiveAsyncRpcCtx send_ctx(
			[send_buffer](void** buffer, std::size_t* size, std::function<void()>* Cb) -> Maybe<void> {
        *buffer = send_buffer.get();
        *size = sizeof(FlatShape);
        *Cb = [send_buffer] {};
        return Maybe<void>::Ok();
      });
	const auto& map = std::make_shared<Hash<int64_t, std::shared_ptr<FlatShape>>>();
  NaiveAsyncRpcCtx recv_ctx(
			[map](int64_t rank, void** buffer, std::size_t* size, std::function<void()>* Cb) -> Maybe<void> {
        const auto& recv_buffer = std::make_shared<FlatShape>();
        *buffer = recv_buffer.get();
        *size = sizeof(FlatShape);
        *Cb = [recv_buffer] {};
				CHECK_OR_RETURN(map.emplace(rank, recv_buffer).second);
        return Maybe<void>::Ok();
      });
  const auto& rank_group = JUST(RankGroupScope::CurrentRankGroup());
  const auto& rpc_token =
      JUST(RpcToken::NewCtrlRpcToken(kRankGroupRpcCmdAll2AllSyncShape));
  JUST(RpcUtil::BroadcastToAllOtherRanks(rank_group, rpc_token, &send_ctx));
  JUST(RpcUtil::CollectFromAllOtherRanks(rank_group, rpc_token, &recv_ctx));
  JUST(RpcUtil::WaitUntilDoneOrTimeout(send_ctx, RpcUtil::TimeoutSeconds()));
  JUST(RpcUtil::WaitUntilDoneOrTimeout(recv_ctx, RpcUtil::TimeoutSeconds()));
  return map;
}

}
