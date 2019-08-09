#include "oneflow/core/actor/new_actor.h"

namespace oneflow {

namespace actor {

namespace {

void UpdateCtxWithMsg(OpActorCtx* ctx, const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    ctx->UpdateWithRegstMsg(msg);
  } else if (msg.msg_type() == ActorMsgType::kEordMsg) {
    ctx->UpdateWithEordMsg(msg);
  } else if (msg.msg_type() == ActorMsgType::kCmdMsg) {
    ctx->UpdateWithCmdMsg(msg);
  } else {
    LOG(FATAL) << "ActorMsgType error";
  }
}

void ActUntilFail(OpActorCtx* ctx) {
  while (ctx->IsReady()) {
    ctx->Act();
    ctx->HandleRegstMsgAfterAct();
  }
}

}  // namespace

int OpActor::HandlerNormal(OpActor* actor, const ActorMsg& msg) {
  OpActorCtx* ctx = actor->op_actor_ctx();
  UpdateCtxWithMsg(ctx, msg);
  ActUntilFail(ctx);
  if (ctx->NoLongerConsumeRegst()) {
    actor->set_msg_handler(std::bind(&OpActor::HandlerZombie, actor, std::placeholders::_1));
  }
  return 0;
}

int OpActor::HandlerZombie(OpActor* actor, const ActorMsg& msg) {
  OpActorCtx* ctx = actor->op_actor_ctx();
  ctx->UpdateWithProducedRegstMsg(msg);
  if (ctx->NoLongerConsumedByOthers()) {
    actor->set_msg_handler(MsgHandler());
    return 1;
  }
  return 0;
}

}  // namespace actor

}  // namespace oneflow
