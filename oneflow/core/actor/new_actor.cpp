#include "oneflow/core/actor/new_actor.h"

namespace oneflow {

namespace actor {

namespace {

void UpdateCtxWithMsg(ActorCtx* ctx, const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kCmdMsg) {
    ctx->UpdateWithCmdMsg(msg);
  } else if (msg.msg_type() == ActorMsgType::kEordMsg) {
    ctx->UpdateWithEordMsg(msg);
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    ctx->UpdateWithRegstMsg(msg);
  } else {
    LOG(FATAL) << "ActorMsgType error";
  }
}

void ActUntilFail(ActorCtx* ctx) {
  while(ctx->IsReady4Act()) { ctx->Act(); }
}

}

int OpActor::HandlerNormal(const ActorMsg& msg) {
  ActorCtx* ctx = this->actor_ctx();
  UpdateCtxWithMsg(ctx, msg);
  ActUntilFail(ctx);
  if (ctx->EndOfRead()) {
    this->set_msg_handler(HandlerEord);
  }
}

int OpActor::HandlerZombie(const ActorMsg& msg) {
  ActorCtx* ctx = this->actor_ctx();
  ctx->ProcessMsgFromConsumers();
  if (ctx->RecvAllProducedMsg()) {
    this->set_msg_handler(MsgHandler());
  }
}

}

}
