#include "oneflow/core/actor/loss_print_compute_actor.h"

namespace oneflow {

void SinkCompActor::VirtualCompActorInit(const TaskProto& proto) {
  in_regst_ = nullptr;
  OF_SET_MSG_HANDLER(&SinkCompActor::HandlerNormal);
  VirtualSinkCompActorInit(proto);
}

int SinkCompActor::HandlerNormal(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kEordMsg) {
    return 1;
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    in_regst_ = msg.regst();
    ActUntilFail();
  } else {
    UNIMPLEMENTED();
  }
  return 0;
}

void SinkCompActor::Act() {
  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  kernel_ctx.other = NewOther();
  AsyncLaunchKernel(kernel_ctx,
                    [&](int64_t regst_desc_id) -> Regst* { return in_regst_; });
  AsyncSendRegstMsgToProducer(in_regst_);
  DeleteOther(kernel_ctx.other);
  in_regst_ = nullptr;
}

bool SinkCompActor::IsReadAlwaysUnReadyFromNow() {
  UNIMPLEMENTED();
  return false;
}

void SinkCompActor::ForEachCurReadableRegst(
    std::function<void(const Regst*)> handler) {
  handler(in_regst_);
}

void SinkCompActor::AsyncReturnAllReadableRegst() { UNIMPLEMENTED(); }

}  // namespace oneflow
