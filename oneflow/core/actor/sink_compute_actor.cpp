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
    UNEXPECTED_RUN();
  }
  return 0;
}

void SinkCompActor::Act() {
  AsyncLaunchKernel(GenSinkKernelCtx(),
                    [&](int64_t regst_desc_id) -> Regst* { return in_regst_; });
  AsyncSendRegstMsgToProducer(in_regst_);
  in_regst_ = nullptr;
}

bool SinkCompActor::IsReadAlwaysUnReadyFromNow() {
  UNEXPECTED_RUN();
  return false;
}

void SinkCompActor::AsyncReturnAllReadableRegst() { UNEXPECTED_RUN(); }

std::list<std::string> SinkCompActor::InputActUidsOfCurAct() const {
  TODO();
  return {""};
}

}  // namespace oneflow
