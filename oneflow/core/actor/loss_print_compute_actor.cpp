#include "oneflow/core/actor/loss_print_compute_actor.h"

namespace oneflow {

void LossPrintCompActor::VirtualCompActorInit(const TaskProto& proto) {
  loss_acc_regst_ = nullptr;
  OF_SET_MSG_HANDLER(&LossPrintCompActor::HandlerNormal);
}

int LossPrintCompActor::HandlerNormal(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kEordMsg) {
    return 1;
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    loss_acc_regst_ = msg.regst();
    ActUntilFail();
  } else {
    UNEXPECTED_RUN();
  }
  return 0;
}

void LossPrintCompActor::Act() {
  AsyncLaunchKernel(
      GenDefaultKernelCtx(),
      [&](int64_t regst_desc_id) -> Regst* { return loss_acc_regst_; });
  AsyncSendRegstMsgToProducer(loss_acc_regst_);
  loss_acc_regst_ = nullptr;
}

bool LossPrintCompActor::IsReadAlwaysUnReadyFromNow() {
  UNEXPECTED_RUN();
  return false;
}

void LossPrintCompActor::AsyncReturnAllReadableRegst() { UNEXPECTED_RUN(); }

REGISTER_ACTOR(TaskType::kLossPrint, LossPrintCompActor);

}  // namespace oneflow
