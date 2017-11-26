#include "oneflow/core/actor/loss_record_compute_actor.h"

namespace oneflow {

void LossRecordCompActor::VirtualCompActorInit(const TaskProto& proto) {
  loss_acc_regst_ = nullptr;
  OF_SET_MSG_HANDLER(&LossRecordCompActor::HandlerNormal);
}

int LossRecordCompActor::HandlerNormal(const ActorMsg& msg) {
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

void LossRecordCompActor::Act() {
  AsyncLaunchKernel(
      GenDefaultKernelCtx(),
      [&](int64_t regst_desc_id) -> Regst* { return loss_acc_regst_; });
  AsyncSendRegstMsgToProducer(loss_acc_regst_);
  loss_acc_regst_ = nullptr;
}

bool LossRecordCompActor::IsReadAlwaysUnReadyFromNow() {
  UNEXPECTED_RUN();
  return false;
}

void LossRecordCompActor::AsyncReturnAllReadableRegst() { UNEXPECTED_RUN(); }

REGISTER_ACTOR(TaskType::kLossRecord, LossRecordCompActor);

}  // namespace oneflow
