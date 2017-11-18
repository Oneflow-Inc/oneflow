#include "oneflow/core/actor/loss_record_actor.h"

namespace oneflow {

void LossRecordActor::VirtualActorInit(const TaskProto& proto,
                                       const ThreadCtx& ctx) {
  CHECK(ctx.cpu_stream);
  mut_device_ctx().reset(new CpuDeviceCtx(ctx.cpu_stream));
  OF_SET_MSG_HANDLER(&LossRecordActor::HandlerNormal);
  regst_ = nullptr;
}

int LossRecordActor::HandlerNormal(const ActorMsg& actor_msg) {
  if (actor_msg.msg_type() == ActorMsgType::kCmdMsg) {
    CHECK_EQ(actor_msg.actor_cmd(), ActorCmd::kEORD);
    return 1;
  } else if (actor_msg.msg_type() == ActorMsgType::kRegstMsg) {
    regst_ = actor_msg.regst();
    ActUntilFail();
  } else {
    UNEXPECTED_RUN();
  }
  return 0;
}

void LossRecordActor::Act() {
  AsyncLaunchKernel(GenDefaultKernelCtx(),
                    [&](int64_t regst_desc_id) -> Regst* { return regst_; });
  AsyncSendRegstMsgToProducer(regst_);
  regst_ = nullptr;
}

}  // namespace oneflow
