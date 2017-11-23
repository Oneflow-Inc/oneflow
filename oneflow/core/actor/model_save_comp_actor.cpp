#include "oneflow/core/actor/model_save_comp_actor.h"

namespace oneflow {

void MdSaveCompActor::VirtualCompActorInit(const TaskProto& task_proto) {
  model_regst_desc_id_ = RegstDescId4Name("model");
  model_regst_ = nullptr;
  next_snapshot_id_ = 0;
  OF_SET_MSG_HANDLER(&MdSaveCompActor::HandlerNormal);
}

int MdSaveCompActor::HandlerNormal(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kEordMsg) {
    return 1;
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    model_regst_ = msg.regst();
    ActUntilFail();
  } else {
    UNEXPECTED_RUN();
  }
  return 0;
}

void MdSaveCompActor::Act() {
  Snapshot* snapshot =
      SnapshotMgr::Singleton()->GetWriteableSnapshot(next_snapshot_id_++);
  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  kernel_ctx.other = snapshot;
  AsyncLaunchKernel(kernel_ctx, [&](int64_t regst_desc_id) -> Regst* {
    CHECK_EQ(regst_desc_id, model_regst_desc_id_);
    return model_regst_;
  });
  AsyncSendRegstMsgToProducer(model_regst_);
  model_regst_ = nullptr;
}

bool MdSaveCompActor::IsReadAlwaysUnReadyFromNow() {
  UNEXPECTED_RUN();
  return false;
}

void MdSaveCompActor::AsyncReturnAllReadableRegst() { UNEXPECTED_RUN(); }

REGISTER_ACTOR(TaskType::kMdSave, MdSaveCompActor);

}  // namespace oneflow
