#include "oneflow/core/actor/model_save_comp_actor.h"

namespace oneflow {

void MdSaveCompActor::VirtualCompActorInit(const TaskProto& task_proto,
                                           const ThreadCtx& thread_ctx) {
  model_regst_desc_id_ = RegstDescId4Name("model");
  CHECK(thread_ctx.cpu_stream);
  mut_device_ctx().reset(new CpuDeviceCtx(thread_ctx.cpu_stream));
  OF_SET_MSG_HANDLER(&MdSaveCompActor::HandlerNormal);
  regst_ = nullptr;
  next_snapshot_id_ = 0;
}

int MdSaveCompActor::HandlerNormal(const ActorMsg& actor_msg) {
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

void MdSaveCompActor::Act() {
  Snapshot* snapshot =
      SnapshotMgr::Singleton()->GetWriteableSnapshot(next_snapshot_id_++);
  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  std::tuple<Snapshot*, int64_t, int64_t, ParallelPolicy> save_ctx =
      std::make_tuple(snapshot, 0, 0, ParallelPolicy::kDataParallel);
  kernel_ctx.other = &save_ctx;
  AsyncLaunchKernel(kernel_ctx, [&](int64_t regst_desc_id) -> Regst* {
    CHECK_EQ(regst_desc_id, model_regst_desc_id_);
    return regst_;
  });
  AsyncSendRegstMsgToProducer(regst_);
  regst_ = nullptr;
}

}  // namespace oneflow
