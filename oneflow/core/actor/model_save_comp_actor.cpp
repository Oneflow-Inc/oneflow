#include "oneflow/core/actor/model_save_comp_actor.h"
#include "oneflow/core/actor/actor_registry.h"

namespace oneflow {

void MdSaveCompActor::Init(const TaskProto& task_proto,
                           const ThreadCtx& thread_ctx) {
  CompActor::Init(task_proto, thread_ctx);
  model_regst_desc_id_ = RegstDescId4Name("model");
  CHECK(thread_ctx.cpu_stream);
  mut_device_ctx().reset(new CpuDeviceCtx(thread_ctx.cpu_stream));
  OF_SET_MSG_HANDLER(&MdSaveCompActor::HandlerNormal);
  next_snapshot_id_ = 0;
}

int MdSaveCompActor::HandlerNormal(const ActorMsg& actor_msg) {
  if (actor_msg.msg_type() == ActorMsgType::kCmdMsg) {
    CHECK_EQ(actor_msg.actor_cmd(), ActorCmd::kEORD);
    return 1;
  } else if (actor_msg.msg_type() == ActorMsgType::kRegstMsg) {
    regst_wrapper_ = actor_msg.regst_wrapper();
    VLOG(4) << "model save actor " << actor_id() << " "
            << "receive readable regst " << regst_wrapper_->regst_raw_ptr()
            << ", regst_desc_id:" << regst_wrapper_->regst_desc_id();
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
      std::make_tuple(snapshot, parallel_id(), parallel_num(),
                      parallel_policy());
  kernel_ctx.other = &save_ctx;
  AsyncLaunchKernel(
      kernel_ctx, [&](int64_t regst_desc_id) -> std::shared_ptr<RegstWrapper> {
        CHECK_EQ(regst_desc_id, model_regst_desc_id_);
        return regst_wrapper_;
      });
  AsyncSendRegstMsgToProducer(regst_wrapper_);
  regst_wrapper_.reset();
}

REGISTER_ACTOR(kMdSaveCompTask, true, MdSaveCompActor);

}  // namespace oneflow
