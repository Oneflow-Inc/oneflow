#include "oneflow/core/actor/model_save_comp_actor.h"
#include "oneflow/core/actor/actor_registry.h"

namespace oneflow {

void MdSaveCompActor::Init(const TaskProto& task_proto,
                           const ThreadCtx& thread_ctx) {
  CompActor::Init(task_proto, thread_ctx);
  model_regst_desc_id_ = RegstDescId4Name("model");
  CHECK(thread_ctx.cpu_stream);
  mut_device_ctx().reset(new CpuDeviceCtx(thread_ctx.cpu_stream));
  OF_SET_MSG_HANDLE(&MdSaveCompActor::HandleNormal);
}

int MdSaveCompActor::HandleNormal(const ActorMsg& actor_msg) {
  if (actor_msg.msg_type() == ActorMsgType::kCmdMsg) {
    CHECK_EQ(actor_msg.actor_cmd(), ActorCmd::kEORD);
    return 1;
  } else if (actor_msg.msg_type() == ActorMsgType::kRegstMsg) {
    std::shared_ptr<RegstWarpper> regst_warpper = actor_msg.regst_warpper();
    int64_t model_version_id = regst_warpper->model_version_id();
    int32_t num_of_batches_in_snapshot =
        JobDesc::Singleton()->num_of_batches_in_snapshot();
    CHECK_GT(num_of_batches_in_snapshot, 0);
    if (model_version_id % num_of_batches_in_snapshot == 0) {
      int64_t snapshot_id = model_version_id / num_of_batches_in_snapshot;
      Snapshot* snapshot =
          SnapshotMgr::Singleton()->GetWriteableSnapshot(snapshot_id);
      KernelCtx kernel_ctx = GenDefaultKernelCtx();
      std::tuple<Snapshot*, int64_t> save_ctx =
          std::make_tuple(snapshot, parallel_id());
      kernel_ctx.other = &save_ctx;
      AsyncLaunchKernel(
          kernel_ctx,
          [&](int64_t regst_desc_id) -> std::shared_ptr<RegstWarpper> {
            CHECK_EQ(regst_desc_id, model_regst_desc_id_);
            return regst_warpper;
          });
    }
    ActorMsg msg = ActorMsg::BuildRegstMsgToProducer(
        regst_warpper->producer_actor_id(), regst_warpper->regst_raw_ptr());
    AsyncDo([msg]() { ActorMsgBus::Singleton()->SendMsg(msg); });
  } else {
    UNEXPECTED_RUN();
  }
  return 0;
}

REGISTER_ACTOR(kMdSaveCompTask, true, MdSaveCompActor);

}  // namespace oneflow
