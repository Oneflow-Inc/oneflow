#include "oneflow/core/actor/model_save_comp_actor.h"
#include "oneflow/core/actor/actor_registry.h"

namespace oneflow {

// need review
void MdSaveCompActor::Init(const TaskProto& task_proto) {
  CompActor::Init(task_proto);
  model_regst_desc_id_ = RegstDescId4Name("model");
}

int MdSaveCompActor::ProcessMsg(
    const ActorMsg& actor_msg,
    const ThreadContext& thread_ctx) {
  return (this->*cur_msg_handle_)(actor_msg, thread_ctx);
}

int MdSaveCompActor::HandleBeforeInitDeviceCtx(
    const ActorMsg& actor_msg,
    const ThreadContext& thread_ctx) {
  CHECK(actor_msg.actor_cmd() == ActorCmd::kInitDeviceCtx);
  CHECK(thread_ctx.cpu_stream);
  mut_device_ctx().reset(new CpuDeviceCtx(thread_ctx.cpu_stream));
  cur_msg_handle_ = &MdSaveCompActor::HandleSaveModel;
  return 0;
}

int MdSaveCompActor::HandleSaveModel(
    const ActorMsg& actor_msg,
    const ThreadContext& thread_ctx) {
  if (actor_msg.msg_type() == ActorMsgType::kCmdMsg) {
    CHECK(actor_msg.actor_cmd() == ActorCmd::kOneRegstDescDone);
    return 1;
  } else if (actor_msg.msg_type() == ActorMsgType::kRegstMsg) {
    std::shared_ptr<RegstWarpper> regst_warpper = actor_msg.regst_warpper();
    uint64_t model_version_id = regst_warpper->model_version_id();
    int32_t num_of_batches_in_snapshot = 
        JobDesc::Singleton().num_of_batches_in_snapshot();
    CHECK_GT(num_of_batches_in_snapshot, 0);
    if (model_version_id % num_of_batches_in_snapshot == 0) {
      KernelCtx kernel_ctx = GenDefaultKernelCtx();
      std::tuple<uint64_t, uint64_t> save_ctx = std::make_tuple(model_version_id,
                                                                parallel_id());
      kernel_ctx.other = &save_ctx;
      AsyncWardKernel(
          kernel_ctx,
          [&](uint64_t regst_desc_id) -> std::shared_ptr<RegstWarpper> {
        CHECK_EQ(regst_desc_id, model_regst_desc_id_);
        return regst_warpper;
      });
    }
    ActorMsg msg = ActorMsg::BuildRegstMsgToProducer(
        regst_warpper->producer_actor_id(),
        regst_warpper->regst_raw_ptr());
    AsyncDo([msg]() {
      ActorMsgBus::Singleton().SendMsg(msg);
    });
  } else {
    UNEXPECTED_RUN();
  }
  return 0;
}

REGISTER_ACTOR(kMdSaveCompTask, true, MdSaveCompActor);

}  // namespace oneflow
