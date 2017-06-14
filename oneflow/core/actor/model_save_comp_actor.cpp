#include "oneflow/core/actor/model_save_comp_actor.h"
#include "oneflow/core/actor/actor_registry.h"

namespace oneflow {

// need review
void MdSaveCompActor::Init(const TaskProto& task_proto) {
  CompActor::Init(task_proto);
  model_regst_desc_id_ = RegstDescId4Name("model");
}

int MdSaveCompActor::ProcessMsg(const ActorMsg& actor_msg, const ThreadContext&) {
  if (actor_msg.msg_type() == ActorMsgType::kCmdMsg) {
    CHECK(actor_msg.actor_cmd() == ActorCmd::kOneRegstDescDone);
    return 1;
  } else if (actor_msg.msg_type() == ActorMsgType::kRegstMsg) {
    auto regst_warpper = actor_msg.regst_warpper();
    auto kernel_ctx = GenDefaultKernelCtx();
    kernel_ctx.other = &std::make_tuple(regst_warpper->model_version_id(),
                                        parallel_id());
    AsyncWardKernel(
        kernel_ctx,
        [&](uint64_t regst_desc_id) -> std::shared_ptr<RegstWarpper> {
      CHECK_EQ(regst_desc_id, model_regst_desc_id_);
      return regst_warpper;
    });
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
