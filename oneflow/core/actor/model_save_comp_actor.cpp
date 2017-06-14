#include "oneflow/core/actor/model_save_comp_actor.h"
#include "oneflow/core/actor/actor_registry.h"

namespace oneflow {

// need review
void MdSaveCompActor::Init(const TaskProto& task_proto) {
  CompActor::Init(task_proto);
  model_regst_desc_id_ = RegstDescId4Name("model");
}

void MdSaveCompActor::ProcessMsg(const ActorMsg& actor_msg, const ThreadContext&) {
  if (actor_msg.msg_type() == ActorMsgType::kCmdMsg) {
    CHECK(actor_msg.actor_cmd() == ActorCmd::kStop);
    TODO();
  } else if (actor_msg.msg_type() == ActorMsgType::kRegstMsg) {
    auto regst_warpper = actor_msg.regst_warpper();
    AsyncWardKernel(
        [&](uint64_t regst_desc_id) -> std::shared_ptr<RegstWarpper> {
      CHECK_EQ(regst_desc_id, model_regst_desc_id_);
      return regst_warpper;
    });
    ActorMsgBus::Singleton().SendMsg(ActorMsg::BuildMsgForRegstWriter(
        regst_warpper->producer_actor_id(),
        regst_warpper->regst_raw_ptr()));
  } else {
    UNEXPECTED_RUN();
  }
}

REGISTER_ACTOR(kMdSaveCompTask, true, MdSaveCompActor);

}  // namespace oneflow
