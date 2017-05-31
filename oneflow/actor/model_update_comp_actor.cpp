#include "actor/model_update_comp_actor.h"
#include "actor/actor_register.h"

namespace oneflow {

void MdUpdtCompActor::Init(const TaskProto& task_proto) {
  TODO();
}

void MdUpdtCompActor::ProcessMsg(const ActorMsg& actor_msg) {
  TODO();
}

REGISTER_ACTOR(TaskType::kMdUpdtCompTask, true, MdUpdtCompActor);
REGISTER_ACTOR(TaskType::kMdUpdtCompTask, false, MdUpdtCompActor);

}  // namespace oneflow
