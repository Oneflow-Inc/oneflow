#include "oneflow/actor/model_update_comp_actor.h"
#include "oneflow/actor/actor_registry.h"

namespace oneflow {

void MdUpdtCompActor::Init(const TaskProto& task_proto) {
  TODO();
}

void MdUpdtCompActor::ProcessMsg(const ActorMsg& actor_msg) {
  TODO();
}

REGISTER_ACTOR(kMdUpdtCompTask, true, MdUpdtCompActor);

}  // namespace oneflow
