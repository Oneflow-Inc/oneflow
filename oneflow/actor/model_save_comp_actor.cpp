#include "oneflow/actor/model_save_comp_actor.h"
#include "oneflow/actor/actor_registry.h"

namespace oneflow {

void MdSaveCompActor::Init(const TaskProto& task_proto) {
  TODO();
}

void MdSaveCompActor::ProcessMsg(const ActorMsg& actor_msg) {
  TODO();
}

REGISTER_ACTOR(kMdSaveCompTask, true, MdSaveCompActor);

}  // namespace oneflow
