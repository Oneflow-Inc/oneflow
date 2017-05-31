#include "actor/boxing_actor.h"
#include "actor/actor_registry.h"

namespace oneflow {

void BoxingActor::Init(const TaskProto& task_proto) {
  TODO();
}

void BoxingActor::ProcessMsg(const ActorMsg& actor_msg) {
  TODO();
}

REGISTER_ACTOR(TaskType::kBoxingTask, true, BoxingActor);
REGISTER_ACTOR(TaskType::kBoxingTask, false, BoxingActor);

}  // namespace oneflow
