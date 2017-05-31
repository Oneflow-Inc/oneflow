#include "actor/copy_hd_actor.h"
#include "actor/actor_registry.h"

namespace oneflow {

void CopyHdActor::Init(const TaskProto& task_proto) {
  TODO();
}

void CopyHdActor::ProcessMsg(const ActorMsg& actor_msg) {
  TODO();
}

REGISTER_ACTOR(TaskType::kCopyHdTask, true, CopyHdActor);
REGISTER_ACTOR(TaskType::kCopyHdTask, false, CopyHdActor);

}  // namespace oneflow
