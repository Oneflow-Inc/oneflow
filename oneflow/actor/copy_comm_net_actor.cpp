#include "actor/copy_comm_net_actor.h"
#include "actor/actor_registry.h"

namespace oneflow {

void CopyCommNetActor::Init(const TaskProto& task_proto) {
  TODO();
}

void CopyCommNetActor::ProcessMsg(const ActorMsg& actor_msg) {
  TODO();
}

REGISTER_ACTOR(TaskType::kCopyCommNetTask, true, CopyCommNetActor);
REGISTER_ACTOR(TaskType::kCopyCommNetTask, false, CopyCommNetActor);

}  // namespace oneflow
