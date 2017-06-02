#include "oneflow/actor/copy_hd_actor.h"
#include "oneflow/actor/actor_registry.h"

namespace oneflow {

void CopyHdActor::Init(const TaskProto& task_proto) {
  TODO();
}

void CopyHdActor::ProcessMsg(const ActorMsg& actor_msg) {
  TODO();
}

REGISTER_ACTOR(kCopyHdTask, true, CopyHdActor);
REGISTER_ACTOR(kCopyHdTask, false, CopyHdActor);

}  // namespace oneflow
