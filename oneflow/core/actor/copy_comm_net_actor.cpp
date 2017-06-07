#include "oneflow/core/actor/copy_comm_net_actor.h"
#include "oneflow/core/actor/actor_registry.h"

namespace oneflow {

void CopyCommNetActor::Init(const TaskProto& task_proto) {
  TODO();
}

void CopyCommNetActor::ProcessMsg(const ActorMsg& actor_msg,
                                  const ThreadContext&) {
  TODO();
}

REGISTER_ACTOR(kCopyCommNetTask, true, CopyCommNetActor);
REGISTER_ACTOR(kCopyCommNetTask, false, CopyCommNetActor);

}  // namespace oneflow
