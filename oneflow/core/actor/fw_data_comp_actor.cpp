#include "oneflow/core/actor/fw_data_comp_actor.h"
#include "oneflow/core/actor/actor_registry.h"

namespace oneflow {

void FwDataCompActor::Init(const TaskProto& task_proto) {
  TODO();
}

void FwDataCompActor::ProcessMsg(const ActorMsg& actor_msg,
                                 const ThreadContext&) {
  TODO();
}

REGISTER_ACTOR(kDataCompTask, true, FwDataCompActor);

}  // namespace oneflow
