#include "oneflow/core/actor/bp_data_comp_actor.h"
#include "oneflow/core/actor/actor_registry.h"

namespace oneflow {

void BpDataCompActor::Init(const TaskProto& task_proto) {
  TODO();
}

void BpDataCompActor::ProcessMsg(const ActorMsg& actor_msg) {
  TODO();
}

REGISTER_ACTOR(kDataCompTask, false, BpDataCompActor);

}  // namespace oneflow
