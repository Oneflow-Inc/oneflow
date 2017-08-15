#include "oneflow/core/actor/loss_accumulate_actor.h"
#include "oneflow/core/actor/actor_registry.h"

namespace oneflow {

REGISTER_ACTOR(kLossAccCompTask, true, LossAccActor);

}  // namespace oneflow
