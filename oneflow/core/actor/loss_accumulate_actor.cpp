#include "oneflow/core/actor/loss_accumulate_actor.h"

namespace oneflow {

REGISTER_ACTOR(TaskType::kLossAcc, LossAccActor);

}  // namespace oneflow
