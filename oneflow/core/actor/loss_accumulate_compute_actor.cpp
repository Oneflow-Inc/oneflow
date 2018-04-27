#include "oneflow/core/actor/loss_accumulate_compute_actor.h"

namespace oneflow {

REGISTER_ACTOR(TaskType::kLossAcc, LossAccCompActor);

}  // namespace oneflow
