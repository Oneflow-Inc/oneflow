#include "oneflow/core/actor/accuracy_accumulate_compute_actor.h"

namespace oneflow {

REGISTER_ACTOR(TaskType::kAccuracyAcc, AccuracyAccCompActor);

}  // namespace oneflow
