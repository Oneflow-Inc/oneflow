#include "oneflow/core/actor/loss_print_compute_actor.h"

namespace oneflow {

REGISTER_ACTOR(TaskType::kLossPrint, LossPrintCompActor);

}  // namespace oneflow
