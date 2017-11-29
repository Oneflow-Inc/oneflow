#include "oneflow/core/actor/print_compute_actor.h"

namespace oneflow {

REGISTER_ACTOR(TaskType::kPrint, PrintCompActor);
}
