#include "oneflow/core/actor/accuracy_print_compute_actor.h"

namespace oneflow {

REGISTER_ACTOR(TaskType::kAccuracyPrint, AccuracyPrintCompActor);

}  // namespace oneflow