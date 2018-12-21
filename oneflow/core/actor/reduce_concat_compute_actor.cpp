#include "oneflow/core/actor/reduce_concat_compute_actor.h"

namespace oneflow {

REGISTER_ACTOR(TaskType::kReduceConcat, ReduceConcatCompActor);

}  // namespace oneflow
