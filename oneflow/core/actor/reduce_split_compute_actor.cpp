#include "oneflow/core/actor/reduce_split_compute_actor.h"

namespace oneflow {

REGISTER_ACTOR(TaskType::kReduceSplit, ReduceSplitCompActor);

}  // namespace oneflow
