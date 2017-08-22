#include "oneflow/core/actor/model_diff_accumulate_actor.h"
#include "oneflow/core/actor/actor_registry.h"

namespace oneflow {

REGISTER_ACTOR(kMdDiffAccCompTask, true, MdDiffAccActor);

}  // namespace oneflow
