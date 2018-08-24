#include "oneflow/core/actor/reduce_concat_compute_actor.h"

namespace oneflow {

void ReduceConcatCompActor::SetKernelCtxOther(void** other) { TODO(); }

REGISTER_ACTOR(TaskType::kReduceConcat, ReduceConcatCompActor);

}  // namespace oneflow
