#include "oneflow/core/actor/reduce_split_compute_actor.h"

namespace oneflow {

void ReduceSplitCompActor::SetKernelCtxOther(void** other) {
  bool is_inplace_in_blob = EnableInplace() ? true : false;
  *other = static_cast<void*>(&is_inplace_in_blob);
}

REGISTER_ACTOR(TaskType::kReduceSplit, ReduceSplitCompActor);

}  // namespace oneflow
