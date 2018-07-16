#include "oneflow/core/actor/reduce_gather_compute_actor.h"

namespace oneflow {

void ReduceGatherCompActor::SetKernelCtxOther(void** other) {
  other_val_ = InBnId4RegstDescId(cur_processed_regst_desc_id());
  *other = static_cast<void*>(&other_val_);
}

REGISTER_ACTOR(TaskType::kReduceGather, ReduceGatherCompActor);

}  // namespace oneflow
