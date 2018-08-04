#include "oneflow/core/actor/reduce_global_add_compute_actor.h"

namespace oneflow {

void ReduceGlobalAddCompActor::SetKernelCtxOther(void** other) {
  other_val_.first = InBnId4RegstDescId(cur_processed_regst_desc_id());
  other_val_.second = processed_regst_desc_id_cnt() == 0;
  *other = static_cast<void*>(&other_val_);
}

REGISTER_ACTOR(TaskType::kReduceGlobalAdd, ReduceGlobalAddCompActor);

}  // namespace oneflow
