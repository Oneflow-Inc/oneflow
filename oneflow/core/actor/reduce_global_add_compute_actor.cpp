#include "oneflow/core/actor/reduce_global_add_compute_actor.h"

namespace oneflow {

void ReduceGlobalAddCompActor::SetKernelCtxOther(void** other) {
  int64_t in_bn_id = InBnId4RegstDescId(cur_processed_regst_desc_id());
  bool is_inited = processed_regst_desc_id_cnt() != 0;
  bool is_inplace_in_bn_id = in_bn_id == parallel_ctx()->parallel_id();
  *other = static_cast<void*>(&other_val_);
}

REGISTER_ACTOR(TaskType::kReduceGlobalAdd, ReduceGlobalAddCompActor);

}  // namespace oneflow
