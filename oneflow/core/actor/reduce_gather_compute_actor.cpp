#include "oneflow/core/actor/reduce_gather_compute_actor.h"

namespace oneflow {

void ReduceGatherCompActor::SetKernelCtxOther(void** other) {
  int64_t in_bn_id = InBnId4RegstDescId(cur_processed_regst_desc_id());
  bool is_first_in_blob = processed_regst_desc_id_cnt() == 0;
  other_val_ = std::make_tuple(in_bn_id, EnableInplace(), is_first_in_blob);
  *other = static_cast<void*>(&other_val_);
}

REGISTER_ACTOR(TaskType::kReduceGather, ReduceGatherCompActor);

}  // namespace oneflow
