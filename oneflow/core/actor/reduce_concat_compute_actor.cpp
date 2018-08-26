#include "oneflow/core/actor/reduce_concat_compute_actor.h"

namespace oneflow {

void ReduceConcatCompActor::SetKernelCtxOther(void** other) {
  int64_t in_bn_id = InBnId4RegstDescId(cur_processed_regst_desc_id());
  bool is_inplace_in_blob = EnableInplace() ? true : false;

  other_val_ = std::make_pair(in_bn_id, is_inplace_in_blob);
  *other = static_cast<void*>(&other_val_);
}

REGISTER_ACTOR(TaskType::kReduceConcat, ReduceConcatCompActor);

}  // namespace oneflow
