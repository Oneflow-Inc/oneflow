#include "oneflow/core/actor/slice_boxing_actor.h"

namespace oneflow {

void SliceBoxingActor::SetKernelCtxOther(void** other) {
  other_val_.first = InBnId4RegstDescId(cur_processed_regst_desc_id());
  other_val_.second = processed_regst_desc_id_cnt();
  *other = static_cast<void*>(&other_val_);
}

REGISTER_ACTOR(TaskType::kSliceBoxing, SliceBoxingActor);

}  // namespace oneflow
