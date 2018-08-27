#include "oneflow/core/actor/reduce_global_add_compute_actor.h"

namespace oneflow {

void ReduceGlobalAddCompActor::SetKernelCtxOther(void** other) {
  int64_t in_bn_id = InBnId4RegstDescId(cur_processed_regst_desc_id());
  bool is_inited = EnableInplace() ? true : processed_regst_desc_id_cnt() != 0;
  bool is_inplace_in_blob =
      EnableInplace()
          ? (parallel_ctx()->parallel_id() / parallel_ctx()->device_num_of_each_machine()
             == in_bn_id)
          : false;

  other_val_ = std::make_tuple(in_bn_id, is_inited, is_inplace_in_blob);
  *other = static_cast<void*>(&other_val_);
}

REGISTER_ACTOR(TaskType::kReduceGlobalAdd, ReduceGlobalAddCompActor);

}  // namespace oneflow
