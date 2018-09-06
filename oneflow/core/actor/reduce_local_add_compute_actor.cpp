#include "oneflow/core/actor/reduce_local_add_compute_actor.h"

namespace oneflow {

void ReduceLocalAddCompActor::VirtualCompActorInit(const TaskProto& proto) {
  InputWiseCompActor::Init(proto);
}

void ReduceLocalAddCompActor::SetKernelCtxOther(void** other) {
  int64_t in_bn_id = InBnId4RegstDescId(cur_processed_regst_desc_id());
  bool is_inited = EnableInplace() ? true : processed_regst_desc_id_cnt() != 0;
  bool is_inplace_in_blob =
      EnableInplace()
          ? in_bn_id == parallel_ctx()->parallel_id() % parallel_ctx()->device_num_of_each_machine()
          : false;
  bool enable_inplace = EnableInplace();

  other_val_ = std::make_tuple(in_bn_id, is_inited, is_inplace_in_blob, enable_inplace);
  *other = static_cast<void*>(&other_val_);
}

REGISTER_ACTOR(TaskType::kReduceLocalAdd, ReduceLocalAddCompActor);

}  // namespace oneflow
