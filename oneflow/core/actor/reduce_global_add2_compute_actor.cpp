#include "oneflow/core/actor/reduce_global_add2_compute_actor.h"

namespace oneflow {

void ReduceGlobalAdd2CompActor::VirtualCompActorInit(const TaskProto& proto) {
  InputWiseCompActor::Init(proto);
}

void ReduceGlobalAdd2CompActor::SetKernelCtxOther(void** other) {
  int64_t in_bn_id = InBnId4RegstDescId(cur_processed_regst_desc_id());
  bool do_local_reduce_scatter =
      parallel_ctx()->parallel_num() / parallel_ctx()->device_num_of_each_machine() > 1
      && parallel_ctx()->device_num_of_each_machine() > 1;
  int64_t machine_id_if_do_local_reduce =
      do_local_reduce_scatter
          ? (parallel_ctx()->parallel_num() / parallel_ctx()->device_num_of_each_machine())
          : parallel_ctx()->parallel_id();
  bool is_inited = EnableInplace() ? true : processed_regst_desc_id_cnt() != 0;
  bool is_inplace_in_blob = EnableInplace() ? (in_bn_id == machine_id_if_do_local_reduce) : false;

  other_val_ = std::make_tuple(in_bn_id, is_inited, is_inplace_in_blob);
  *other = static_cast<void*>(&other_val_);
}

REGISTER_ACTOR(TaskType::kReduceGlobalAdd2, ReduceGlobalAdd2CompActor);

}  // namespace oneflow
