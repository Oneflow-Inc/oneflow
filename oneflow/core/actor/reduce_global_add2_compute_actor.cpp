#include "oneflow/core/actor/reduce_global_add2_compute_actor.h"

namespace oneflow {

void ReduceGlobalAdd2CompActor::VirtualCompActorInit(const TaskProto& proto) {
  InputWiseCompActor::Init(proto);
  for (const auto& pair : exec_kernel_vec().at(0).bn_in_op2regst_desc_id) {
    CHECK(regst_desc_id2bn_in_op_.emplace(pair.second, pair.first).second);
  }
}

void ReduceGlobalAdd2CompActor::SetKernelCtxOther(void** other) {
  bool do_local_reduce_scatter =
      parallel_ctx()->parallel_num() / parallel_ctx()->device_num_of_each_machine() > 1
      && parallel_ctx()->device_num_of_each_machine() > 1;
  int64_t machine_id_if_do_local_reduce =
      do_local_reduce_scatter
          ? (parallel_ctx()->parallel_num() / parallel_ctx()->device_num_of_each_machine())
          : parallel_ctx()->parallel_id();
  const std::string& ibn = regst_desc_id2bn_in_op_.at(cur_processed_regst_desc_id());
  bool is_inited = EnableInplace() ? true : processed_regst_desc_id_cnt() != 0;
  bool is_inplace_in_blob =
      EnableInplace() ? (oneflow_cast<int64_t>(ibn.substr(3)) == machine_id_if_do_local_reduce)
                      : false;

  other_val_ = std::make_tuple(ibn, is_inited, is_inplace_in_blob);
  *other = static_cast<void*>(&other_val_);
}

REGISTER_ACTOR(TaskType::kReduceGlobalAdd2, ReduceGlobalAdd2CompActor);

}  // namespace oneflow
