#include "oneflow/core/actor/reduce_concat_compute_actor.h"

namespace oneflow {

void ReduceConcatCompActor::VirtualCompActorInit(const TaskProto& proto) {
  InputWiseCompActor::Init(proto);
  for (const auto& pair : exec_kernel_vec().at(0).bn_in_op2regst_desc_id) {
    CHECK(regst_desc_id2bn_in_op_.emplace(pair.second, pair.first).second);
  }
}

void ReduceConcatCompActor::SetKernelCtxOther(void** other) {
  int64_t in_bn_id = InBnId4RegstDescId(cur_processed_regst_desc_id());
  bool is_inplace_in_blob = EnableInplace() ? true : false;

  other_val_ = std::make_pair(in_bn_id, is_inplace_in_blob);
  *other = static_cast<void*>(&other_val_);
}

REGISTER_ACTOR(TaskType::kReduceConcat, ReduceConcatCompActor);

}  // namespace oneflow
