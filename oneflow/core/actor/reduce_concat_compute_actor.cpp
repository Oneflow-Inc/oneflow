#include "oneflow/core/actor/reduce_concat_compute_actor.h"

namespace oneflow {

void ReduceConcatCompActor::VirtualCompActorInit(const TaskProto& proto) {
  InputWiseCompActor::Init(proto);
}

void ReduceConcatCompActor::SetKernelCtxOther(void** other) {
  int64_t in_bn_id = InBnId4RegstDescId(cur_processed_regst_desc_id());
  other_val_ = std::make_pair(in_bn_id, EnableInplace());
  *other = static_cast<void*>(&other_val_);
}

REGISTER_ACTOR(TaskType::kReduceConcat, ReduceConcatCompActor);

}  // namespace oneflow
