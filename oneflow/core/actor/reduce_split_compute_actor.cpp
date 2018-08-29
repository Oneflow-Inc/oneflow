#include "oneflow/core/actor/reduce_split_compute_actor.h"

namespace oneflow {

void ReduceSplitCompActor::VirtualCompActorInit(const TaskProto& proto) {
  InputWiseCompActor::Init(proto);
}

void ReduceSplitCompActor::SetKernelCtxOther(void** other) {
  other_val_ = EnableInplace();
  *other = static_cast<void*>(&other_val_);
}

REGISTER_ACTOR(TaskType::kReduceSplit, ReduceSplitCompActor);

}  // namespace oneflow
