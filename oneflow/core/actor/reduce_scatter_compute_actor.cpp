#include "oneflow/core/actor/reduce_scatter_compute_actor.h"

namespace oneflow {

void ReduceScatterCompActor::VirtualCompActorInit(const TaskProto& proto) {
  InputWiseCompActor::Init(proto);
}

void ReduceScatterCompActor::SetKernelCtxOther(void** other) {
  other_val_ = EnableInplace();
  *other = static_cast<void*>(&other_val_);
}

REGISTER_ACTOR(TaskType::kReduceScatter, ReduceScatterCompActor);

}  // namespace oneflow
