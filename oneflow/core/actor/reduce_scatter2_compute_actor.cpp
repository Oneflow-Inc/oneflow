#include "oneflow/core/actor/reduce_scatter2_compute_actor.h"

namespace oneflow {

void ReduceScatter2CompActor::VirtualCompActorInit(const TaskProto& proto) {
  InputWiseCompActor::Init(proto);
}

void ReduceScatter2CompActor::SetKernelCtxOther(void** other) {
  other_val_ = EnableInplace();
  *other = static_cast<void*>(&other_val_);
}

REGISTER_ACTOR(TaskType::kReduceScatter2, ReduceScatter2CompActor);

}  // namespace oneflow
