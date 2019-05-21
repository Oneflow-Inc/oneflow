#include "oneflow/core/actor/boxing_copy_compute_actor.h"

namespace oneflow {

void BoxingCopyCompActor::VirtualCompActorInit(const TaskProto& proto) {
  InputWiseCompActor::Init(proto);
}

void BoxingCopyCompActor::SetKernelCtxOther(void** other) {}

}  // namespace oneflow
