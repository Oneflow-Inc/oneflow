#include "oneflow/core/actor/model_init_compute_actor.h"
#include "oneflow/core/kernel/model_init_kernel.h"

namespace oneflow {

void MdInitCompActor::VirtualCompActorInit(const TaskProto& task_proto) {
  OF_SET_MSG_HANDLER(&MdInitCompActor::HandlerNormal);
}

void MdInitCompActor::Act() {
  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  AsyncLaunchKernel(kernel_ctx);
}

REGISTER_ACTOR(TaskType::kMdInit, MdInitCompActor);

}  // namespace oneflow
