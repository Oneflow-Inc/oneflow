#include "oneflow/core/actor/model_load_compute_actor.h"
#include "oneflow/core/kernel/model_load_kernel.h"

namespace oneflow {

void MdLoadCompActor::VirtualCompActorInit(const TaskProto& task_proto) {
  OF_SET_MSG_HANDLER(&MdLoadCompActor::HandlerNormal);
}

void MdLoadCompActor::Act() {
  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  AsyncLaunchKernel(kernel_ctx);
}

REGISTER_ACTOR(TaskType::kMdLoad, MdLoadCompActor);

}  // namespace oneflow
