#include "oneflow/core/actor/sink_compute_actor.h"

namespace oneflow {

void SinkCompActor::VirtualCompActorInit(const TaskProto& proto) {
  OF_SET_MSG_HANDLER(&SinkCompActor::HandlerNormal);
  VirtualSinkCompActorInit(proto);
}

void SinkCompActor::Act() {
  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  kernel_ctx.other = NewOther();
  AsyncLaunchKernel(kernel_ctx);
  DeleteOther(kernel_ctx.other);
}

}  // namespace oneflow
