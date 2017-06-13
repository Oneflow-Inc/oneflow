#include "oneflow/core/actor/copy_hd_actor.h"
#include "oneflow/core/actor/actor_registry.h"
#include "oneflow/core/register/local_register_warpper.h"

namespace oneflow {

void CopyHdActor::ProcessMsg(const ActorMsg& msg,
                             const ThreadContext& thread_ctx) {
  CudaKernelCtx kernel_ctx(thread_ctx.copy_hd_cuda_stream);
  ProcessMsgWithKernelCtx(msg, kernel_ctx);
}

REGISTER_ACTOR(kCopyHdTask, true, CopyHdActor);
REGISTER_ACTOR(kCopyHdTask, false, CopyHdActor);

}  // namespace oneflow
