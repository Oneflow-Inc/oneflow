#include "oneflow/core/actor/copy_hd_actor.h"
#include "oneflow/core/actor/actor_registry.h"
#include "oneflow/core/register/local_register_warpper.h"

namespace oneflow {

void CopyHdActor::Init(const TaskProto& task_proto, const ThreadCtx& thread_ctx) {
  CopyActor::Init(task_proto, thread_ctx);
  CHECK(thread_ctx.copy_hd_cuda_stream);
  mut_device_ctx().reset(new CudaDeviceCtx(cuda_handle_.cuda_stream(),
                                           cuda_handle_.cublas_handle(),
                                           cuda_handle_.cudnn_handle()));
}

REGISTER_ACTOR(kCopyHdTask, true, CopyHdActor);
REGISTER_ACTOR(kCopyHdTask, false, CopyHdActor);

}  // namespace oneflow
