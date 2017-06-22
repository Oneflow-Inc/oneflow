#include "oneflow/core/actor/copy_comm_net_actor.h"
#include "oneflow/core/actor/actor_registry.h"
#include "oneflow/core/register/local_register_warpper.h"

namespace oneflow {

void CopyCommNetActor::Init(const TaskProto& task_proto, const ThreadCtx& thread_ctx) {
  CopyActor::Init(task_proto, thread_ctx);
  CHECK(thread_ctx.cpu_stream);
  mut_device_ctx().reset(new CpuDeviceCtx(thread_ctx.cpu_stream));
}

REGISTER_ACTOR(kCopyCommNetTask, true, CopyCommNetActor);
REGISTER_ACTOR(kCopyCommNetTask, false, CopyCommNetActor);

}  // namespace oneflow
