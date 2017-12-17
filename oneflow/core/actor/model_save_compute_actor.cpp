#include "oneflow/core/actor/model_save_compute_actor.h"

namespace oneflow {

void MdSaveCompActor::VirtualSinkCompActorInit(const TaskProto& task_proto) {
  next_snapshot_id_ = 0;
}

KernelCtx MdSaveCompActor::GenSinkKernelCtx() {
  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  kernel_ctx.other =
      SnapshotMgr::Singleton()->GetWriteableSnapshot(next_snapshot_id_++);
  return kernel_ctx;
}

REGISTER_ACTOR(TaskType::kMdSave, MdSaveCompActor);

}  // namespace oneflow
