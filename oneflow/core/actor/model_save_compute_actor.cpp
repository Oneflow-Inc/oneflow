#include "oneflow/core/actor/model_save_compute_actor.h"
#include "oneflow/core/kernel/model_save_kernel.h"

namespace oneflow {

void MdSaveCompActor::VirtualSinkCompActorInit(const TaskProto& task_proto) {
  next_snapshot_id_ = 0;
}

void* MdSaveCompActor::NewOther() {
  auto tpl = new MdSaveOther;
  std::get<0>(*tpl) = Global<SnapshotMgr>::Get()->GetWriteableSnapshot(next_snapshot_id_++);
  std::get<1>(*tpl) = [this](LbiBlobHandler Handler) {
    ForEachNaiveCurReadable([&](const Regst* regst) {
      for (const auto& pair : regst->lbi2blob()) {
        Handler(pair.first, static_cast<const Blob*>(pair.second.get()));
      }
    });
  };
  return tpl;
}

void MdSaveCompActor::DeleteOther(void* other) {
  auto tpl = static_cast<MdSaveOther*>(other);
  delete tpl;
}

REGISTER_ACTOR(TaskType::kMdSave, MdSaveCompActor);

}  // namespace oneflow
