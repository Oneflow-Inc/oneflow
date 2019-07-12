#include "oneflow/core/persistence/snapshot_manager.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/machine_context.h"

namespace oneflow {

SnapshotMgr::SnapshotMgr(const Plan& plan) {
  if (Global<JobDesc>::Get()->enable_write_snapshot()) {
    model_save_snapshots_path_ = Global<JobDesc>::Get()->MdSaveSnapshotsPath();
    if (Global<MachineCtx>::Get()->IsThisMachineMaster()) {
      SnapshotFS()->CreateDirIfNotExist(model_save_snapshots_path_);
      CHECK(SnapshotFS()->IsDirEmpty(model_save_snapshots_path_))
          << "Error: model save snapshots path directory need to be emtpy!";
    }
  }
  const std::string& load_path = Global<JobDesc>::Get()->MdLoadSnapshotPath();
  if (load_path != "") { readable_snapshot_.reset(new Snapshot(load_path)); }
  total_mbn_num_ = plan.total_mbn_num();
}

Snapshot* SnapshotMgr::GetWriteableSnapshot(int64_t snapshot_id) {
  CHECK(Global<JobDesc>::Get()->enable_write_snapshot());
  std::unique_lock<std::mutex> lck(snapshot_id2writeable_snapshot_mtx_);
  auto it = snapshot_id2writeable_snapshot_.find(snapshot_id);
  if (it == snapshot_id2writeable_snapshot_.end()) {
    std::string snapshot_root_path =
        JoinPath(model_save_snapshots_path_, "snapshot_" + std::to_string(snapshot_id));
    OfCallOnce(snapshot_root_path, SnapshotFS(), &fs::FileSystem::CreateDirIfNotExist);
    std::unique_ptr<Snapshot> ret(new Snapshot(snapshot_root_path));
    auto emplace_ret = snapshot_id2writeable_snapshot_.emplace(snapshot_id, std::move(ret));
    it = emplace_ret.first;
    CHECK(emplace_ret.second);
  }
  return it->second.get();
}

}  // namespace oneflow
