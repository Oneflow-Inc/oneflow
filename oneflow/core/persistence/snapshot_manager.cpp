#include "oneflow/core/persistence/snapshot_manager.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

SnapshotMgr::SnapshotMgr(const Plan& plan) {
  if (JobDesc::Singleton()->IsTrain()) {
    model_save_snapshots_path_ = JobDesc::Singleton()->MdSaveSnapshotsPath();
    OF_CALL_ONCE(model_save_snapshots_path_,
                 GlobalFS()->MakeEmptyDir(model_save_snapshots_path_));
  }
  const std::string& load_path = JobDesc::Singleton()->MdLoadSnapshotPath();
  if (load_path != "") { readable_snapshot_.reset(new Snapshot(load_path)); }
  total_mbn_num_ = plan.total_mbn_num();
}

Snapshot* SnapshotMgr::GetWriteableSnapshot(int64_t snapshot_id) {
  auto it = snapshot_id2writeable_snapshot_.find(snapshot_id);
  if (it == snapshot_id2writeable_snapshot_.end()) {
    std::string snapshot_root_path = JoinPath(
        model_save_snapshots_path_, "snapshot_" + std::to_string(snapshot_id));
    OF_CALL_ONCE(snapshot_root_path,
                 GlobalFS()->CreateDirIfNotExist(snapshot_root_path));
    std::unique_ptr<Snapshot> ret(new Snapshot(snapshot_root_path));
    auto emplace_ret =
        snapshot_id2writeable_snapshot_.emplace(snapshot_id, std::move(ret));
    it = emplace_ret.first;
    CHECK(emplace_ret.second);
  }
  return it->second.get();
}

}  // namespace oneflow
