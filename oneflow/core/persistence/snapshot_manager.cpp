#include "oneflow/core/persistence/snapshot_manager.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

void SnapshotMgr::Init() {
  LOG(INFO) << "SnapshotMgr Init";
  model_save_snapshots_path_ = JobDesc::Singleton()->md_save_snapshots_path();
  OF_ONCE_GUARD(model_save_snapshots_path_,
                GlobalFS()->CreateDirIfNotExist(model_save_snapshots_path_));
  std::vector<std::string> result;
  FS_CHECK_OK(GlobalFS()->GetChildren(model_save_snapshots_path_, &result));
  CHECK_EQ(result.size(), 0);
  const std::string& load_path = JobDesc::Singleton()->md_load_snapshot_path();
  if (load_path != "") {
    readable_snapshot_ptr_.reset(new Snapshot(load_path));
  }
}

Snapshot* SnapshotMgr::GetWriteableSnapshot(int64_t snapshot_id) {
  auto it = snapshot_id2writeable_snapshot_.find(snapshot_id);
  if (it == snapshot_id2writeable_snapshot_.end()) {
    std::string snapshot_root_path = JoinPath(
        model_save_snapshots_path_, "snapshot_" + std::to_string(snapshot_id));
    GlobalFS()->CreateDirIfNotExist(snapshot_root_path);
    std::unique_ptr<Snapshot> ret(new Snapshot(snapshot_root_path));
    auto emplace_ret =
        snapshot_id2writeable_snapshot_.emplace(snapshot_id, std::move(ret));
    it = emplace_ret.first;
    CHECK(emplace_ret.second);
  }
  return it->second.get();
}

}  // namespace oneflow
