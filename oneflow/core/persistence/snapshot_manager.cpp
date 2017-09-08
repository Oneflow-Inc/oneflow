#include "oneflow/core/persistence/snapshot_manager.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/job/job_desc.h"

namespace oneflow {

void SnapshotMgr::Init(const Plan& plan) {
  LOG(INFO) << "SnapshotMgr Init";
  model_save_snapshots_path_ = JobDesc::Singleton()->md_save_snapshots_path();
  OF_ONCE_GUARD(model_save_snapshots_path_,
                GlobalFS()->CreateDirIfNotExist(model_save_snapshots_path_);
                CHECK(GlobalFS()->IsDirEmpty(model_save_snapshots_path_)););
  const std::string& load_path = JobDesc::Singleton()->md_load_snapshot_path();
  if (load_path != "") {
    readable_snapshot_ptr_.reset(new Snapshot(load_path));
  }
  HashSet<std::string> model_blob_set;
  for (const OperatorProto& op_proto : plan.op()) {
    if (op_proto.op_conf().has_model_save_conf()) {
      for (const std::string& lbn :
           op_proto.op_conf().model_save_conf().lbns()) {
        model_blob_set.insert(lbn);
      }
    }
  }
  num_of_model_blobs_ = model_blob_set.size();
}

Snapshot* SnapshotMgr::GetWriteableSnapshot(int64_t snapshot_id) {
  auto it = snapshot_id2writeable_snapshot_.find(snapshot_id);
  if (it == snapshot_id2writeable_snapshot_.end()) {
    std::string snapshot_root_path = JoinPath(
        model_save_snapshots_path_, "snapshot_" + std::to_string(snapshot_id));
    OF_ONCE_GUARD(snapshot_root_path,
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
