#include "oneflow/core/persistence/snapshot_manager.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"

namespace oneflow {

void SnapshotManager::InitFromPlan(const Plan& plan) {
  model_save_snapshots_path_ = plan.job_desc().model_save_snapshots_path();
  tensorflow::Env* env = tensorflow::Env::Default();
  if (env->IsDirectory(model_save_snapshots_path_).code() != tensorflow::error::OK) {
    TF_CHECK_OK(env->CreateDir(model_save_snapshots_path_));
  }
  std::vector<std::string> result;
  TF_CHECK_OK(env->GetChildren(model_save_snapshots_path_, &result));
  CHECK_EQ(result.size(), 0);
  load_snapshot_ptr_ = new Snapshot(plan.job_desc().model_load_snapshot_path());
}

const Snapshot* SnapshotManager::GetWriterSnapshotFromSnapshotId(uint64_t snapshot_id) {
  if (snapshot_id2snapshot_.find(snapshot_id) == snapshot_id2snapshot_.end()) {
    std::string snapshot_root_path = tensorflow::io::JoinPath(
        model_save_snapshots_path_, "snapshot_" + std::to_string(snapshot_id));
    tensorflow::Env* env = tensorflow::Env::Default();
    TF_CHECK_OK(env->CreateDir(snapshot_root_path));
    CHECK(snapshot_id2snapshot_.emplace(
        snapshot_id, new Snapshot(snapshot_root_path)).second);
  }
  return snapshot_id2snapshot_.at(snapshot_id);
}

}  // namespace oneflow
